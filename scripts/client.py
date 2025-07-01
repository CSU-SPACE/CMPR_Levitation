#! coding: utf-8
from dataclasses import dataclass
import hashlib
import os
from typing import List
import xml.etree.ElementTree as ET
import requests
from requests.auth import AuthBase, HTTPBasicAuth
from datetime import datetime, timezone, timedelta
import time
from threading import Lock
import json
import copy

# 禁用安全请求警告
requests.packages.urllib3.disable_warnings()

# 配置
_SERVER_ADDR = "data.msadc.cn"
_ACCOUNT = "15694879346"
_PWD = "msadc2024"

@dataclass
class Info(object):
    name: str  # 文件（夹）完整路径
    size: int  # 文件为真实大小，文件夹为 -1
    is_dir: bool  # 是否是文件夹

    doc_id: str = ""  # 文件（夹）的 id，内部使用
    rev: str = ""  # 文件（夹）版本号，内部使用


class File(object):
    """
    AnyShare 中的文件
    """

    def __init__(self, docid, name, rev=None, client=None, *args, **kwargs):
        """
        文件初始化
        """
        self.docid = docid
        self.rev = rev
        self.name = name
        self.__client = client  # 保存Client实例的引用
        self._info = None
        self._verify = False

        self._retry_times = 24  # 重试次数

    def get_content(self, max_size=100 * 1024 * 1024) -> bytes:
        """
        获取文件内容 *1
        :param max_size: 可下载到内存的最大文件大小，默认为 100MB， 防止内存溢出
        :return: 文件内容（bytes），如果用户没有权限，应该返回null
        """
        # 检验大小
        size = self.get_info().size
        if size > max_size:
            raise Exception(f"文件过大： {size} > {max_size}")
        stream = self._get_stream()
        content = b""
        for v in stream:
            content += v
        return content

    def get_download_request(self):
        for i in range(self._retry_times):  # 重试
            try:
                res = self.__client.api_manager.request(
                    self.__client._server_addr + "/api/efast/v1/file/osdownload",
                    payload={
                        "docid": self.docid,
                        "rev": self.rev,
                        "authtype": "QUERY_STRING",
                    },
                )
                return res["authrequest"]
            except Exception as e:
                if i == self._retry_times - 1:
                    raise e
                print(f"retry osdownload {self.docid}-{self.rev}", e)
                time.sleep(5)
                pass

    def _get_stream(self):
        """
        返回文件流
        """
        file_size = self.get_info().size
        chunk_size = 128 * 1024 * 1024  # 128MB
        total_saved = 0
        for start in range(0, file_size, chunk_size):
            end_byte = start + chunk_size - 1
            if end_byte >= file_size:
                end_byte = file_size - 1
            size_saved = start
            chunk_saved = 0
            auth_request = self.get_download_request()
            for i in range(self._retry_times):  # 重试
                if auth_request is None:
                    auth_request = self.get_download_request()

                try:
                    response = self._oss_request(
                        auth_request,
                        headers={"Range": f"bytes={size_saved}-{end_byte}"},
                        stream=True,
                    )
                    for vv in response.iter_content(chunk_size=1024 * 1024):
                        yield vv
                        size_saved += len(vv)
                        total_saved += len(vv)
                        chunk_saved += len(vv)
                    if chunk_saved != end_byte - start + 1:
                        print(
                            f"chunk_saved {chunk_saved} != {end_byte - start + 1}, bytes={size_saved}-{end_byte}, {auth_request[1]}"
                        )
                    break
                except Exception as e:
                    if i == self._retry_times - 1:
                        raise e
                    print(f"retry oss_request {size_saved}-{end_byte}", e)
                    if "<Code>RequestTimeTooSkewed</Code>" in str(e):
                        # 下载链接过期，需要重新续期
                        auth_request = None
                    time.sleep(5)
        assert (
            total_saved == self.get_info().size
        ), f"size error {total_saved} {self.get_info().size}"

    # def get_xml_content(self) -> bytes:
    def get_xml_content(self) -> ET.Element:
        """
        获取同名对应的xml文件全部内容
        :return: 返回xml.etree.ElementTree 对象，方便用户直接获取xml文件的内容  如果用户没有权限，那么返回空值
        """
        try:
            res = self.__client.api_manager.request(
                self.__client._server_addr + "/api/dataservice/v1/osdownloadxml",
                payload={
                    "docid": self.docid,
                    "rev": self.rev,
                    "authtype": "QUERY_STRING",
                },
            )

            auth_request = res["authrequest"]
            response = self._oss_request(auth_request)
            # 解析响应内容为 XML
            root = ET.fromstring(response.content.strip(b"\0").strip())
            return root
        except APIResponseError:
            return

    def get_info(self) -> Info:
        """
        获取文件信息，如大小、名称等
        :return: 文件信息，
        """
        if self._info:
            return self._info
        res = self.__client.api_manager.request(
            self.__client._server_addr + "/api/efast/v1/file/attribute",
            payload={
                "docid": self.docid,
            },
        )
        res2 = self.__client.api_manager.request(
            self.__client._server_addr + "/api/efast/v1/file/convertpath",
            payload={
                "docid": self.docid,
            },
        )
        abs_name = res2["namepath"]
        file_info = Info(
            name=abs_name,
            size=res["size"],
            is_dir=False,
            doc_id=res["id"],
            rev=res["rev"],
        )
        self._info = file_info
        return file_info

    def save_local(self, name=None):
        """
        把文件保存到本地
        :param name: 保存的文件名，默认为当前文件名
        """
        if not name:
            name = self.name
        with open(name, "wb") as fp:
            self.download_fileobj(fp)

    def get_url(self) -> str:
        """
        获取文件下载链接
        :return: 文件下载链接
        """
        res = self.__client.api_manager.request(
            self.__client._server_addr + "/api/efast/v1/file/osdownload",
            payload={
                "docid": self.docid,
                "rev": self.rev,
                "authtype": "QUERY_STRING",
            },
        )
        return res['authrequest'][1]

    def save_curl(self, name=None, max_retries=100, check_size=True):
        """
        使用curl一次性下载文件，支持断点续传和失败重试
        :param check_size: 检查下载后的文件大小是否与服务器提供的一致
        :param max_retries: 最大重试次数
        :param name: 保存的文件名，默认为当前文件名
        """
        assert os.system('curl --version') == 0, '无法使用系统上的 curl，请安装 curl。'
        if not name:
            name = self.name
        # 必须加'Accept-Encoding: gzip, deflate, br, zstd'请求头，否则下载会失败
        command = f'curl -o \'{name}\' -C - --create-dirs -H \'Accept-Encoding: gzip, deflate, br, zstd\' \'{self.get_url()}\''
        for i in range(max_retries + 1):
            return_code = os.system(command)
            if return_code == 0:
                if check_size:
                    assert os.path.getsize(
                        name) == self.get_info().size, f'已下载的 {self.name} 文件大小与服务器提供的大小不一致'
                break
        else:
            raise Exception(f'在重试 {max_retries} 次后，文件 {self.name} 仍然下载失败')

    def download_fileobj(self, zip_stream):
        """
        把文件保存到 writer 中，可用于生成 zip 文件
        :param zip_stream: 支持 write 方法的对象，如 io.BytesIO()
        """
        for v in self._get_stream():
            zip_stream.write(v)

    def set_labels(self, labels: List[str]):
        """
        给文件打标签，管理员、操作员等有权限的用户可用
        :param labels: 标签列表
        """
        if not isinstance(labels, list):
            raise Exception("Parameter format error")
        item_id = self.docid.split("/")[-1]
        self.__client.api_manager.request(
            self.__client._server_addr +
            f"/api/metadata/v1/item/{item_id}/tag",
            payload=labels,
        )

    def del_labels(self, labels: List[str]):
        """
        文件删标签，管理员、操作员等有权限的用户可用
        :param labels: 标签列表
        """
        if not isinstance(labels, list):
            raise Exception("Parameter format error")
        for label in labels:
            item_id = self.docid.split("/")[-1]
            self.__client.api_manager.request(
                self.__client._server_addr
                + f"/api/metadata/v1/item/{item_id}/tag/{label}",
                method="delete",
            )

    def _oss_request(self, auth_request, file_content=b"", **kwargs):
        headers = kwargs.pop("headers", {})
        for item in auth_request[2:]:
            k, v = item.split(":", 1)
            headers[k] = v.strip()
        req_url = auth_request[1]
        req_method = auth_request[0]
        res = requests.request(
            req_method,
            req_url,
            headers=headers,
            data=file_content,
            timeout=60,
            verify=self._verify,
            **kwargs,
        )
        if res.status_code >= 300 or res.status_code < 200:
            raise Exception(f"{res.status_code} oss request err: {res.text}")
        return res

    def __repr__(self):
        return f"File(docid={self.docid}, rev={self.rev}, name={self.name})"


class Folder(object):
    """
    AnyShare 中的文件夹
    """

    def __init__(self, docid, name=None, rev=None, client=None, *args, **kwargs):
        """
        文件夹初始化
        """
        super().__init__(*args, **kwargs)
        self.docid = docid
        self.rev = rev
        self.name = name
        self.__client = client

        self._info = None

    def get_sub_folders(self, by="", sort="asc") -> List:
        """
        获取子文件夹
        return List[Folder]
        """
        folders_list = []
        res = self.__client.api_manager.request(
            self.__client._server_addr + "/api/efast/v1/dir/list",
            payload={
                "docid": self.docid,
                "by": by,  # string 指定按哪个字段排序 若不指定，默认按docid升序排序
                "sort": sort,  # asc，升序desc，降序 默认为升序
            },
        )
        dirs = res["dirs"]
        for dir in dirs:
            dir_docid = dir["docid"]
            dir_rev = dir.get("rev")
            dir_name = dir.get("name")
            folders_list.append(
                Folder(
                    docid=dir_docid, rev=dir_rev, name=dir_name, client=self.__client
                )
            )

        return folders_list

    def get_sub_files(self, by="", sort="asc") -> List[File]:
        """
        获取子文件
        """
        file_list = []
        res = self.__client.api_manager.request(
            self.__client._server_addr + "/api/efast/v1/dir/list",
            payload={
                "docid": self.docid,
                "by": by,  # string 指定按哪个字段排序 若不指定，默认按docid升序排序
                "sort": sort,  # asc，升序desc，降序 默认为升序
            },
        )
        files = res.get("files", [])
        for item in files:
            docid = item["docid"]
            rev = item.get("rev")
            name = item.get("name", "")
            file_list.append(
                File(docid=docid, name=name, rev=rev, client=self.__client)
            )
        return file_list

    def get_info(self) -> Info:
        """
        获取文件夹信息，如大小、名称等
        :return: 文件夹信息
        """
        if self._info:
            return self._info
        res = self.__client.api_manager.request(
            self.__client._server_addr + "/api/efast/v1/dir/attribute",
            payload={
                "docid": self.docid,
            },
        )
        dir_info = res

        res2 = self.__client.api_manager.request(
            self.__client._server_addr + "/api/efast/v1/file/convertpath",
            payload={
                "docid": self.docid,
            },
        )
        abs_name = res2["namepath"]

        self._info = Info(
            name=abs_name,
            size=-1,
            is_dir=True,
            doc_id=dir_info["id"],
            rev=dir_info["rev"],
        )
        return self._info

    def set_labels(self, labels: List[str]):
        """
        给文件夹打标签，管理员、操作员等有权限的用户可用
        :param labels: 标签列表
        """
        if not isinstance(labels, list):
            raise Exception("Parameter format error")
        item_id = self.docid.split("/")[-1]
        self.__client.api_manager.request(
            self.__client._server_addr +
            f"/api/metadata/v1/item/{item_id}/tag",
            payload=labels,
        )

    def del_labels(self, labels: List[str]):
        """
        删标文件夹标签，签管理员、操作员等有权限的用户可用
        :param labels: 标签列表
        """
        if not isinstance(labels, list):
            raise Exception("Parameter format error")
        for label in labels:
            item_id = self.docid.split("/")[-1]
            self.__client.api_manager.request(
                self.__client._server_addr
                + f"/api/metadata/v1/item/{item_id}/tag/{label}",
                method="delete",
            )

    def __repr__(self):
        return f"Folder(docid={self.docid}, rev={self.rev})"


class APIManage(requests.Session):
    def __init__(self, auth=None):
        super().__init__()  # 初始化父类
        self.auth = auth
        self.verify = False

    def check_resp(self, resp):
        """
        检测参数是否异常
        """
        resp.close()  # close connection
        if not resp.text:
            return
        try:
            result = resp.json()
        except Exception as e:
            raise Exception(
                f"APIManage.check_resp error: {e}, code: {resp.status_code}, detail: {resp.text}"
            )

        if "code" in result:
            raise APIResponseError(
                resp.url, result["code"], result["message"], result["cause"]
            )
        return result

    def request(self, url, params={}, payload={}, method="post", headers={}, **kwargs):
        resp = super().request(
            method,
            url,
            params=params,
            json=payload,
            headers=headers,
            timeout=60,
            verify=self.verify,
            **kwargs,
        )
        return self.check_resp(resp)


class Client(object):
    """
    AnyShare 客户端
    """

    def __init__(self, protocol="https", *args, **kwargs):
        """
        初始化客户端
        """
        super().__init__(*args, **kwargs)
        self._protocol = protocol
        self._server_addr = self._get_server_addr()
        self._hj_username = _ACCOUNT
        self._hj_pwd = _PWD
        self._token = None
        self._token_expire = time.time()
        self._lock = Lock()
        if len(self._hj_username) == 36:
            # as正常应用账户
            self._auth = Client2Auth(
                self._server_addr, self._hj_username, self._hj_pwd
            )  # 中间服务地址，换取token
        else:
            # 和鲸账户
            self._auth = ClientAuth(
                self._server_addr, self._hj_username, self._hj_pwd
            )  # 中间服务地址，换取token
        self.api_manager = APIManage(auth=self._auth)

        # 获取数据字典
        self._DATA_DICT = {"SCID_DICT": {}, "APID_DICT": {}, "LV_DICT": {}}
        self._data_product_dict()

    def _data_product_dict(self):
        # 数据字典-舱段， dict_id根据环境修改
        res = self.api_manager.request(
            self._server_addr
            + "/api/ecotag/v1/dict-item?dict_id=01HRP7BGJV2YH6X0XG32QFHJD0",
            method="get",
        )
        _scid_dict = {}
        for option in res:
            value = option["value"]
            id = option["id"]
            children = option.get("children")
            _children = copy.copy(children)
            _scid_dict.update({value: id})
            if not _children:
                continue
            while True:
                obj = _children.pop()  # []
                children2 = obj.get("children")
                _scid_dict.update({obj["value"]: obj["id"]})
                if children2:
                    _children.extend(children2)
                if not _children:
                    break

        self._DATA_DICT["SCID_DICT"].update(_scid_dict)

        # 数据字典-载荷， dict_id根据环境修改
        res = self.api_manager.request(
            self._server_addr
            + "/api/ecotag/v1/dict-item?dict_id=01HRP7H77GF75ZTX1DXSKN95PZ",
            method="get",
        )
        _apid_dict = {}
        for option in res:
            value = option["value"]
            id = option["id"]
            children = option.get("children")
            _children = copy.copy(children)
            _apid_dict.update({value: id})
            if not _children:
                continue
            while True:
                obj = _children.pop()  # []
                children2 = obj.get("children")
                _apid_dict.update({obj["value"]: obj["id"]})
                if children2:
                    _children.extend(children2)
                if not _children:
                    break
        self._DATA_DICT["APID_DICT"].update(_apid_dict)

        # 数据字典-数据级别， dict_id根据环境修改
        res = self.api_manager.request(
            self._server_addr
            + "/api/ecotag/v1/dict-item?dict_id=01HRP7CCWVNERN8512W9YZ6ANP",
            method="get",
        )
        _lv_dict = {i["value"]: i["id"] for i in res}
        self._DATA_DICT["LV_DICT"].update(_lv_dict)

    def _get_server_addr(self):
        return "{0}://{1}".format(self._protocol, _SERVER_ADDR)

    def get_experiments(self) -> List[Folder]:
        """
        获取有下载权限的实验
        :return: 实验项目列表
        """
        products_list = []
        res = self.api_manager.request(
            self._server_addr + "/api/dataservice/v1/experiments", method="get"
        )
        products = res.get("experiments", [])
        for item in products:
            docid = item["id"]
            rev = item.get("rev", "")
            name = item.get("name", "")
            products_list.append(
                Folder(docid=docid, name=name, rev=rev, client=self))
        return products_list

    def search_file_by_label(
        self, folder: Folder, labels: List[str], type="all", rows=100
    ) -> List[File]:
        """
        通过标签搜索文件
        :return : 文件列表
        '"""
        file_list = []
        docid = folder.docid
        res = self.api_manager.request(
            self._server_addr + "/api/ecosearch/v1/file-search",
            payload={
                "type": type,
                "start": 0,
                "rows": rows,
                "range": [docid + "/*"],
                "custom": [{"key": "tags", "type": "multiselect", "value": labels}],
                "dimension": ["file"],
            },
        )
        files = res.get("files", [])
        for item in files:
            docid = item["doc_id"]
            rev = item.get("rev")
            name = item.get("basename", "") + item.get("extension", "")
            file_list.append(
                File(docid=docid, name=name, rev=rev, client=self))
        return file_list

    def search_folder_by_label(
        self, labels: List[str], type="all", rows=50
    ) -> List[Folder]:
        """
        获取满足labels标签的实验项目的Folder
        :return : 实验项目列表，有些可能没有权限，但是也会返回
        """
        folder_list = []
        res = self.api_manager.request(
            self._server_addr + "/api/ecosearch/v1/file-search",
            payload={
                "type": type,
                "start": 0,
                "rows": rows,
                "custom": [{"key": "tags", "type": "multiselect", "value": labels}],
                "dimension": ["folder"],
            },
        )
        files = res.get("files", [])
        for item in files:
            docid = item["doc_id"]
            rev = item.get("rev")
            name = item.get("basename", "")
            folder_list.append(
                Folder(docid=docid, name=name, rev=rev, client=self))
        return folder_list

    def search(
        self,
        scid,
        apid=None,
        dpid=None,
        level=None,
        startTime=None,
        endTime=None,
        mode=0,
        type="all",
        rows=50,
        start=0,
        path="数据服务",
    ) -> List[File]:
        """
        搜索接口
        只可以检索数据服务目录下文件
        scid str 航天器标识，不允许为空
        apid: str 载荷标识，允许为空
        dpid，str 数据产品标识，可以为空，为空代表无条件
        level: 0,1,2,3,4....，可以为空，为空代表无条件
        startTime 开始时间 20240530000000，可以为空， 14位日期
        endTime，结束时间， 可以为空，为空以当前时间
        start 开始页数；第一页从0开始，往后的页传入回复消息中的next；
        path 检索目录，绝对路径
        mode: 0采集时间，1归档时间，不可以为空，必须为0或1， 为0时，startTime和EndTime代表采集开始结束，为1，代表归档开始和结束时间。
         :return: 返回产品文件名称满足apid、dpid、level条件，且 采集/归档的开始、结束时间中任意一个落在[startTime,endTime]中，且用户具有下载权限的，所有文件，即为满足检索要求，返回文件列表
                  next, 是否有下一页 0没有， 其他数值有，并且在search方法中以next传入
        """
        file_list = []
        custom = []
        sub_custom = []
        and_search_condition = []
        or_search_condition = []
        startTime = self.date2stamp(startTime)
        endTime = self.date2stamp(endTime)
        scid_2asid = self._DATA_DICT["SCID_DICT"].get(scid)
        res = self.api_manager.request(
            self._server_addr + "/api/efast/v1/file/getinfobypath",
            payload={"namepath": path},
        )
        if "docid" not in res:
            raise Exception("not exist addr")
        path_docid = res["docid"]

        if not scid_2asid:
            raise Exception("not exist scid")
        scid_search_condition = {
            "key": "SCID",
            "value": [scid_2asid],
            "mode": "=",
            "scope": "aishu",
            "template": "data_product_file",
            "type": "enum",
        }
        or_scid_search_condition = copy.deepcopy(scid_search_condition)
        if apid:
            apid_2asid = self._DATA_DICT["APID_DICT"].get(apid)
            if not apid_2asid:
                raise Exception("not exist apid")
            and_search_condition.append(
                {
                    "template": "data_product_file",
                    "scope": "aishu",
                    "key": "APID",
                    "mode": "=",
                    "type": "enum",
                    "value": [apid_2asid],
                }
            )
        if dpid:
            and_search_condition.append(
                {
                    "template": "data_product_file",
                    "scope": "aishu",
                    "key": "DPID",
                    "mode": "exact",
                    "type": "string",
                    "value": dpid,
                }
            )

        if level is not None:
            level = str(level)
            if not level.startswith("L"):
                level = f"L{level}"
            LV_2asid = self._DATA_DICT["LV_DICT"].get(level)
            if not LV_2asid:
                raise Exception("not exist LV")
            and_search_condition.append(
                {
                    "template": "data_product_file",
                    "scope": "aishu",
                    "key": "ProdLevel",
                    "mode": "=",
                    "type": "enum",
                    "value": [LV_2asid],
                }
            )

        if mode == 0:
            _or_search_sub_and_condition = copy.deepcopy(and_search_condition)
            if startTime and endTime:
                # If （文件采集开始>=搜索开始&&文件采集开始<=搜索结束）or （文件采集结束>=搜索开始&&文件采集结束<=搜索结束）
                if len(str(startTime)) != 13 or len(str(endTime)) != 13:
                    raise Exception("Pass in a valid 13-bit timestamp")
                and_search_condition.append({
                    "template": "data_product_file",
                    "scope": "aishu",
                    "key": "TSTR",
                    "mode": ">=",
                    "type": "date",
                    "value": [int(startTime)],
                })
                and_search_condition.append({
                    "template": "data_product_file",
                    "scope": "aishu",
                    "key": "TSTR",
                    "mode": "<=",
                    "type": "date",
                    "value": [int(endTime)],
                })
                _or_search_sub_and_condition.append({
                    "template": "data_product_file",
                    "scope": "aishu",
                    "key": "TEND",
                    "mode": ">=",
                    "type": "date",
                    "value": [int(startTime)],
                })
                _or_search_sub_and_condition.append({
                    "template": "data_product_file",
                    "scope": "aishu",
                    "key": "TEND",
                    "mode": "<=",
                    "type": "date",
                    "value": [int(endTime)],
                })
                _or_search_sub_and_condition.append(or_scid_search_condition)
                _or_search_sub_and_condition_key = _or_search_sub_and_condition.pop(
                    0)
                _or_search_sub_and_condition_key.update(
                    {"and": _or_search_sub_and_condition})
                or_search_condition.append(_or_search_sub_and_condition_key)

            if startTime and not endTime:
                if len(str(startTime)) != 13:
                    raise Exception("Pass in a valid 13-bit timestamp")
                and_search_condition.append(
                    {
                        "template": "data_product_file",
                        "scope": "aishu",
                        "key": "TSTR",
                        "mode": ">=",
                        "type": "date",
                        "value": [int(startTime)],
                    }
                )
                _or_search_sub_and_condition.append({
                    "template": "data_product_file",
                    "scope": "aishu",
                    "key": "TEND",
                    "mode": ">=",
                    "type": "date",
                    "value": [int(startTime)],
                })
                _or_search_sub_and_condition.append(or_scid_search_condition)
                _or_search_sub_and_condition_key = _or_search_sub_and_condition.pop(
                    0)
                _or_search_sub_and_condition_key.update(
                    {"and": _or_search_sub_and_condition})
                or_search_condition.append(_or_search_sub_and_condition_key)

            if endTime and not startTime:
                if len(str(endTime)) != 13:
                    raise Exception("Pass in a valid 13-bit timestamp")
                and_search_condition.append(
                    {
                        "template": "data_product_file",
                        "scope": "aishu",
                        "key": "TEND",
                        "mode": "<=",
                        "type": "date",
                        "value": [int(endTime)],
                    }
                )
                _or_search_sub_and_condition.append({
                    "template": "data_product_file",
                    "scope": "aishu",
                    "key": "TEND",
                    "mode": "<=",
                    "type": "date",
                    "value": [int(endTime)],
                })
                _or_search_sub_and_condition.append(or_scid_search_condition)
                _or_search_sub_and_condition_key = _or_search_sub_and_condition.pop(
                    0)
                _or_search_sub_and_condition_key.update(
                    {"and": _or_search_sub_and_condition})
                or_search_condition.append(_or_search_sub_and_condition_key)

        else:
            if startTime and endTime:
                if len(str(startTime)) != 13 or len(str(endTime)) != 13:
                    raise Exception("Pass in a valid 13-bit timestamp")
                sub_custom.append(
                    {
                        "key": "created_at",
                        "type": "date",
                        "value": [int(startTime) * 1000, int(endTime) * 1000],
                    }
                )
            if startTime and not endTime:
                if len(str(startTime)) != 13:
                    raise Exception("Pass in a valid 13-bit timestamp")
                sub_custom.append(
                    {
                        "key": "created_at",
                        "type": "date",
                        "value": [int(startTime) * 1000, -1],
                    }
                )
            if endTime and not startTime:
                if len(str(endTime)) != 13:
                    raise Exception("Pass in a valid 13-bit timestamp")
                sub_custom.append(
                    {
                        "key": "created_at",
                        "type": "date",
                        "value": [-1, int(endTime) * 1000],
                    }
                )

        if and_search_condition:
            scid_search_condition.update({"and": and_search_condition})
        if or_search_condition:
            scid_search_condition.update({"or": or_search_condition})

        custom.append(scid_search_condition)
        data = {
            "type": type,
            "start": start,
            "rows": rows,
            "dimension": ["file"],
            "custom": custom,
            "range": [path_docid + "/*"],
        }
        print(json.dumps(data))
        res = self.api_manager.request(
            self._server_addr + "/api/ecosearch/v1/file-search", payload=data
        )
        files = res.get("files", [])
        next = res.get("next", [])
        for item in files:
            docid = item["doc_id"]
            rev = item.get("rev")
            name = item.get("basename", "") + item.get("extension", "")
            file_list.append(
                File(docid=docid, name=name, rev=rev, client=self))
        if not files:
            return file_list, 0
        else:
            return file_list, next

    def search_keyword(self, addr, keyword, rows=50, start=0) -> List[File]:
        """
        搜索接口
        只可以检索数据服务目录下文件
        addr str 文件所在文件夹的路径，不允许为空
        keyword: str 检索的关键字，不允许为空
        :return: 返回产品文件名称满足文件路径、关键字且用户具有下载权限的，所有文件，即为满足检索要求，返回文件列表
                  next, 是否有下一页 0没有， 其他数值有，并且在search方法中以next传入
        """
        file_list = []
        custom = []
        # 获取文件docid
        res = self.api_manager.request(
            self._server_addr + "/api/efast/v1/file/getinfobypath",
            payload={"namepath": addr},
        )
        if "docid" not in res:
            raise Exception("not exist addr")
        docid = res["docid"]
        data = {
            "keyword": keyword,
            "start": start,
            "range": [docid + "/*"],
            "dimension": ["basename"],
            "type": "doc",
            "rows": rows,
            "quick_search": True,
        }
        res = self.api_manager.request(
            self._server_addr + "/api/ecosearch/v1/file-search", payload=data
        )
        files = res.get("files", [])
        next = res.get("next", [])
        for item in files:
            docid = item["doc_id"]
            rev = item.get("rev")
            name = item.get("basename", "") + item.get("extension", "")
            file_list.append(
                File(docid=docid, name=name, rev=rev, client=self))
        if not files:
            return file_list, 0
        else:
            return file_list, next

    def date2stamp(self, datestring):
        # 14位日期字符串转13位时间戳
        if not isinstance(datestring, str) or len(datestring) != 14:
            return
        # 解析14位的日期字符串，假设它是 UTC+8 时间
        dt = datetime.strptime(datestring, "%Y%m%d%H%M%S")
        # 明确将时间设置为 UTC+8 时区
        tz_offset = timedelta(hours=8)
        dt_utc_plus_8 = dt.replace(tzinfo=timezone(tz_offset))
        # 获取 UTC+8 时区的时间戳（秒级别），然后转换为毫秒级
        timestamp = str(int(dt_utc_plus_8.timestamp() * 1000))  # 转换为毫秒级
        return timestamp


class ClientAuth(AuthBase):
    _except_token_valid = Exception("token valid")

    def __init__(self, base_url, username, pwd):
        self._base_url = base_url
        self.hj_username = username
        self.hj_pwd = pwd
        self._token = None
        self._token_expire = time.time()
        self._lock = Lock()

    def _refresh_token(self):
        self._lock.acquire()
        try:
            if self._token_expire - time.time() > 120:
                raise self._except_token_valid
            session = requests.session()
            session.verify = False
            url_str = f"{self._base_url}/api/authingcallback/v1/call?pwd_auth=1"
            res = session.post(
                url_str,
                json={
                    "username": self.hj_username,
                    "pwd": self.hj_pwd,
                },
            )
            if res.status_code != 200:
                raise Exception(
                    f"post {url_str} err: code: {res.status_code}, detail: {res.text}"
                )
            res = res.json()
            self._token = "Bearer " + res["access_token"]
            self._token_expire = time.time() + (
                res.get("expires_in") or res["expirses_in"]
            )

        except Exception as e:
            if e is not self._except_token_valid:
                raise e
        finally:
            self._lock.release()

    def __call__(self, r):
        if self._token_expire - time.time() < 120:
            self._refresh_token()
        if self._token:
            r.headers["Authorization"] = self._token
        return r


class Client2Auth(AuthBase):
    _except_token_valid = Exception("token valid")

    def __init__(self, base_url, client_id, client_secret):
        self._base_url = base_url
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = None
        self._token_expire = time.time()

        self._lock = Lock()

    def _refresh_token(self):
        self._lock.acquire()
        try:
            if self._token_expire - time.time() > 120:
                raise self._except_token_valid
            session = requests.session()
            session.verify = False
            url_str = f"{self._base_url}/oauth2/token"
            res = session.post(
                url_str,
                auth=HTTPBasicAuth(self.client_id, self.client_secret),
                data={
                    "grant_type": "client_credentials",
                    "scope": "all",
                },
            )
            if res.status_code != 200:
                raise Exception(
                    f"post {url_str} err: code: {res.status_code}, detail: {res.text}"
                )
            res = res.json()
            self._token = "Bearer " + res["access_token"]
            self._token_expire = time.time() + res["expires_in"]
        except Exception as e:
            if e is not self._except_token_valid:
                # logging.warning("refresh token err", exc_info=True)
                raise e
        finally:
            self._lock.release()

    def __call__(self, r):
        if not r.url.startswith(self._base_url):
            return r
        if self._token_expire - time.time() < 120:
            self._refresh_token()
        if self._token:
            r.headers["Authorization"] = self._token
        return r


class ASError(Exception):
    def __repr__(self):
        return self.__str__()


class APIResponseError(ASError):
    def __init__(self, url, errcode, errmsg, causemsg):

        if isinstance(url, bytes):
            self.url = errmsg.encode("utf-8")
        else:
            self.url = url

        self.errcode = str(errcode)
        if isinstance(errmsg, bytes):
            self.errmsg = errmsg.encode("utf-8")
        else:
            self.errmsg = errmsg

        if isinstance(causemsg, bytes):
            self.cause_msg = causemsg.encode("utf-8")
        else:
            self.cause_msg = causemsg

        self.message = str(self)

    def __str__(self):
        return "Request %s failed. errcode: %s, errmsg: %s, causemsg:%s" % (
            self.url,
            self.errcode,
            self.errmsg,
            self.cause_msg,
        )


class Md5Writer:
    def __init__(self):
        self.md5 = hashlib.md5()

    def write(self, data):
        self.md5.update(data)

    def hexdigest(self):
        return self.md5.hexdigest()

    def __repr__(self):
        return self.hexdigest()


def time_str(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    asclient = Client()
    # file = File(
    #     "gns://86CF9578847048C0BD0D7E05F4E9809D/66688DAA20CE4CBBA89FE52E89C19C2D/FE2DA5C97CE64BF681551B427AE50CD6/CA7A648DE9424819B49372223587C1BF/E40C9FB7241F45FEA7E243A178F8D5AB/77109D5C255047518F0C8B02EDC5563F",
    #     "test.zip",
    #     client=asclient,
    # )
    # for i in range(1):
    #     md5 = Md5Writer()
    #     print(f"[{time_str()}] start: {i + 1}")
    #     try:
    #         file.download_fileobj(md5)
    #     except Exception as e:
    #         print(f"[{time_str()}] err: {i + 1}, detail: {e}")
    #         continue
    #     print(f"[{time_str()}] done: {i + 1}, md5: {md5}")

    # 检索数据服务目录
    start = 0
    file_list = []
    # 元数据
    print("开始测试搜索")
    while True:
        files, next = asclient.search(
            scid="TGTH",
            apid="CMPR",
            dpid="",
            level=1,
            startTime="20220823000000",
            endTime="20220824000000",
            mode=0,
            start=start
        )
        if not next:
            break
        file_list.extend(files)
        start = next
    for file in file_list:
        print(file.name)
    print(len(file_list))
