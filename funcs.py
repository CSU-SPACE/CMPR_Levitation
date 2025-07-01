import os
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from io import BytesIO
from typing import List, Union

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, dates as mdates
from pandas import DataFrame
from scipy.signal import savgol_filter
from scipy.stats import linregress
from tqdm import tqdm

from client_zhengshi_v17 import Client

plot_time_format = '%H:%M:%S'
dcm1_scale = 0.0016506116493361
dcm2_scale = 0.001657290624493


def sine_wave(x, A, B, C, D):
    return D + A * np.sin(B * x + C)


def laser(image) -> int:
    # 自适应阈值参数
    BLACK_THRESH = 46  # 黑色阈值，可根据实际情况调整
    MIN_CIRCLE_AREA = 1000  # 最小圆面积阈值
    CIRCLE_RATIO = 0.9  # 圆形度阈值（面积/包围圆面积）
    # 亮点检测参数
    SPOT_THRESH = 100  # 亮点亮度阈值
    MIN_SPOT_AREA = 2  # 最小亮点面积（像素）
    TOPHAT_SIZE = (3, 3)  # 顶帽运算核大小

    h, w, _ = image.shape
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化处理（反色处理，因为要检测黑色区域）
    _, thresh = cv2.threshold(gray, BLACK_THRESH, 255, cv2.THRESH_BINARY_INV)

    # 形态学操作消除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选圆形轮廓
    black_circles = []
    for cnt in contours:
        # 面积过滤
        area = cv2.contourArea(cnt)
        if area < MIN_CIRCLE_AREA:
            continue

        # 圆形度检测
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        radius = int(radius)
        circle_area = np.pi * (radius ** 2)
        ratio = area / circle_area

        if ratio > CIRCLE_RATIO:
            black_circles.append((int(x), int(y), radius))

    result_img = gray.copy()

    ##########################
    # 第二步：检测内部亮点 #
    ##########################
    for (x, y, r) in black_circles:
        # 绘制圆形轮廓

        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)

        masked_img = cv2.bitwise_and(result_img, result_img, mask=mask)
        _, thresh = cv2.threshold(masked_img, 100, 255, cv2.THRESH_BINARY)

        # 查找连通区域
        # 连通域查找
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        # 统计符合条件的区域数量
        count = 0
        for i in range(1, num_labels):  # 跳过背景区域（标签0）
            area = stats[i, cv2.CC_STAT_AREA]
            if 1 < area < 100:

                # 提取当前区域的轮廓
                region_mask = (labels == i).astype(np.uint8)
                contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                valid_regions = []
                distance_threshold = r * 0.3  # 阈值设为半径的一定比例
                for cnt in contours:
                    # 计算中心
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # 计算距离
                    distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                    if distance <= distance_threshold:
                        count += 1
                        valid_regions.append((cx, cy))
        if count > 0:
            return count
    return 0


def calc_height(dcm1_zip_paths: List[str], dcm2_zip_paths: List[str]):
    total = 0
    start_lines = [0]
    for file in dcm1_zip_paths + dcm2_zip_paths:
        print(f'正在读取文件 {file}，大小 {format_size(os.path.getsize(file))}')
        try:
            with zipfile.ZipFile(file, 'r') as zf:
                image_count = len(zf.namelist()) // 2
                start_lines.append(start_lines[-1] + image_count)
                total += image_count
        except:
            print(f'{file} 已损坏')
    print(f'共 {total} 行高度数据')
    df = pd.DataFrame(columns=['unix时间', '高度1', '高度2'], index=range(total))

    height_config = [
        (dcm1_zip_paths, dcm1_scale, 1),
        (dcm2_zip_paths, dcm2_scale, 2),
    ]

    start_index = 0

    for zip_paths, scale, col in height_config:
        for file in zip_paths:
            try:
                with zipfile.ZipFile(file, 'r') as zf:
                    file_list = list(set(map(lambda x: os.path.splitext(x)[0], zf.namelist())))
                    print(f'文件 {file} 中包含 {len(file_list)} 张图像')
                    failed = 0
                    for index, name in tqdm(enumerate(file_list), total=len(file_list)):
                        try:
                            xml_file = BytesIO(zf.read(f'{name}.xml')).read()
                            root = ET.fromstring(xml_file.decode('utf-8'))
                            private_attribute = root.find('PrivateAttribute')
                            image_info = private_attribute.find('ImageInfo')
                            image_time = image_info.find('imgTime').text
                            date_str, ms = image_time.split('.')
                            time_object = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                            time_unix_seconds = (time_object.timestamp() + eval(f'0.{ms}')) * 1000
                            image = cv2.imdecode(np.frombuffer(zf.read(f'{name}.bmp'), np.uint8), cv2.IMREAD_COLOR)
                            # 转换为灰度图并二值化
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            # 应用高斯模糊
                            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                            # 使用Canny边缘检测
                            edged = cv2.Canny(blurred, 50, 150)

                            # 查找轮廓
                            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            for contour in contours:
                                # 尝试拟合椭圆，如果轮廓点太少可能无法拟合
                                if len(contour) >= 5:  # 至少需要5个点来拟合椭圆
                                    ellipse = cv2.fitEllipse(contour)
                                    (center, axes, angle) = ellipse
                                    major_axis = max(axes)  # 长轴长度
                                    minor_axis = min(axes)  # 短轴长度
                                    if major_axis > 100:
                                        x, y, w, h = cv2.boundingRect(contour)
                                        # height = h * scale
                                        height = h
                                        new_line = [time_unix_seconds, None, None]
                                        new_line[col] = height
                                        df.loc[start_lines[start_index] + index] = new_line
                                        break
                            else:
                                print(name)
                                cv2.imshow("Result", image)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                raise ("未找到椭圆")
                        except:
                            failed += 1
                            pass
                    if failed > 0:
                        print(f'{file} 中有 {failed} 行数据处理失败')
            except:
                print(f'{file} 已损坏')
            finally:
                start_index += 1

    print('正在删除空行')
    df.dropna(subset=['高度1', '高度2'], how='all', inplace=True)
    df.dropna(subset=['unix时间'], how='all', inplace=True)
    df['unix时间'] = df['unix时间'].astype('float')
    df['高度1'] = df['高度1'].astype('float')
    df['高度2'] = df['高度2'].astype('float')
    print('正在合并重复的时间码')
    df = df.groupby('unix时间', as_index=False).max()
    print('正在按照时间码排序')
    df = df.sort_values(by=['unix时间'])
    df.reset_index(inplace=True)
    return df


def calc_density(dcm1_zip_paths: List[str], dcm2_zip_paths: List[str], mass: float) -> DataFrame:
    """
    计算密度
    :param dcm1_zip_paths: DCM1文件列表
    :param dcm2_zip_paths: DCM2文件列表
    :param mass:
    :return:
    """
    total = 0
    start_lines = [0]
    for file in dcm1_zip_paths + dcm2_zip_paths:
        print(f'正在读取文件 {file}，大小 {format_size(os.path.getsize(file))}')
        try:
            with zipfile.ZipFile(file, 'r') as zf:
                image_count = len(zf.namelist()) // 2
                start_lines.append(start_lines[-1] + image_count)
                total += image_count
        except:
            print(f'{file} 已损坏')
    print(f'共 {total} 行密度数据')
    df = pd.DataFrame(columns=['unix时间', '密度1', '密度1激光', '密度2', '密度2激光'], index=range(total))

    density_config = [
        (dcm1_zip_paths, dcm1_scale, 1, 2),
        (dcm2_zip_paths, dcm2_scale, 3, 4),
    ]

    start_index = 0

    for zip_paths, scale, density_col, laser_col in density_config:
        for file in zip_paths:
            try:
                with zipfile.ZipFile(file, 'r') as zf:
                    file_list = list(set(map(lambda x: os.path.splitext(x)[0], zf.namelist())))
                    print(f'文件 {file} 中包含 {len(file_list)} 张图像')
                    failed = 0
                    for index, name in tqdm(enumerate(file_list), total=len(file_list)):
                        try:
                            xml_file = BytesIO(zf.read(f'{name}.xml')).read()
                            root = ET.fromstring(xml_file.decode('utf-8'))
                            private_attribute = root.find('PrivateAttribute')
                            image_info = private_attribute.find('ImageInfo')
                            image_time = image_info.find('imgTime').text
                            date_str, ms = image_time.split('.')
                            time_object = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                            time_unix_seconds = (time_object.timestamp() + eval(f'0.{ms}')) * 1000
                            image = cv2.imdecode(np.frombuffer(zf.read(f'{name}.bmp'), np.uint8), cv2.IMREAD_COLOR)
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            gray = cv2.GaussianBlur(gray, (9, 9), 2)
                            # 应用霍夫圆变换
                            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                                       param1=100, param2=100, minRadius=0, maxRadius=0)
                            radius = 0
                            if circles is not None:
                                for x, y, r in circles[0, :]:
                                    radius = r

                            if radius > 0:
                                volume = 4 / 3 * np.pi * (radius * scale) ** 3
                                density = mass / volume
                                new_line = [time_unix_seconds, None, None, None, None]
                                new_line[density_col] = density
                                new_line[laser_col] = laser(image)
                                df.loc[start_lines[start_index] + index] = new_line
                        except:
                            failed += 1
                            pass
                    if failed > 0:
                        print(f'{file} 中有 {failed} 行数据处理失败')
            except:
                print(f'{file} 已损坏')
            finally:
                start_index += 1

    print('正在删除空行')
    df.dropna(subset=['密度1', '密度2'], how='all', inplace=True)
    df.dropna(subset=['unix时间'], how='all', inplace=True)
    df['unix时间'] = df['unix时间'].astype('float')
    df['密度1'] = df['密度1'].astype('float')
    df['密度2'] = df['密度2'].astype('float')
    print('正在合并重复的时间码')
    df = df.groupby('unix时间', as_index=False).max()
    print('正在按照时间码排序')
    df = df.sort_values(by=['unix时间'])
    df.reset_index(inplace=True)
    return df


def parse_csv_time_to_unix_mills(time_str: str) -> Union[float, None]:
    try:
        date_str_arr = time_str.strip().split('.')
        time_object = datetime.strptime(date_str_arr[0], '%Y-%m-%d %H:%M:%S')
        time_unix_mills = time_object.timestamp() * 1000
        if len(date_str_arr) > 1:
            time_unix_mills += int(date_str_arr[1])
        return time_unix_mills
    except:
        return None


pattern = re.compile(r'(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?')


def parse_time_string(time_str):
    # 使用正则表达式提取天、小时、分钟和秒
    match = pattern.match(time_str)

    if not match:
        return None

    days, hours, minutes, seconds = match.groups()
    days = int(days) if days else 0
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0

    # 计算总秒数
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds

    # 将总秒数转换为毫秒
    total_milliseconds = total_seconds * 1000

    return total_milliseconds


def temperature(swth_paths: List[str], dwth_paths: List[str]) -> DataFrame:
    """
    计算温度
    :param swth_paths: SWTH文件列表
    :param dwth_paths: DWTH文件列表
    :return:
    """
    total = 0
    data_per_line = 250
    start_lines = [0]

    for file in swth_paths + dwth_paths:
        csv = pd.read_csv(file)
        csv_lines = len(csv) * data_per_line
        start_lines.append(start_lines[-1] + csv_lines)
        total += csv_lines
    print(f'共 {total} 行温度数据')
    df = pd.DataFrame(columns=['unix时间', '单波长温度', '双波长温度'], index=range(total))

    start_index = 0

    csv_config = [
        (swth_paths, '单波长温度', 1),
        (dwth_paths, '双波长温度', 2),
    ]

    for file_list, title, col in csv_config:
        for file in file_list:
            csv = pd.read_csv(file)
            print(f'正在读取文件 {file}')
            for index, row in tqdm(csv.iterrows(), total=len(csv)):
                time = parse_csv_time_to_unix_mills(row['时间码'])
                if time is not None:
                    for i in range(data_per_line):
                        new_line = [time + i * 2, None, None]
                        new_line[col] = row[f'{title}数据{i + 1}']
                        df.loc[start_lines[start_index] + index * data_per_line + i] = new_line
            start_index += 1

    print('正在合并重复的时间码')
    df['unix时间'] = df['unix时间'].astype('float')
    df['单波长温度'] = df['单波长温度'].astype('float')
    df['双波长温度'] = df['双波长温度'].astype('float')
    df.dropna(how='all', inplace=True)
    df.dropna(how='all', subset=['unix时间'], inplace=True)
    df = df.groupby('unix时间', as_index=False).max()
    print('正在按照时间码排序')
    df = df.sort_values(by=['unix时间'])
    print('正在过滤负值')
    df['双波长温度'] = df['双波长温度'].apply(lambda x: x if x >= 0 else np.nan)
    df['单波长温度'] = df['单波长温度'].apply(lambda x: x if x >= 0 else np.nan)
    df.dropna(how='all', subset=['单波长温度', '双波长温度'], inplace=True)
    df.reset_index(inplace=True)
    return df


def power(phdp_paths: List[str]) -> DataFrame:
    total = 0
    package_per_line = 8
    data_per_package = 60
    pkgs_per_4s = 667
    time_delta_per_data = 4000 / (pkgs_per_4s * data_per_package)
    data_per_line = data_per_package * package_per_line
    start_lines = [0]

    for file in phdp_paths:
        csv = pd.read_csv(file)
        csv_lines = len(csv) * data_per_line
        start_lines.append(start_lines[-1] + csv_lines)
        total += csv_lines
    print(f'共 {total} 行功率数据')
    df = pd.DataFrame(columns=['文件时间', '毫秒时间', '功率'], index=range(total))

    start_index = 0

    for file in phdp_paths:
        csv = pd.read_csv(file)
        print(f'正在读取文件 {file}')
        for index, row in tqdm(csv.iterrows(), total=len(csv)):
            for i in range(package_per_line):
                if i == 0:
                    package_id = row['功率包序号']
                    time_base = parse_time_string(row['开始采集时间码秒计数(s)'])
                else:
                    package_id = row[f'功率包序号.{i}']
                    time_base = parse_time_string(row[f'开始采集时间码秒计数(s).{i}'])
                for j in range(data_per_package):
                    time_delta = ((package_id - 1) * data_per_package + j) * time_delta_per_data
                    if i == 0:
                        new_line = [row['开始采集时间码秒计数(s)'], time_base + time_delta, row[f'功率数据{j + 1}']]
                    else:
                        new_line = [row['开始采集时间码秒计数(s)'], time_base + time_delta,
                                    row[f'功率数据{j + 1}.{i}']]
                    df.loc[start_lines[start_index] + index * data_per_line + i * data_per_package + j] = new_line
        start_index += 1

    print('正在合并重复的时间码')
    df['毫秒时间'] = df['毫秒时间'].astype('float')
    df['功率'] = df['功率'].astype('float')
    df.dropna(how='all', inplace=True)
    df.dropna(how='all', subset=['毫秒时间'], inplace=True)
    df = df.groupby('毫秒时间', as_index=False).max()
    print('正在按照时间码排序')
    df = df.sort_values(by=['毫秒时间'])
    # print('正在过滤负值')
    # df['功率'] = df['功率'].apply(lambda x: x if x >= 0 else np.nan)
    df.dropna(how='all', subset=['功率'], inplace=True)
    df.reset_index(inplace=True)
    return df


def linear_fit(X, Y):
    """
    线性回归
    :param X:
    :param Y:
    :return:
    """
    slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    return slope, intercept


def format_unix_ms(time_ms: float):
    ms = int(time_ms % 1000)
    return datetime.fromtimestamp(time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S') + f'.{ms:03}'


def sliding_window_smooth(data, window_size):
    """
    使用滑动窗口平滑数据
    :param data: 输入数据
    :param window_size: 滑动窗口大小
    :return: 平滑后的数据
    """
    if window_size % 2 == 0:
        raise ValueError("窗口大小必须是奇数")
    return savgol_filter(data, window_size, 4)


def find_zero_start(df: DataFrame, key: str):
    _df = df.dropna(subset=[key])

    mask = (_df[key] == 0) & (_df[key].shift().notna()) & (_df[key].shift() != 0)
    result_indices = _df[mask].index
    return result_indices


def merge_data_with_time(density_data: DataFrame, temperature_data: DataFrame, start_time: str, end_time: str,
                         density_offset_ms=0, window_size=4501, k=4, manual=False, key='密度1') -> DataFrame:
    assert key in ['密度1', '密度2'], 'key must be 密度1 or 密度2'
    if manual:
        print('手动模式不过滤时间')
        _temperature_data = temperature_data.copy()
    else:
        # 找到温度最大值的时间
        start_time_ms = parse_file_time_to_unix_mills(start_time)
        end_time_ms = parse_file_time_to_unix_mills(end_time)
        print('正在过滤时间')
        _temperature_data = temperature_data[
            (temperature_data['unix时间'] >= start_time_ms) & (temperature_data['unix时间'] <= end_time_ms)].copy()
    _temperature_data.drop('双波长温度', axis=1, inplace=True)
    _temperature_data.dropna(how='all', subset=['单波长温度'], inplace=True)
    _temperature_data.reset_index(inplace=True, drop=True)

    density_data_offset = density_data.copy()
    density_data_offset['unix时间'] += density_offset_ms

    laser_index = find_zero_start(density_data_offset, key + '激光')

    if not manual:
        print('正在匹配时间')
        max_index_swth = _temperature_data['单波长温度'].idxmax()

        to_fit = _temperature_data.loc[max_index_swth:].copy()
        to_fit.reset_index(inplace=True, drop=True)

        y_smooth = sliding_window_smooth(to_fit['单波长温度'], window_size)

        # 计算一阶导数
        first_derivative = np.diff(y_smooth)

        # 计算导数的符号
        signs = np.sign(first_derivative)

        # 找到符号变化的位置
        change_points = np.where(np.diff(signs) != 0)[0]

        # 由于 np.diff 会减少一个元素，所以需要加 1 来调整索引
        change_points += 1

        if change_points[-1] < len(to_fit) - 1:
            change_points = np.append(change_points, len(to_fit) - 1)
        if change_points[0] > 0:
            change_points = np.insert(change_points, 0, 0)

        distance_array = None
        for i in range(1, len(change_points)):
            # 区间长度*斜率
            vertical_distance = abs(y_smooth[change_points[i]] - y_smooth[change_points[i - 1]])
            if not distance_array or vertical_distance > distance_array[0] * k:
                distance_array = [vertical_distance, i]

        swth_start = change_points[distance_array[1] - 1]
        swth_end = change_points[distance_array[1]]

        temperature_start_time = _temperature_data['unix时间'][max_index_swth + swth_start]
        temperature_end_time = _temperature_data['unix时间'][max_index_swth + swth_end]

        draw_temperature_and_density_fit_vs_time(to_fit['unix时间'], y_smooth, to_fit['单波长温度'], change_points,
                                                 y_smooth[change_points], temperature_start_time, temperature_end_time,
                                                 density_data_offset, False, key=key, laser_index=laser_index)

        print('计算得到的温度开始时间', format_unix_ms(temperature_start_time))
        print('计算得到的开始温度', _temperature_data['单波长温度'].loc[max_index_swth + swth_start])
        print('计算得到的温度结束时间', format_unix_ms(temperature_end_time))
        print('计算得到的结束温度', _temperature_data['单波长温度'].loc[max_index_swth + swth_end])

    else:
        temperature_start_time = parse_csv_time_to_unix_mills(manual[0])
        temperature_end_time = parse_csv_time_to_unix_mills(manual[1])

        draw_temperature_and_density_fit_vs_time(_temperature_data['unix时间'], None, _temperature_data['单波长温度'],
                                                 None,
                                                 None, temperature_start_time, temperature_end_time,
                                                 density_data_offset,
                                                 manual,
                                                 key=key, laser_index=laser_index)

    _temperature_data = temperature_data[
        (temperature_data['unix时间'] >= temperature_start_time) & (
                temperature_data['unix时间'] <= temperature_end_time)].copy()
    _temperature_data.reset_index(inplace=True, drop=True)
    if '密度1激光' in density_data_offset.columns:
        density_data_offset.drop('密度1激光', axis=1, inplace=True)
    if '密度2激光' in density_data_offset.columns:
        density_data_offset.drop('密度2激光', axis=1, inplace=True)
    _density_data = density_data_offset[
        (density_data_offset['unix时间'] >= temperature_start_time) & (
                density_data_offset['unix时间'] <= temperature_end_time)].copy()
    if key == '密度1':
        _density_data.drop('密度2', axis=1, inplace=True)
    elif key == '密度2':
        _density_data.drop('密度1', axis=1, inplace=True)
    _density_data.dropna(how='all', subset=[key], inplace=True)
    if 'index' in _density_data.columns:
        _density_data.drop(['index'], axis=1, inplace=True)
    if 'index' in _temperature_data.columns:
        _temperature_data.drop(['index'], axis=1, inplace=True)

    print('过滤后的温度数据量', len(_temperature_data))
    print('过滤后的密度数据量', len(_density_data))
    assert len(_temperature_data) > 0 and len(_density_data) > 0, '无法匹配温度和密度数据'
    print('过滤后的温度数据开始时间', format_unix_ms(_temperature_data.iloc[0]['unix时间']))
    print('过滤后的温度数据结束时间', format_unix_ms(_temperature_data.iloc[-1]['unix时间']))
    print('过滤后的密度数据开始时间', format_unix_ms(_density_data.iloc[0]['unix时间']))
    print('过滤后的密度数据结束时间', format_unix_ms(_density_data.iloc[-1]['unix时间']))
    _density_data.rename(columns={key: '密度'}, inplace=True)

    print(f'密度有 {_density_data["密度"].nunique()} 种值')

    merged_data = pd.merge(_density_data, _temperature_data, on='unix时间', how='outer')
    print(f'合并后的数据量：{len(merged_data)}')
    merged_data.sort_values(by=['unix时间'], inplace=True)
    merged_data['时间'] = merged_data['unix时间'].apply(format_unix_ms)

    merged_data['单波长温度'] = merged_data['单波长温度'].interpolate(method='polynomial', order=2)
    merged_data['双波长温度'] = merged_data['双波长温度'].interpolate(method='polynomial', order=2)
    merged_data.dropna(subset=['密度'], inplace=True)
    merged_data.dropna(how='any', inplace=True)
    merged_data.reset_index(inplace=True)
    merged_data['密度'] = merged_data['密度'].astype('float')
    merged_data['单波长温度'] = merged_data['单波长温度'].astype('float')
    merged_data['双波长温度'] = merged_data['双波长温度'].astype('float')
    return merged_data


def draw_density_vs_temperature(merged_data: DataFrame, path: str):
    print('正在绘制密度与温度关系图')
    fig, ax1 = plt.subplots(figsize=(20, 10), dpi=200)
    plt.rcParams['font.size'] = 16
    slope1, intercept1 = linear_fit(merged_data['单波长温度'], merged_data['密度'])
    slope2, intercept2 = linear_fit(merged_data['双波长温度'], merged_data['密度'])

    ax1.plot(merged_data['单波长温度'], [slope1 * x + intercept1 for x in merged_data['单波长温度']],
             label=f'Density - Single wavelength infrared pyrometer - fit ρ={round(intercept1, 2)}{format(slope1, ".2e")}(T-TL)')
    ax1.plot(merged_data['双波长温度'], [slope2 * x + intercept2 for x in merged_data['双波长温度']],
             label=f'Density - Dual wavelength infrared pyrometer - fit ρ={round(intercept2, 2)}{format(slope2, ".2e")}(T-TL)')

    ax1.scatter(merged_data['单波长温度'], merged_data['密度'], label='Density - Single wavelength infrared pyrometer',
                marker='o', facecolors='none', edgecolors='blue')
    ax1.scatter(merged_data['双波长温度'], merged_data['密度'], label='Density - Dual wavelength infrared pyrometer',
                marker='o', facecolors='none', edgecolors='orange')
    ax1.set_xlabel('Temperature ℃')
    ax1.set_ylabel('Density g/cm³')
    ax1.legend()
    ax1.set_title('Density vs Temperature')
    plt.grid()
    fig.tight_layout()
    plt.savefig(path)
    plt.show()


def draw_density_vs_time(density_data: DataFrame, path: str):
    print('正在绘制密度与时间关系图')
    plt.rcParams['font.size'] = 16
    fig, ax1 = plt.subplots(figsize=(50, 10), dpi=200)
    formatter = mdates.DateFormatter(plot_time_format)
    locator = mdates.AutoDateLocator()

    x_values = [datetime.fromtimestamp(x / 1000) for x in density_data['unix时间']]
    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_locator(locator)
    ax1.scatter(x_values, density_data['密度1'], label='Density - DCM1')
    ax1.scatter(x_values, density_data['密度2'], label='Density - DCM2')
    ax1.set_ylabel('Density g/cm³')
    for label in ax1.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')

    ax1.legend()
    fig.tight_layout()
    plt.grid()
    ax1.set_title('Density vs time')
    plt.savefig(path)
    plt.show()


def draw_temperature_vs_time(temperature_data: DataFrame, path: str):
    print('正在绘制温度与时间关系图')
    plt.rcParams['font.size'] = 16
    fig, ax1 = plt.subplots(figsize=(50, 10), dpi=200)
    x_values = [datetime.fromtimestamp(x / 1000) for x in temperature_data['unix时间']]
    formatter = mdates.DateFormatter(plot_time_format)
    locator = mdates.AutoDateLocator()

    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_locator(locator)

    ax1.plot(x_values, temperature_data['单波长温度'], label='Single wavelength infrared pyrometer')
    ax1.plot(x_values, temperature_data['双波长温度'], label='Dual wavelength infrared pyrometer')
    ax1.set_ylabel('Temperature ℃')
    for label in ax1.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')

    ax1.legend()
    fig.tight_layout()
    plt.grid()
    ax1.set_title('Temperature vs time')
    plt.savefig(path)
    plt.show()


def draw_temperature_and_density_fit_vs_time(time, smooth, actual, points_index, points_y, draw_start_x, draw_end_x,
                                             density, manual: bool, key='密度1', laser_index=[]):
    print('正在绘制温度与时间关系图')
    assert key in ['密度1', '密度2'], 'key must be 密度1 or 密度2'
    plt.rcParams['font.size'] = 16
    fig, ax1 = plt.subplots(figsize=(50, 10), dpi=200)
    ax2 = ax1.twinx()

    plt.xlim(datetime.fromtimestamp(draw_start_x / 1000 - 5), datetime.fromtimestamp(draw_end_x / 1000 + 5))

    x_values = [datetime.fromtimestamp(x / 1000) for x in time]
    x_values_density = [datetime.fromtimestamp(x / 1000) for x in density['unix时间']]

    formatter = mdates.DateFormatter(plot_time_format)
    locator = mdates.AutoDateLocator()

    ax2.scatter(x_values_density, density[key], label='Density', marker='o', facecolors='none',
                edgecolors='red')
    for i in laser_index:
        unix_time_ms = density.loc[i]['unix时间']
        x_value = datetime.fromtimestamp(unix_time_ms / 1000)
        if draw_start_x / 1000 - 5 <= unix_time_ms / 1000 <= draw_end_x / 1000 + 5:
            plt.axvline(x=x_value, color='r', linestyle='--')
            print(f'识别到的激光消失时间 {format_unix_ms(unix_time_ms)}')
            ax1.text(
                x=x_value,
                y=0,  # y轴位置为下方（相对于ax1的坐标系）
                s=f'{format_unix_ms(unix_time_ms)}',
                ha='right',
                va='top',
                color='black',
                rotation=45,
                fontsize=10,
                transform=ax1.get_xaxis_transform()  # 使用x轴坐标系
            )
    ax2.set_ylabel('Density g/cm³')

    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_locator(locator)

    ax1.fill_between(x_values, actual.min(), actual.max(),
                     where=[
                         datetime.fromtimestamp(draw_start_x / 1000) <= x <= datetime.fromtimestamp(draw_end_x / 1000)
                         for x in x_values], color='green', alpha=0.2, label='Selected area')
    ax1.plot(x_values, actual, label='Single wavelength infrared pyrometer')
    if not manual:
        ax1.plot(x_values, smooth, label='fit')
        scatter_x_values = [datetime.fromtimestamp(x / 1000) for x in time[points_index]]
        ax1.scatter(scatter_x_values, points_y, label='points', color='red')
    ax1.set_ylabel('Temperature ℃')
    for label in ax1.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')

    ax1.legend()
    fig.tight_layout()
    plt.grid()
    ax1.set_title('Temperature vs time')
    plt.show()


def draw_temperature_and_density_vs_time(density_data: DataFrame, temperature_data: DataFrame, path: str, key='密度1'):
    print('正在绘制温度和密度与时间关系图')
    assert key in ['密度1', '密度2'], 'key must be 密度1 or 密度2'
    plt.rcParams['font.size'] = 16
    fig, ax1 = plt.subplots(figsize=(50, 10), dpi=200)
    ax2 = ax1.twinx()

    x_values_density = [datetime.fromtimestamp(x / 1000) for x in density_data['unix时间']]
    x_values_temperature = [datetime.fromtimestamp(x / 1000) for x in temperature_data['unix时间']]
    formatter = mdates.DateFormatter(plot_time_format)
    locator = mdates.AutoDateLocator()

    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_locator(locator)

    ax1.scatter(x_values_density, density_data[key], label='Density', marker='o', facecolors='none',
                edgecolors='green')
    ax1.set_ylabel('Density g/cm³')

    ax2.plot(x_values_temperature, temperature_data['单波长温度'], label='Single wavelength infrared pyrometer')
    ax2.plot(x_values_temperature, temperature_data['双波长温度'], label='Dual wavelength infrared pyrometer')
    ax2.set_ylabel('Temperature ℃')
    for label in ax1.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')

    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout()
    plt.grid()
    ax1.set_title('Temperature and Density vs time')
    plt.savefig(path)
    plt.show()


def draw_merged_temperature_and_density_vs_time(merged_data: DataFrame, path: str):
    print('正在绘制温度和密度与时间关系图')
    plt.rcParams['font.size'] = 16
    fig, ax1 = plt.subplots(figsize=(50, 10), dpi=200)
    ax2 = ax1.twinx()

    x_values = [datetime.fromtimestamp(x / 1000) for x in merged_data['unix时间']]
    formatter = mdates.DateFormatter(plot_time_format)
    locator = mdates.AutoDateLocator()

    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_locator(locator)

    ax1.scatter(x_values, merged_data['密度'], label='Density', marker='o', facecolors='none',
                edgecolors='green')
    ax1.set_ylabel('Density g/cm³')

    ax2.plot(x_values, merged_data['单波长温度'], label='Single wavelength infrared pyrometer')
    ax2.plot(x_values, merged_data['双波长温度'], label='Dual wavelength infrared pyrometer')
    ax2.set_ylabel('Temperature ℃')
    for label in ax1.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')

    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout()
    plt.grid()
    ax1.set_title('Temperature and Density vs time')
    plt.savefig(path)
    plt.show()


def strict_match_file_date(start_time: str, end_time: str, filename: str) -> bool:
    segs = filename.split('_')
    return int(start_time) <= int(segs[4]) < int(segs[5]) <= int(end_time)


def parse_file_time_to_unix_mills(time_str: str) -> Union[float, None]:
    try:
        time_object = datetime.strptime(time_str, '%Y%m%d%H%M%S')
        time_unix_mills = time_object.timestamp() * 1000
        return time_unix_mills
    except:
        return None


def format_size(size_bytes):
    """
    将字节数转换为更易读的格式，如KB、MB、GB等。

    :param size_bytes: 文件大小，以字节为单位
    :return: 格式化后的字符串
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    if size_bytes == 0:
        return "0B"

    index = 0
    while size_bytes >= 1024 and index < len(units) - 1:
        size_bytes /= 1024.0
        index += 1

    return f"{size_bytes:.2f}{units[index]}"


def filter_file(files: list):
    if len(files) < 2:
        print('已选择的文件：')
        for i in range(len(files)):
            print(f'{files[i].name} 大小：{format_size(files[i].get_info().size)}')
        return files
    for i in range(len(files)):
        print(f'{i + 1}. {files[i].name} 大小：{format_size(files[i].get_info().size)}')
    target = []
    nums = set()
    s = input('请输入需要下载的文件序号，以英文逗号分隔，例如 1,2,3，按回车确认').strip()
    while len(nums) == 0:
        for i in s.split(','):
            try:
                index = int(i.strip()) - 1
                if index < 0 or index >= len(files):
                    print(f'序号 {index + 1} 超出范围')
                nums.add(index)
            except:
                print(f'序号 {i} 无效')
        if len(nums) == 0:
            print('请至少选择 1 个文件')
    for i in nums:
        target.append(files[i])
    print('已选择的文件：')
    for i in range(len(target)):
        print(f'{target[i].name} 大小：{format_size(target[i].get_info().size)}')
    return target


def download_data(start_time: str, end_time: str, strict: bool, save_path='/home/mw/temp'):
    asclient = Client()
    file_lists = []

    search_condition = [
        ('密度相机1', 'DCM1'),
        ('密度相机2', 'DCM2'),
        ('单波长温度', 'SWTH'),
        ('双波长温度', 'DWTH'),
    ]

    for title, dpid in search_condition:
        print(f'\n\n正在搜索{title}数据')
        start = 0
        file_list = []
        while True:
            files, next = asclient.search(scid="TGTH", apid="CMPR", dpid=dpid, level=1, startTime=start_time,
                                          endTime=end_time, mode=0, start=start)
            if not next: break
            if strict:
                files = list(filter(lambda x: strict_match_file_date(start_time, end_time, x.name), files))
            file_list.extend(files)
            start = next
        if len(file_list) == 0:
            print(f'未找到{title}数据')
        file_lists.append(filter_file(file_list))

    dcm1_file_list, dcm2_file_list, swth_file_list, dwth_file_list = file_lists

    for i in dcm1_file_list + dcm2_file_list + swth_file_list + dwth_file_list:
        print(f'正在下载 {i.name}')
        path = f'{save_path}/{i.name}'
        if not os.path.exists(path) or os.path.getsize(path) != i.get_info().size or input(
                f'{i.name} 已存在，且大小与服务器提供的大小一致，是否重新下载？输入 y 重新下载，输入其他内容跳过，按回车确认').strip().lower() == 'y':
            i.save_local(path)
        assert os.path.getsize(
            path) == i.get_info().size, f'下载的文件大小不一致，与服务器提供的大小差 {i.get_info().size - os.path.getsize(path)} 字节'
        if path.endswith('.zip'):
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    print(f'{i.name} 中包含 {len(zf.namelist()) // 2} 张图像')
            except:
                print(f'{i.name} 已损坏')

    dcm1_file_list = list(map(lambda x: f'{save_path}/{x.name}', dcm1_file_list))
    dcm2_file_list = list(map(lambda x: f'{save_path}/{x.name}', dcm2_file_list))
    swth_file_list = list(map(lambda x: f'{save_path}/{x.name}', swth_file_list))
    dwth_file_list = list(map(lambda x: f'{save_path}/{x.name}', dwth_file_list))

    return dcm1_file_list, dcm2_file_list, swth_file_list, dwth_file_list

def time_delta(time0: str, time1: str):
    return parse_csv_time_to_unix_mills(time1) - parse_csv_time_to_unix_mills(time0)


if __name__ == '__main__':
    # start_time = "20240319110400"
    # end_time = "20240319112000"
    # density_data = calc_density(['data/0319/TGTH_CMPR_DCM1_SCI_20240319110412_20240319111915_607_0_L1_V1.zip'], ['data/0319/TGTH_CMPR_DCM2_SCI_20240319110419_20240319111930_1373_0_L1_V1.zip'],
    #                             28.5E-3)
    # density_data.to_csv('data/0319/density.csv', index=False)
    # draw_density_vs_time(density_data, 'data/0204/density.pdf')
    # temperature_data = temperature(['data/0319/TGTH_CMPR_SWTH_SCI_20240319102207_20240319111435_1119_0_L1_V1.csv'],
    #                                      ['data/0319/TGTH_CMPR_DWTH_SCI_20240319102207_20240319111435_1119_0_L1_V1.csv']
    # )
    # temperature_data.to_csv('data/0319/temperature.csv', index=False)
    # draw_temperature_vs_time(temperature_data, 'data/0204/temperature.pdf')
    # density_data = pd.read_csv('data/0319/density.csv')
    # temperature_data = pd.read_csv('data/0319/temperature.csv')
    # merged_data = merge_data_with_time(density_data, temperature_data, start_time, end_time, density_offset_ms=0,
    #                                    window_size=4501, k=4, manual=False)
    # merged_data.to_csv('data/0319/merged.csv', index=False)
    # print(time_delta('2024-03-19 11:07:03.720', '2024-03-19 11:07:03.718'))
    # power_data = power(['data/0430/TGTH_CMPR_PHDP_SCI_20250430143516_20250430155740_1421_0_L1_V1.csv'])
    # power_data.to_csv('data/0430/power.csv', index=False)
    pass
