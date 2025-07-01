# CMPR-Levitation

**Containerless Materials Processing Rack (CMPR) Levitation Dataset**  
A dual-environment dataset from electrostatic levitation experiments onboard the China Space Station and on Earth.

---

## 📘 Overview

`CMPR-Levitation` provides a structured materials dataset collected using the Containerless Materials Processing Rack (CMPR). It includes two matched datasets:

- **In_Orbit**: Thermophysical measurements under microgravity aboard the China Space Station.
- **Ground_Matched**: Terrestrial validation experiments with an equivalent levitation system.

The dataset supports comparative studies of gravitational effects, modeling of high-temperature molten materials, and data-driven property prediction.



## 📂 Repository Structure

```text
CMPR-Levitation/
├── data/
│   └── CMPR_Levitation_Dataset.xlsx   # Two sheets: In_Orbit, Ground_Matched
├── scripts/
│   ├── client.py                      # Main analysis script
│   └── funcs.py                       # Utility functions
├── requirements.txt                   # Python dependencies
├── README.md                          # Project overview (this file)
└── LICENSE                            # Project license
```

## 📊 Dataset Description

The dataset is stored in `data/CMPR_Levitation_Dataset.xlsx` and contains:

### 🛰️ Sheet 1: `In_Orbit`
Organized into six thematic sections:
1. Sample Profile  
2. Physical Properties  
3. Chemical Properties  
4. Electromagnetic Properties  
5. Thermophysical Properties  
6. Tissue Properties (partially empty—updated after sample return)

### 🌍 Sheet 2: `Ground_Matched`
Organized into four sections:
1. Sample Information  
2. Physicochemical Properties  
3. Thermophysical Properties  
4. Tissue Properties  

All records use SI units and consistent field names. Full field definitions will be provided in accompanying documentation.


## 🧪 Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt

