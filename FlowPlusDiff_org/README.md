# Flow + Diffusion for Time Series Anomaly Detection

## Running the Code

To train or evaluate the model, run:

```bash
python main.py DATASET_NAME TRAINING
```

where:

```text
DATASET_NAME ∈ {SWAT, WADI, PSM, SMD, MSL, SMAP}

TRAINING ∈ {true, false}
```

Examples:

```bash
# Train the model
python main.py PSM true

# Evaluate a pretrained model
python main.py PSM false
```

---

## Dataset Preparation

Place all datasets inside the `dataset/` directory.

### WADI Dataset

The original WADI dataset can be obtained by filling out the official request form:

https://docs.google.com/forms/d/1GOLYXa7TX0KlayqugUOOPMvbcwSQiGNMOjHuNqKcieA/viewform?edit_requested=true

After downloading the dataset, preprocess and rename the files as follows:

```text
dataset/
└── WADI/
    ├── train.csv
    ├── test.csv
    └── test_label.csv
```

### SWaT Dataset

The original SWaT dataset can be obtained from the same request form above.

Organize the dataset as:

```text
dataset/
└── SWAT/
    ├── train.csv
    └── test.csv
```

### Other Datasets

The remaining benchmark datasets (**PSM**, **SMD**, **MSL**, and **SMAP**) can be downloaded from the Google Drive link below. Extract the archive and place the dataset folders directly inside the `dataset/` directory.

```text
dataset/
├── SWAT/
├── WADI/
├── PSM/
├── SMD/
├── MSL/
└── SMAP/
```

**Google Drive:** *<insert Google Drive link here>*
