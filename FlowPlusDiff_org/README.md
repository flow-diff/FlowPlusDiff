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

# Dataset Preparation

Place all datasets inside the `dataset/` directory.

## WADI Dataset

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

## SWaT Dataset

The original SWaT dataset can be obtained from the same request form above.

Organize the dataset as:

```text
dataset/
└── SWAT/
    ├── train.csv
    └── test.csv
```

## PSM, SMD, MSL, and SMAP

The processed benchmark datasets can be downloaded from the Google Drive link below:

**Google Drive:** `<Google Drive link for processed datasets>`

Extract the downloaded archive and place the dataset folders directly inside the `dataset/` directory.

The final directory structure should be:

```text
dataset/
├── SWAT/
│   ├── train.csv
│   └── test.csv
├── WADI/
│   ├── train.csv
│   ├── test.csv
│   └── test_label.csv
├── PSM/
├── SMD/
├── MSL/
└── SMAP/
```

---

# Pretrained Checkpoints

Pretrained checkpoints for all datasets are provided. Each archive contains the checkpoints for the **five random seeds** used in our experiments.

| Dataset | Google Drive |
|---------|--------------|
| SWAT | `<SWAT checkpoint link>` |
| WADI | `<WADI checkpoint link>` |
| PSM | `<PSM checkpoint link>` |
| SMD | `<SMD checkpoint link>` |
| MSL | `<MSL checkpoint link>` |
| SMAP | `<SMAP checkpoint link>` |

Download the desired checkpoint and place the `.pth` file inside the `main_pth/` directory.

Example:

```text
FlowPlusDiff/
├── main.py
├── main_pth/
│   └── MSL_seed_1000_i_d_256_nfl_4_ndl_4_p_0.5_ntimes_100_bmin_0.001_bmax_0.1_nh_4_fat_4_st_16_ud_[1, 2, 4, 8]_cond_True_ot_False_lmo_True.pth
├── dataset/
│   ├── SWAT/
│   ├── WADI/
│   ├── PSM/
│   ├── SMD/
│   ├── MSL/
│   └── SMAP/
└── ...
```
