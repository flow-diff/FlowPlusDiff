# Flow + Diffusion for Time Series Anomaly Detection

This repository contains the official implementation of **Flow + Diffusion for Time Series Anomaly Detection**.

---

# Requirements

Install the required dependencies before running the code:

```bash
pip install -r requirements.txt
```

---

# Running the Code

Train or evaluate the model using:

```bash
python main.py DATASET_NAME TRAINING
```

where

```text
DATASET_NAME ∈ {SWAT, WADI, PSM, SMD, MSL, SMAP}

TRAINING ∈ {true, false}
```

### Examples

Train a model:

```bash
python main.py PSM true
```

Evaluate using a pretrained checkpoint:

```bash
python main.py PSM false
```

When `TRAINING=false`, the script automatically loads the corresponding pretrained checkpoint from the `main_pth/` directory.

---

# Dataset Preparation

Place all datasets inside the `dataset/` directory.

## WADI

The original WADI dataset can be obtained by submitting the request form available at:

https://www.sutd.edu.sg/itrust/request-for-datasets/

After downloading the dataset, preprocess and rename the files as follows:

```text
dataset/
└── WADI/
    ├── train.csv
    ├── test.csv
    └── test_label.csv
```

## SWaT

The original SWaT dataset can be obtained from the same request form above.

Organize the files as

```text
dataset/
└── SWAT/
    ├── train.csv
    └── test.csv
```

## PSM, SMD, MSL, and SMAP

The processed benchmark datasets can be downloaded from:

**Google Drive**

https://drive.google.com/file/d/1URiYg8bRidmm7bYKy8IKB-heCsqBC22i/view?usp=sharing

Extract the downloaded archive and place the dataset folders directly inside the `dataset/` directory.

The final directory structure should be

```text
dataset/
├── MSL/
├── PSM/
├── SMAP/
├── SMD/
├── SWAT/
│   ├── train.csv
│   └── test.csv
└── WADI/
    ├── train.csv
    ├── test.csv
    └── test_label.csv
```

---

# Pretrained Checkpoints

We provide pretrained checkpoints for all benchmark datasets. Each archive contains the checkpoints corresponding to the **five random seeds** used in our experiments.

| Dataset | Download |
|---------|----------|
| SMD | <SMD checkpoint link> |
| MSL | https://drive.google.com/file/d/1I1GlU6tzSNBl25raAUmoTd5tAr0-sFw5/view?usp=sharing |
| SMAP | https://drive.google.com/file/d/1Gb2xm0-tXKKd5Co4MzNtcguCxtH8NlMs/view?usp=sharing |
| SWAT | https://drive.google.com/file/d/1F96YeyAoCjQ0BN2Oa_5-kEf9iVCBXD3L/view?usp=sharing |
| PSM | https://drive.google.com/file/d/1DNJJuekltOCVmUrmnO-DJOX2CCcAcYmO/view?usp=sharing |
| WADI | https://drive.google.com/file/d/1o5L6zHE7pqN5q2gW8hio2yjg8_GEgNE3/view?usp=sharing |

Download the desired archive and extract the `.pth` checkpoint files into the `main_pth/` directory.

The checkpoint filenames encode the dataset, random seed, and training hyperparameters. No renaming is required.

Example:

```text
FlowPlusDiff/
├── main.py
├── main_pth/
│   └── MSL_seed_1000_i_d_256_nfl_4_ndl_4_p_0.5_ntimes_100_bmin_0.001_bmax_0.1_nh_4_fat_4_st_16_ud_[1,2,4,8]_cond_True_ot_False_lmo_True.pth
├── dataset/
│   ├── MSL/
│   ├── PSM/
│   ├── SMAP/
│   ├── SMD/
│   ├── SWAT/
│   └── WADI/
├── requirements.txt
└── ...
```

---

# Experimental Results

The evaluation results reported in the paper are provided in the `main_pth/` directory.

Each CSV file corresponds to one random seed and contains the evaluation metrics for all benchmark datasets.

```text
main_pth/
├── 1000metrics_by_dataset.csv
├── 1001metrics_by_dataset.csv
├── 1002metrics_by_dataset.csv
├── 1003metrics_by_dataset.csv
└── 1004metrics_by_dataset.csv
```

To compute the mean, standard deviation, minimum, and maximum of each evaluation metric across all five random seeds, run

```bash
cd main_pth
python compute_seed_stats.py
```

The script automatically reads the five metric files above and generates summary files of the form

```text
metrics_seed_stats_<DATASET>.csv
```

for each benchmark dataset.

---

# Citation

If you find this repository useful in your research, please consider citing our paper.

```bibtex
@article{YOUR_CITATION,
  title   = {Flow + Diffusion for Time Series Anomaly Detection},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

---

# License

This project is released under the license provided in the repository.
