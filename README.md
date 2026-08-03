# Flow + Diffusion for Time Series Anomaly Detection

## Requirements

Install the required dependencies before running the code:

```bash
pip install -r requirements.txt
```

---

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

When `TRAINING=false`, the script automatically loads the corresponding pretrained checkpoint from the `main_pth/` directory.

---

## Dataset Preparation

Place all datasets inside the `dataset/` directory.

### WADI Dataset

The original WADI dataset can be obtained by filling out the official request form:

https://www.sutd.edu.sg/itrust/request-for-datasets/

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

### PSM, SMD, MSL, and SMAP

The processed benchmark datasets can be downloaded from the Google Drive link below:

**Google Drive:** [`<Google Drive link for processed datasets>`](https://drive.google.com/file/d/1URiYg8bRidmm7bYKy8IKB-heCsqBC22i/view?usp=sharing)

Extract the downloaded archive and place the dataset folders directly inside the `dataset/` directory.

The final directory structure should be:

```text
dataset/
├── MSL/
├── PSM/
├── SMAP/
├── SMD/
├── SWAT/
│   ├── train.csv
│   └── test.csv
├── WADI/
│   ├── train.csv
│   ├── test.csv
│   └── test_label.csv
```

---

## Pretrained Checkpoints

Pretrained checkpoints for all datasets are provided. Each archive contains the checkpoints for the **five random seeds** used in our experiments.

| Dataset | Google Drive |
|---------|--------------|
| SMD | [Download](<SMD checkpoint link>) |
| MSL | [Download](https://drive.google.com/file/d/1I1GlU6tzSNBl25raAUmoTd5tAr0-sFw5/view?usp=sharing) |
| SMAP | [Download](https://drive.google.com/file/d/1Gb2xm0-tXKKd5Co4MzNtcguCxtH8NlMs/view?usp=sharing) |
| SWAT | [Download](https://drive.google.com/file/d/1F96YeyAoCjQ0BN2Oa_5-kEf9iVCBXD3L/view?usp=sharing) |
| PSM | [Download](https://drive.google.com/file/d/1DNJJuekltOCVmUrmnO-DJOX2CCcAcYmO/view?usp=sharing) |
| WADI | [Download](<WADI checkpoint link>) |

Download the desired checkpoint and place the `.pth` file inside the `main_pth/` directory.

The checkpoint filenames encode the dataset name, random seed, and training hyperparameters. No renaming is required.

Example:

```text
FlowPlusDiff/
├── main.py
├── main_pth/
│   └── MSL_seed_1000_i_d_256_nfl_4_ndl_4_p_0.5_ntimes_100_bmin_0.001_bmax_0.1_nh_4_fat_4_st_16_ud_[1, 2, 4, 8]_cond_True_ot_False_lmo_True.pth
├── dataset/
│   ├── MSL/
│   ├── PSM/
│   ├── SMAP/
│   ├── SMD/
│   ├── SWAT/
│   └── WADI/
└── ...
```

---

## Experimental Results

The evaluation results for all experiments are provided in the `main_pth/` directory. Each CSV file corresponds to one random seed and reports the evaluation metrics for all benchmark datasets.

```text
main_pth/
├── 1000metrics_by_dataset.csv
├── 1001metrics_by_dataset.csv
├── 1002metrics_by_dataset.csv
├── 1003metrics_by_dataset.csv
└── 1004metrics_by_dataset.csv
```

To compute the mean, standard deviation, minimum, and maximum of each evaluation metric across all seeds, run:

```bash
cd main_pth
python compute_seed_stats.py
```

The script reads the `1000`–`1004` metric files and generates `metrics_seed_stats_<DATASET>.csv` for each benchmark dataset.
