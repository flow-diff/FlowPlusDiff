import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataset.preprocessing.preprocess import UCR_AnomalySequence
from evaluation.basic_metrics import compute_nonstationarity
import numpy as np
import matplotlib.pyplot as plt

# ====================================================
# Utility: Auto-detect CSV header
# ====================================================
def detect_csv_header(filepath, nrows=5):
    """
    Automatically detect if CSV has a header row.
    Returns header=0 if first row looks like headers, header=None otherwise.
    """
    try:
        # Read first few rows without header
        df_peek = pd.read_csv(filepath, nrows=nrows, header=None, dtype=str)
        first_row = df_peek.iloc[0]
        
        # Check if first row contains numeric values
        numeric_count = 0
        for val in first_row:
            try:
                float(val)
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        # If most values are numeric, first row is data (no header)
        # If most values are non-numeric, first row is header
        is_header = numeric_count < len(first_row) * 0.5
        return 0 if is_header else None
    except:
        # Default to header=0 if detection fails
        return 0

# ====================================================
# Utility: Non-overlapping segment sampler
# ====================================================
import numpy as np

# ====================================================
# Budget-aware segment sampler (OVERLAP ALLOWED)
# ====================================================
def sample_segments_budget(T, seg_len, budget, used, rng, max_attempts=5000):
    segments = []
    added = 0
    attempts = 0

    while added < budget and attempts < max_attempts:
        attempts += 1
        start = rng.integers(0, T - seg_len)
        end = start + seg_len

        idx = np.arange(start, end)
        new_idx = idx[~used[idx]]

        if len(new_idx) == 0:
            continue

        # Cap to remaining budget
        remain = budget - added
        if len(new_idx) > remain:
            new_idx = rng.choice(new_idx, size=remain )

        used[new_idx] = True
        added += len(new_idx)
        segments.append((start, end))

    return segments, used



# ====================================================
# Random Spike Anomalies (OVERLAP ALLOWED, BUDGETED)
# ====================================================
def inject_spike_random(X, N_spike, used, rng,std_clean):
    T, F = X.shape
    X_out = X.copy()

    n_feat = 1
    added = 0
    attempts = 0
    max_attempts = 5000

    while added < N_spike and attempts < max_attempts:
        attempts += 1
        t = rng.integers(0, T)

        if not used[t]:
            used[t] = True
            added += 1

        feats = rng.choice(F, size=n_feat )
        base_mag = rng.uniform(2.0, 6.0)

        for f in feats:
            mag_f = base_mag * rng.uniform(0.8, 1.2)
            X_out[t, f] += mag_f * std_clean[0,f]
    return X_out, used


# ====================================================
# Random Collective Anomalies (OVERLAP ALLOWED)
# ====================================================
def inject_collective_random(X, N_coll, used, rng,std_clean):
    T, F = X.shape
    X_out = X.copy()

    n_feat = 1
    seg_len = rng.integers(10, 50)

    segments, used = sample_segments_budget(T, seg_len, N_coll, used, rng)

    for s, e in segments:
        feats = rng.choice(F, size=n_feat )
        mag = rng.uniform(1.0, 5.0)

        for f in feats:
            X_out[s:e, f] += mag * std_clean[:,f]

    return X_out, used


# ====================================================
# Random Seasonal Anomalies (OVERLAP ALLOWED)
# ====================================================
def inject_seasonal_random(X, N_seas, used, rng,std_clean):
    """
    X         : raw clean signal (T, F)
    N_seas    : number of seasonal segments (budgeted)
    used      : boolean array (T,) for contamination budget
    rng       : np.random.Generator
    std_clean : feature-wise std from CLEAN training data, shape (F,)
    """
    T, F = X.shape
    X_out = X.copy()

    n_feat = 1
    block_len = rng.integers(10, 50)

    segments, used = sample_segments_budget(T, block_len, N_seas, used, rng)

    for s, e in segments:
        feats = rng.choice(F, size=n_feat, replace=False)

        # seasonal parameters
        amp_sigma = rng.uniform(0.5, 2.5)     # amplitude in σ-units
        period = rng.integers(6, max(7, e - s))
        phase = rng.uniform(0, 2 * np.pi)

        t = np.arange(e - s)
        seasonal_unit = np.sin(2 * np.pi * t / period + phase)

        for f in feats:
            seasonal = amp_sigma * std_clean[:,f] * seasonal_unit
            X_out[s:e, f] += seasonal

    return X_out, used

def inject_trend_random(X, N_trend, used, rng,std_clean):
    """
    X         : raw clean signal (T, F)
    N_trend   : number of trend segments (budgeted)
    used      : boolean array (T,) for contamination budget
    rng       : np.random.Generator
    std_clean : feature-wise std computed from CLEAN training data, shape (F,)
    """
    T, F = X.shape
    X_out = X.copy()

    n_feat = 1
    seg_len = rng.integers(10, 50)

    segments, used = sample_segments_budget(T, seg_len, N_trend, used, rng)

    for s, e in segments:
        feats = rng.choice(F, size=n_feat, replace=False)

        # total drift magnitude in σ-units
        final_sigma = rng.uniform(1.0, 4.0)   # ← key control knob
        drift_unit = np.linspace(0.0, 1.0, e - s)

        for f in feats:
            drift = final_sigma * std_clean[:,f] * drift_unit
            X_out[s:e, f] += drift

    return X_out, used



# ====================================================
# MASTER FUNCTION (UNCHANGED)
# ====================================================
def inject_anomalies_random(
    X,
    ratio=0.1,
    seed=42,
    max_retry=1,
    tol_ratio=0.01,
):
    T, F = X.shape
    print("Induced Anomaly Ratio:", ratio)

    N_target = int(T * ratio)
    N_each = N_target // 4
    tol = max(1, int(tol_ratio * T))

    best_err = np.inf
    best_X = None
    best_contaminated = None
    std_clean = np.ones_like(np.std(X, axis=0, keepdims=True))

    # ------------------------
    # main attempts
    # ------------------------
    for k in range(max_retry):
        rng = np.random.default_rng(seed + k)
        contaminated = np.zeros(T, dtype=bool)
        X_out = X.copy()

        X_out, contaminated = inject_trend_random(
            X_out, N_each, contaminated, rng,std_clean
        )
        X_out, contaminated = inject_collective_random(
            X_out, N_each, contaminated, rng,std_clean
        )
        X_out, contaminated = inject_seasonal_random(
            X_out, N_each, contaminated, rng,std_clean
        )
        X_out, contaminated = inject_spike_random(
            X_out, N_each, contaminated, rng,std_clean
        )

        err = abs(contaminated.sum() - N_target)

        if err < best_err:
            best_err = err
            best_X = X_out
            best_contaminated = contaminated.copy()

        if err <= tol:
            break

    # ------------------------
    # deficit filling using SAME injectors
    # ------------------------
    deficit = N_target - best_contaminated.sum()

    if deficit > 0:
        print(f"Filling {deficit} missing contaminated time steps")

        rng_fill = np.random.default_rng(seed + 999)
        X_fill = best_X.copy()
        contaminated = best_contaminated.copy()

        while contaminated.sum() < N_target:
            remain = N_target - contaminated.sum()
            n_fill = max(1, remain // 4)

            X_fill, contaminated = inject_trend_random(
                X_fill, n_fill, contaminated, rng_fill,std_clean
            )
            X_fill, contaminated = inject_collective_random(
                X_fill, n_fill, contaminated, rng_fill,std_clean
            )
            X_fill, contaminated = inject_seasonal_random(
                X_fill, n_fill, contaminated, rng_fill,std_clean
            )
            X_fill, contaminated = inject_spike_random(
                X_fill, n_fill, contaminated, rng_fill,std_clean
            )

        best_X = X_fill
        best_contaminated = contaminated

    # ------------------------
    # hard guarantee
    # ------------------------
    print(best_contaminated.sum())
    print(N_target)

    return best_X



def scalerfunc(train, val, test):
    # Convert to float32
    train = np.float32(train)
    val = np.float32(val)
    test = np.float32(test)


    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train)

    train = scaler.transform(train)
    val = np.clip(scaler.transform(val), -4, 4)
    result = compute_nonstationarity(
    np.concatenate([train, val], axis=0),
        window_size=128,
        stride=128
    )

    print("Mean W2:", result["mean_W2"])
    print("Std W2:", result["std_W2"])

    test = np.clip(scaler.transform(test), -4, 4)

    return train, val, test


class SplitDataset(Dataset):
    def __init__(self, base_dataset, split):
        self.base = base_dataset
        self.split = split
        self.win_size = base_dataset.win_size
        self.step = base_dataset.step
        if split == "train":
            self.data = base_dataset.train
            self.labels = base_dataset.test_labels
        elif split == "val":
            self.data = base_dataset.val
            self.labels = base_dataset.test_labels
        elif split == "test":
            self.data = base_dataset.test
            self.labels = base_dataset.test_labels
            self.step = base_dataset.win_size
        else:
            raise ValueError("split must be train/val")

    def __len__(self):
        return (self.data.shape[0] - self.win_size) // self.step + 1
        

    def __getitem__(self, index):
        index *= self.step
        if self.split!="test":
            
            return (
                np.float32(self.data[index:index + self.win_size]),
                np.float32(self.labels[0:self.win_size])
            )
        else:
            return (
                np.float32(self.data[index:index + self.win_size]),
                np.float32(self.labels[index:index + self.win_size])
            )




class WADISegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train",nratio=0.1,removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.nratio = nratio
        self.removed_binary = removed_binary
        header_mode = detect_csv_header(data_path + '/train.csv')
        data = pd.read_csv(data_path + '/train.csv', header=header_mode).ffill().bfill()
        test_data = pd.read_csv(data_path + '/test.csv', header=header_mode).ffill().bfill()
        data = data.values
        test_data = test_data.values
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values.reshape(-1, 1)
        
        if removed_binary:
            binary_idx = [
                i for i in range(data.shape[1])
                if len(np.unique(data[:, i])) <= 10
            ]
            if len(binary_idx) > 0:
                print(f"Setting {len(binary_idx)} binary feature columns to 0 (indices: {binary_idx})")
                for idx in binary_idx:
                    data[:, idx] = 0
                    test_data[:, idx] = 0
        
        if nratio !=0:
            data = inject_anomalies_random(data, ratio=nratio)
        
        self.val=data[int(0.8*data.shape[0]):data.shape[0]]
        self.train=data[0:int(0.8*data.shape[0])]
        self.train,self.val,self.test=scalerfunc(self.train,self.val,test_data)
        self.dim=self.test.shape[1]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        
    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
     
        
class SWaTSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train",nratio=0,removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.nratio = nratio
        self.removed_binary = removed_binary

        data = pd.read_csv(data_path + '/train.csv', header=1).ffill().bfill()
        data = data.values[:, 1:-1]
        
        x,y=data.shape
        test_data = pd.read_csv(data_path + '/test.csv').ffill().bfill()
        y = test_data.iloc[:,-1].to_numpy()
        test_data=test_data.values[:, 1:-1]
        

        labels = []
        for i in y:
            if i == 'Attack':
                labels.append(1)
            elif i == 'A ttack':
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)
        
        if removed_binary:
            binary_idx = [
                i for i in range(data.shape[1])
                if len(np.unique(data[:, i])) <= 10
            ]
            if len(binary_idx) > 0:
                print(f"Setting {len(binary_idx)} binary feature columns to 0 (indices: {binary_idx})")
                for idx in binary_idx:
                    data[:, idx] = 0
                    test_data[:, idx] = 0

        if nratio !=0:
            data = inject_anomalies_random(data, ratio=nratio)
        self.train = data
        self.test = test_data
        self.train[:, [5,10]] = 0
        self.test[:, [5,10]] = 0
        self.dim=self.test.shape[1]
        self.test_labels = labels.reshape(-1, 1)
        self.val=self.train[int(0.8*self.train.shape[0]):self.train.shape[0]]
        self.train=self.train[0:int(0.8*self.train.shape[0])]
        self.train,self.val,self.test=scalerfunc(self.train,self.val,self.test)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        
    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
     

class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train",nratio=0,removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.nratio = nratio
        self.removed_binary = removed_binary
        header_mode = detect_csv_header(data_path + '/train.csv')
        data = pd.read_csv(data_path + '/train.csv', header=header_mode).ffill().bfill()
        test_data = pd.read_csv(data_path + '/test.csv', header=header_mode).ffill().bfill()
        data = data.values[:, 1:]
        test_data = test_data.values[:, 1:]
        self.test_labels = (
            pd.read_csv(data_path + '/test_label.csv')
            .fillna(0).values[:, 1:]
            .astype(int)
        )
        
        if removed_binary:
            binary_idx = [
                i for i in range(data.shape[1])
                if len(np.unique(data[:, i])) <= 10
            ]
            if len(binary_idx) > 0:
                print(f"Setting {len(binary_idx)} binary feature columns to 0 (indices: {binary_idx})")
                for idx in binary_idx:
                    data[:, idx] = 0
                    test_data[:, idx] = 0
        
        if nratio !=0:
            data = inject_anomalies_random(data, ratio=nratio)
        datalen=len(data)
        trainlen=int(datalen*0.8)
        self.train,self.val,self.test=scalerfunc(data[0:trainlen],data[trainlen:datalen],test_data)
        self.dim=self.test.shape[1]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        
    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])


class SWANSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train",nratio=0,removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.nratio = nratio
        self.removed_binary = removed_binary
        header_mode = detect_csv_header(data_path + '/train.csv')
        data = pd.read_csv(data_path + '/train.csv', header=header_mode).ffill().bfill()
        test_data = pd.read_csv(data_path + '/test.csv', header=header_mode).ffill().bfill()
        data = data.values
        test_data = test_data.values
        self.test_labels = (
            pd.read_csv(data_path + '/test_label.csv')
            .fillna(0).values[:, 1:]
            .astype(int)
        )
        print("Initial data shape:", data[:10])
        print("Initial test data shape:", test_data[:10])
        print("Initial test labels shape:", self.test_labels[:10])
        if removed_binary:
            binary_idx = [
                i for i in range(data.shape[1])
                if len(np.unique(data[:, i])) <= 2
            ]
            if len(binary_idx) > 0:
                print(f"Setting {len(binary_idx)} binary feature columns to 0 (indices: {binary_idx})")
                for idx in binary_idx:
                    data[:, idx] = 0
                    test_data[:, idx] = 0
        
        if nratio !=0:
            data = inject_anomalies_random(data, ratio=nratio)
        datalen=len(data)
        trainlen=int(datalen*0.8)
        self.train,self.val,self.test=scalerfunc(data[0:trainlen],data[trainlen:datalen],test_data)
        self.dim=self.test.shape[1]
        print("test:", self.test.shape)
        print("train:", self.train.shape)
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        
    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train",nratio=0,removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.nratio = nratio
        self.removed_binary = removed_binary
        data = pd.DataFrame(np.load(data_path + "/MSL_train.npy")).ffill().bfill().values
        test_data = pd.DataFrame(np.load(data_path + "/MSL_test.npy")).ffill().bfill().values
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")

        # ensure arrays are writable (avoid read-only views/mmap)
        data = np.array(data, copy=True)
        test_data = np.array(test_data, copy=True)

        if removed_binary:
            binary_idx = [
                i for i in range(data.shape[1])
                if len(np.unique(data[:, i])) <= 10
            ]
            if len(binary_idx) > 0:
                print(f"Setting {len(binary_idx)} binary feature columns to 0 (indices: {binary_idx})")
                for idx in binary_idx:
                    data[:, idx] = 0
                    test_data[:, idx] = 0

        if nratio !=0:
            data = inject_anomalies_random(data, ratio=nratio)
        datalen=len(data)
        trainlen=int(datalen*0.8)
        self.train,self.val,self.test=scalerfunc(data[0:trainlen],data[trainlen:datalen],test_data)
        self.dim=self.test.shape[1]

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        
    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        
     

class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train",nratio=0,removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.nratio = nratio
        self.removed_binary = removed_binary
        lab_tst = []
        total_anomaly_points = 0
        pth=data_path+"/"
        labeled_anomalies = pd.read_csv(data_path + '/labeled_anomalies.csv')
        data_dims = {'SMAP': 25, 'MSL': 55}
        insert=False
        for smap_or_msl in ['SMAP']:
            for i in range(len(labeled_anomalies)):
                #print(f'  -> {labeled_anomalies["chan_id"][i]} ({i+1} / {len(labeled_anomalies)})')
                if labeled_anomalies['spacecraft'][i] == smap_or_msl:
                    # load corresponding .npy file in test and train
                    item = np.load(pth + 'train/' + labeled_anomalies['chan_id'][i] + '.npy')
                    assert item.shape[-1] == data_dims[smap_or_msl]
                    item2 = np.load(pth + 'test/' + labeled_anomalies['chan_id'][i] + '.npy')
                    assert item2.shape[-1] == data_dims[smap_or_msl]
                    labs = labeled_anomalies['anomaly_sequences'][i]
                    labs_s = labs.replace('[', '').replace(']', '').replace(' ', '').split(',')
                    labs_i = [[int(labs_s[i]), int(labs_s[i+1])] for i in range(0, len(labs_s), 2)]
                    assert labeled_anomalies['num_values'][i] == len(item2)
                    item3 = np.zeros(len(item2))
                    for sec in labs_i:
                        item3[sec[0]:sec[1]] = 1
                        total_anomaly_points += sec[1] - sec[0]

                    if insert:
                        data=np.concatenate((data,item),axis=0)
                        test_data=np.concatenate((test_data,item2),axis=0)
                        lab_tst=np.concatenate((lab_tst,item3),axis=0)
                    else:
                        insert=True
                        data=item
                        test_data=item2
                        lab_tst=item3
                # Set binary features to 0 instead of removing them
 
        data = pd.DataFrame(data).ffill().bfill().values
        test_data = pd.DataFrame(test_data).ffill().bfill().values
        data = np.array(data, copy=True)
        test_data = np.array(test_data, copy=True)
        if self.removed_binary:
            binary_idx = [
                i for i in range(data.shape[1])
                if len(np.unique(data[:, i])) <= 10
            ]
            if len(binary_idx) > 0:
                print(f"Setting {len(binary_idx)} binary feature columns to 0 (indices: {binary_idx})")
                for idx in binary_idx:
                    data[:, idx] = 0
                    test_data[:, idx] = 0

        
        if nratio !=0:
            data = inject_anomalies_random(data, ratio=nratio)
        self.test_labels = lab_tst
        datalen=len(data)
        trainlen=int(datalen*0.8)
        
        self.train,self.val,self.test=scalerfunc(data[0:trainlen],data[trainlen:datalen],test_data)
        self.dim=test_data.shape[1]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
     

# --------------------------------------------------
# 1. Extract anomaly segment lengths
# --------------------------------------------------
def anomaly_segment_lengths(labels):
    """
    labels: 1D array-like of binary anomaly labels (0 or 1)
    returns: 1D numpy array of anomaly segment lengths
    """
    labels = np.asarray(labels).astype(int)

    # pad to detect boundary segments
    padded = np.pad(labels, (1, 1), mode="constant")
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    lengths = ends - starts
    return lengths


# --------------------------------------------------
# 2. Compute empirical distribution
# --------------------------------------------------
def segment_length_distribution(lengths):
    """
    lengths: 1D array of segment lengths
    returns: unique lengths, PMF, CDF
    """
    unique_L, counts = np.unique(lengths, return_counts=True)
    pmf = counts / counts.sum()
    cdf = np.cumsum(pmf)
    return unique_L, pmf, cdf


# --------------------------------------------------
# 3. Summary statistics
# --------------------------------------------------
def segment_statistics(lengths):
    return {
        "num_segments": len(lengths),
        "mean": lengths.mean(),
        "median": np.median(lengths),
        "std": lengths.std(),
        "min": lengths.min(),
        "max": lengths.max(),
        "q25": np.percentile(lengths, 25),
        "q75": np.percentile(lengths, 75),
    }


# --------------------------------------------------
# 4. Visualization
# --------------------------------------------------
def plot_segment_distribution(lengths, unique_L, pmf, cdf):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    axes[0].hist(lengths, bins="auto")
    axes[0].set_xlabel("Anomaly segment length")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram of anomaly segment lengths")

    # PMF
    axes[1].bar(unique_L, pmf)
    axes[1].set_xlabel("Anomaly segment length")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("PMF of anomaly segment lengths")

    # CDF
    axes[2].step(unique_L, cdf, where="post")
    axes[2].set_xlabel("Anomaly segment length")
    axes[2].set_ylabel("Cumulative probability")
    axes[2].set_title("CDF of anomaly segment lengths")

    plt.tight_layout()
    plt.show()


class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train",nratio=0,removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.nratio = nratio
        self.removed_binary = removed_binary
        insert=False
        test_data, lab_tst = [], []
        for ent_name in os.listdir(data_path + '/train'):
            item=pd.read_csv(data_path + '/train/' + ent_name, header=None)
            item2=pd.read_csv(data_path + '/test/' + ent_name, header=None)
            item3=np.squeeze(pd.read_csv(data_path + '/test_label/' + ent_name, header=None).to_numpy())
            
            if insert:
                data=np.concatenate((data,item),axis=0)
                test_data=np.concatenate((test_data,item2),axis=0)
                lab_tst=np.concatenate((lab_tst,item3),axis=0)
            else:
                insert=True
                data=item
                test_data=item2
                lab_tst=item3
        
        if removed_binary:
            binary_idx = [
                i for i in range(data.shape[1])
                if len(np.unique(data[:, i])) <= 2
            ]
            if len(binary_idx) > 0:
                print(f"Setting {len(binary_idx)} binary feature columns to 0 (indices: {binary_idx})")
                for idx in binary_idx:
                    data[:, idx] = 0
                    test_data[:, idx] = 0
        
        self.test_labels = lab_tst

        datalen=len(data)
        trainlen=int(datalen*0.8)
        self.train,self.val,self.test=scalerfunc(data[0:trainlen],data[trainlen:datalen],test_data)
        self.dim=self.test.shape[1]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
     
    
class UCRSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train", nratio=0.0, removed_binary=False):
        dataset_ind=[i for i in range(1, 2)]
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.nratio = nratio
        self.removed_binary = removed_binary
        insert=False
        test_data, lab_tst = [], []
        for idx in dataset_ind:
            dataset_idx = int(idx)
            dataset_importer = UCR_AnomalySequence.create_by_id(dataset_idx)
            trn = dataset_importer.train_data[:, None]  # add channel dim; (ts_len, 1)
            tst = dataset_importer.test_data[:, None]  # (ts_len, 1)
            print(trn.shape)
            print(tst.shape)
                # anomaly data
            self.anom_start = dataset_importer.anom_start - dataset_importer.train_stop  # relative to `test_data`
            self.anom_stop = dataset_importer.anom_stop - dataset_importer.train_stop  # relative to `test_data`
            label = np.zeros_like(tst)[:, 0]  # (ts_len,)
            label[self.anom_start:self.anom_stop] = 1.
        
            if insert:
                data=np.concatenate((data,trn),axis=0)
                test_data=np.concatenate((test_data,tst),axis=0)
                lab_tst=np.concatenate((lab_tst,label),axis=0)
            else:
                insert=True
                data=trn
                test_data=tst
                lab_tst=label
        self.test_labels = lab_tst
        self.train,self.test =scalerfunc(data,test_data,self.test_labels)
        self.dim=self.test.shape[1]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

     
def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD',nratio=0.0,removed_binary=False):
    '''
    model : 'train' or 'test'
    '''
    if mode == 'train' or mode == 'val':
        step=step
    else:
        step=win_size
    datasetname=dataset
    if (dataset == 'SMD'):
        dataset= SMDSegLoader(data_path, win_size, step, mode,nratio,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'MSL'):
        dataset= MSLSegLoader(data_path, win_size, step, mode,nratio,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, step, mode,nratio,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, step, mode,nratio,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'SWAT'):
        dataset = SWaTSegLoader(data_path, win_size, step, mode,nratio,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'UCR'):
        dataset = UCRSegLoader(data_path, win_size, step, mode,nratio,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'WADI'):
        dataset = WADISegLoader(data_path, win_size, step, mode,nratio,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'SWAN'):
        dataset = SWANSegLoader(data_path, win_size, step, mode,nratio,removed_binary)
        dimret = dataset.dim 

        
    train_loader = DataLoader(
        SplitDataset(dataset, "train"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        SplitDataset(dataset, "val"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  
    )

    test_loader = DataLoader(
        SplitDataset(dataset, "test"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print("Loaded:"+str(datasetname))
    return train_loader,val_loader,test_loader, dimret