import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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


def scalerfunc(train,val,test):
    train=np.float32(train)
    test=np.float32(test)
    val=np.float32(val)
    scaler = MinMaxScaler(feature_range=(-1,1)).fit(train)
    train = scaler.transform(train)
    test = np.clip(scaler.transform(test), -4,4)
    val = np.clip(scaler.transform(val), -4, 4)
    
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
    def __init__(self, data_path, win_size, step, mode="train",removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
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
    def __init__(self, data_path, win_size, step, mode="train",removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
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

        self.train = data
        self.test = test_data
        self.train[:, [5,10]] = 0 #following https://arxiv.org/pdf/2310.15416 NSPR paper
        self.test[:, [5,10]] = 0 #following https://arxiv.org/pdf/2310.15416 NSPR paper
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
    def __init__(self, data_path, win_size, step, mode="train",removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
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
    def __init__(self, data_path, win_size, step, mode="train",removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.removed_binary = removed_binary
        data = pd.DataFrame(np.load(data_path + "/MSL_train.npy")).ffill().bfill().values
        test_data = pd.DataFrame(np.load(data_path + "/MSL_test.npy")).ffill().bfill().values
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")

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
    def __init__(self, data_path, win_size, step, mode="train",removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
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


class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train",removed_binary=False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
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
     

     
def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD',removed_binary=False):
    '''
    model : 'train' or 'test'
    '''
    if mode == 'train' or mode == 'val':
        step=step
    else:
        step=win_size
    datasetname=dataset
    if (dataset == 'SMD'):
        dataset= SMDSegLoader(data_path, win_size, step, mode,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'MSL'):
        dataset= MSLSegLoader(data_path, win_size, step, mode,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, step, mode,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, step, mode,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'SWAT'):
        dataset = SWaTSegLoader(data_path, win_size, step, mode,removed_binary)
        dimret = dataset.dim 
    elif (dataset == 'WADI'):
        dataset = WADISegLoader(data_path, win_size, step, mode,removed_binary)
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