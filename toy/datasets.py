import random
import pandas as pd 
import numpy as np 
import logging
logging.basicConfig(level=logging.INFO)

# torch inputs 
import torch
from torch.utils.data import DataLoader

# sklearn inputs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# TODO - create a path for when you want to download these datasets
IRIS_DATA_PATH = "iris.csv"


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# dataset template 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_split):
        self.X = torch.from_numpy(ds_split['X']).float()
        self.y = torch.from_numpy(ds_split['y']).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]


# loading iris dataset. 
def load_iris(shuffle=True, seed=None):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8620866&tag=1
    raw_df = pd.read_csv(IRIS_DATA_PATH)

    mappings = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    raw_df["Class"] = raw_df["Class"].apply(lambda x: mappings[x])

    # split and shuffle; shuffle=true will shuffle the elements before the split.
    np.random.seed(seed)
    random.seed(seed)

    train_df, test_df = train_test_split(raw_df, test_size=0.20, shuffle=shuffle)
    train_df, val_df = train_test_split(train_df, test_size=0.25, shuffle=shuffle)  # 0.25 * 0.8 = 0.2

    train_labels = np.array(train_df.pop("Class"))
    val_labels = np.array(val_df.pop("Class"))
    test_labels = np.array(test_df.pop("Class"))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # scaling data.
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    # scaling validation and test based on training data.
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('iris train label shape: {}'.format(train_labels.shape))
    logging.info("iris train feature.shape: {}".format(train_features.shape))
    logging.info('iris val label shape: {}'.format(val_labels.shape))
    logging.info("iris val feature.shape: {}".format(val_features.shape))
    logging.info('iris test label shape: {}'.format(test_labels.shape))
    logging.info("iris test feature.shape: {}".format(test_features.shape))

    return {
        'train': {'X': train_features, 'y': train_labels},
        'val': {'X': val_features, 'y': val_labels},
        'test': {'X': test_features, 'y': test_labels},
    }

