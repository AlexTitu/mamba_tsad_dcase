import os.path as path

import numpy as np
import pandas as pd

def read_dataset(dataset, subname=None):
    if dataset == "NASA":
        assert subname in ["A-4", "T-1", "C-2"]
        data_root = "./datasets/NASA"
        train_data = pd.read_csv(path.join(data_root, f"{subname}.train.csv")).values[:, 1:-1]
        test_data = pd.read_csv(path.join(data_root, f"{subname}.test.csv")).values[:, 1:-1]
        test_label = pd.read_csv(path.join(data_root, f"{subname}.test.csv")).values[:, -1]
        test_label = test_label[:, np.newaxis]
    elif dataset == "SMD":
        assert subname in ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7", "machine-1-6"]
        data_root = "./datasets/SMD"
        train_data = pd.read_csv(path.join(data_root, f"{subname}.train.csv")).values[:, 1:-1]
        test_data = pd.read_csv(path.join(data_root, f"{subname}.test.csv")).values[:, 1:-1]
        test_label = pd.read_csv(path.join(data_root, f"{subname}.test.csv")).values[:, -1]
        test_label = test_label[:, np.newaxis]
    elif dataset == "SWaT":
        data_root = "./datasets/SWaT"
        train_data = np.load(path.join(data_root, "SWaT_train_data.npy"))
        test_data = np.load(path.join(data_root, "SWaT_test_data.npy"))
        test_label = np.load(path.join(data_root, "SWaT_test_label.npy"))
    else:
        raise NotImplementedError

    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)

    return train_data, test_data, test_label
