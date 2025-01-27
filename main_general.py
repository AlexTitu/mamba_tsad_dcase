import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
import numpy as np
from utils import DCASE2024MachineDataset

torch.manual_seed(19)
random.seed(19)
np.random.seed(19)


machineFolders = os.listdir('./DCASE2024')

for machinesName in machineFolders:
    datasetType, machineType = machinesName.split('_')
    dev_train_dataset = DCASE2024MachineDataset('./DCASE2024', ('dev', 'train'), machinesName,
                                         extension='.wav', standardize='min-max')
    dev_train_dataset.preGenerateMelSpecs(True)

    dev_test_dataset = DCASE2024MachineDataset('./DCASE2024', ('dev', 'test'), machinesName,
                                            extension='.wav', standardize='min-max')

    dev_test_dataset.preGenerateMelSpecs(True)

