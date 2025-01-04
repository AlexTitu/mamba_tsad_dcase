import numpy as np
import torch
from test import calculate_total_score

dataset = 'DCASE'
subname = ["dev_bearing", "dev_fan", "dev_gearbox", "dev_slider", "dev_ToyCar", "dev_ToyTrain", "dev_valve"]

for subset in subname:
    auc = np.load(f'./result/{dataset}_{subset}/auc_values.npy').tolist()
    pauc = np.load(f'./result/{dataset}_{subset}/pauc_values.npy').tolist()
    mean = calculate_total_score(auc, pauc)
