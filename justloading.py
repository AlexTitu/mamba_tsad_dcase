import numpy as np
import torch
from test import calculate_total_score

dataset = 'DCASE'
subname = ["dev_bearing", "dev_bearing_per_file", "dev_fan", "dev_fan_per_file", "dev_gearbox_per_file", "dev_gearbox",
           "dev_slider", "dev_slider_per_file", "dev_ToyCar", "dev_ToyCar_per_file",  "dev_ToyTrain",
           "dev_ToyTrain_per_file", "dev_valve", "dev_valve_per_file"]

for subset in subname:
    auc = np.load(f'./result/{dataset}_{subset}/auc_values.npy').tolist()
    pauc = np.load(f'./result/{dataset}_{subset}/pauc_values.npy').tolist()
    mean = calculate_total_score(auc, pauc)
