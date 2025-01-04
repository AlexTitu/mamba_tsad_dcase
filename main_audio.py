import os
import os.path as path
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from test import test_model
from dataset import AudioAnomalyDataset, load_dataset, stratified_split
from model import DecomposeMambaSSM
from trainer import Trainer
from detect import detect

from baseline_read_dataset import read_dataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

METRICS = ["P_af", "R_af", "F1_af"]


def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help='dataset',
                        choices=["SMD", "SWaT", "NASA", "DCASE"]
                        )
    parser.add_argument("--double", type=bool, default=False, help='use double')
    parser.add_argument("--device", type=str, default="gpu", help='device')
    parser.add_argument("--tag", type=str, default=None, help='tag')
    parser.add_argument("--test", action="store_true", help="test only")
    parser.add_argument("--pre_filter", action="store_true", help="switch on HP filter")
    parser.add_argument("--decomp", action="store_true", help="switch on AMA")
    args = parser.parse_args()

    if args.dataset == "NASA":
        subdata = ["A-4", "T-1", "C-2"]
    elif args.dataset == "SMD":
        subdata = ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7", "machine-1-6"]
    elif args.dataset == "DCASE":
        subdata = ["dev_bearing", "dev_fan", "dev_gearbox", "dev_slider", "dev_ToyCar", "dev_ToyTrain", "dev_valve"]
    else:
        subdata = None

    result_df = pd.DataFrame(columns=["Datasets"] + METRICS)
    for subname in subdata:
        if args.tag is not None:
            output_dir = f"./result/{args.dataset}_{subname}_{args.tag}"
        else:
            output_dir = f"./result/{args.dataset}_{subname}"
        os.makedirs(path.join(output_dir), exist_ok=True)

        dataset_source, dataset_target = load_dataset('./DCASE2024', ('dev', 'train'), extension='.npy'
                                                      , machineType=subname)

        source_train, source_val, target_train, target_val = stratified_split(
            dataset_source, dataset_target, validation_size=0.30, random_state=42)

        # get method info
        train_config = {"seed": 42, "lr": 0.001, "optim_conf": {"weight_decay": 0.00001},
                        "schedule_conf": {"step_num": 5, "decay": 0.9}, "batch_size": 100, "max_epochs": 100,
                        "log_period": 10, "num_recent_models": -1, "early_stop_count": -1, "test_bsz": 1,
                        "norm_type": "norm",
                        }

        # other config
        window_length = 62

        init_seed(train_config["seed"])

        # training
        train_dataset = AudioAnomalyDataset((source_train, target_train), window_length=window_length,
                                            standardize=False, extension='.npy')

        # train_dataset.preGenerateMelSpecs()
        val_dataset = AudioAnomalyDataset((source_val, target_val), window_length=window_length, standardize=False,
                                          extension='.npy')
        # val_dataset.preGenerateMelSpecs()

        clf = DecomposeMambaSSM(
            input_size=128,
            window_size=window_length,
            pre_filter=args.pre_filter, # HP filter
            decomp=args.decomp # AMA
        )

        if args.double:
            clf = clf.double()
        logging.info("Training...")
        # train
        trainer = Trainer(clf,
                          output_dir=output_dir,
                          init_model=None,
                          device=args.device,
                          **train_config
                          )
        if not args.test:
            trainer.fit(train_dataset, val_dataset=val_dataset)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            test_model(output_dir, clf, window_length, subname, device)
