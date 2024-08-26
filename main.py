import os
import os.path as path
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import RecDataset
from model import DecomposeMambaSSM
from trainer import Trainer
from detect import detect

from baseline_read_dataset import read_dataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

METRICS = ["P_af", "R_af", "F1_af"]

def get_q(dataset, subname=None):
    level = 0.98
    if dataset == "NASA":
        if subname == "T-1":
            q = 0.1
            level = 0.9
        else:
            q = 0.01
            level = 0.98
    elif dataset == "SMD":
        q = 0.005
    elif dataset == "SWaT":
        q = 0.01  # confirmed
        level = 0.98
    else:
        raise ValueError

    return q, level

def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    # get parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help='dataset',
                        choices=["SMD", "SWaT", "NASA"]
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
        # subdata = ["A-4"]
    elif args.dataset == "SMD":
        subdata = ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7", "machine-1-6"]
    else:
        subdata = [None]

    result_df = pd.DataFrame(columns=["Datasets"] + METRICS)
    # TODO: test seed
    for seed in [598]:
        for subname in subdata:
            if args.tag is not None:
                output_dir = f"./result/{args.dataset}_{subname}_{args.tag}_{seed}"
            else:
                output_dir = f"./result/{args.dataset}_{subname}_{seed}"
            os.makedirs(path.join(output_dir), exist_ok=True)

            # get dataset
            train_data, test_data, test_label = read_dataset(args.dataset, subname)

            # get method info
            train_config = {"seed": 598, "lr": 0.001, "optim_conf": {"weight_decay": 0.00001},
                            "schedule_conf": {"step_num": 5, "decay": 0.9}, "batch_size": 128, "max_epochs": 7,
                            "log_period": 10, "num_recent_models": -1, "early_stop_count": -1, "test_bsz": 1,
                            "norm_type": "norm",
                            }

            # other config
            window_length = 100
            test_window_length = window_length
            test_align = "nonoverlap"

            # init_seed(train_config["seed"])
            init_seed(seed)
            # training
            train_dataset = RecDataset(train_data, label=np.zeros_like(train_data), dtype=np.float64 if args.double else np.float32,
                                       partition=True, shuffle=False, window_length=window_length, normalization_type=train_config["norm_type"])

            clf = DecomposeMambaSSM(
                input_size=train_dataset.input_dim,
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
                trainer.fit(train_dataset, val_dataset=None)

            test_dataset = RecDataset(test_data, test_label, dtype=np.float64 if args.double else np.float32,
                                      partition=False, shuffle=False, window_length=test_window_length,
                                      xscaler=train_dataset.xscaler, align=test_align)

            test_dataloader = DataLoader(test_dataset, batch_size=train_config["test_bsz"], num_workers=0)
            init_dataset = RecDataset(train_data, np.zeros(train_data.shape[0]), dtype=np.float64 if args.double else np.float32,
                                      partition=False, shuffle=False, window_length=test_window_length,
                                      xscaler=train_dataset.xscaler, align=test_align)
            init_dataloader = DataLoader(init_dataset, batch_size=train_config["test_bsz"], num_workers=0)

            results, labels = detect(clf, trainer.final_model, test_dataloader, device=args.device,
                                     init_dataloader=init_dataloader, pot_params=get_q(args.dataset, subname))

            dataname = subname if subname is not None else args.dataset

            row = {"Datasets": f"{dataname}_{seed}"}
            for m in METRICS:
                if m in results:
                    row.update({m: results[m]})
            result_df = pd.concat([result_df, pd.DataFrame(row, index=[0])], ignore_index=True)

        # save results
        result_df = result_df.sort_values(by="Datasets")
        avg_df = pd.Series(result_df.mean(numeric_only=True))
        avg_df["Datasets"] = "Avg"
        result_df = result_df.append(avg_df, ignore_index=True)
        if args.tag is not None:
            result_df.to_csv(path.join(path.abspath(path.join("./result", f"{args.dataset}_{args.tag}_{seed}_result.csv"))), index=False)
        else:
            result_df.to_csv(path.join(path.abspath(path.join("./result", f"{args.dataset}_{seed}_result.csv"))), index=False)
