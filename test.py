import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean
from dataset import AudioAnomalyDataset
from torch.nn.functional import mse_loss
from dataset import load_dataset


# Define the function H with threshold
def H(x, threshold=0):
    return 1 if x > threshold else 0


# Function to calculate AUC
def calculate_auc(normal_scores, anomalous_scores, threshold=0):
    N_d = len(normal_scores)
    N_plus = len(anomalous_scores)

    auc_sum = 0

    for x_j in anomalous_scores:
        for x_i in normal_scores:
            auc_sum += H(x_j - x_i, threshold)

    auc = auc_sum / (N_d * N_plus)
    return auc


# Function to calculate pAUC
def calculate_pauc(normal_scores, anomalous_scores, p, threshold=0):
    N_d = len(normal_scores)
    N_plus = len(anomalous_scores)

    pN_d = int(np.floor(p * N_d))

    if pN_d == 0 or N_plus == 0:
        return 0  # Handle edge case when counts are zero

    normal_scores = sorted(normal_scores, reverse=True)[:pN_d]

    pauc_sum = 0
    for x_j in anomalous_scores:
        for x_i in normal_scores:
            pauc_sum += H(x_j - x_i, threshold)

    pauc = pauc_sum / (pN_d * N_plus)
    return pauc


# Function to calculate the total score Ω
def calculate_total_score(auc_values, pauc_values):
    return hmean(auc_values + pauc_values)


# switching off autograd for eval
def test_loop(model, data_loader, steps, window_length, stride, num_windows_per_file, device):
    scores = {
        'normal': {'source': [], 'target': []},
        'anomalous': {'source': [], 'target': []}
    }
    # set the model in eval mode
    model.eval()  # Set model to evaluation mode
    total_test_loss = 0
    # start valid
    for real_signal, target_signal, label, origin in data_loader:
        hidden = None
        real_signal = real_signal.to(device)
        target_signal = target_signal.to(device)
        with torch.no_grad():
            # Prepare lists to hold predictions and ground-truth windows
            predictions = []
            ground_truths = []

            for window_index in range(num_windows_per_file):
                start = window_index * stride
                end = start + window_length

                # Extract the window
                window_input = real_signal[:, start:end]
                window_target = target_signal[:, start:end]

                if hidden is not None:
                    res = model(window_input, hidden)
                else:
                    res = model(window_input)

                hidden = getattr(res, "hidden", None)
                # Append results and targets for concatenation
                predictions.append(res["x"])  # Adjust if key differs
                ground_truths.append(window_target)

            # Concatenate predictions and ground-truths along the time dimension
            concatenated_predictions = torch.cat(predictions, dim=1)  # Shape: (batch_size, full_length)
            concatenated_ground_truths = torch.cat(ground_truths, dim=1)

            # Shape: (batch_size, full_length)
            squared_differences = (concatenated_predictions - concatenated_ground_truths) ** 2
            # Compute MSE for each item in the batch
            mse_per_item = torch.mean(squared_differences, dim=(1, 2))  # Shape: (batch_size,)
            for item_index in range(len(label)):
                if origin[item_index] == 'source':
                    if label[item_index] == 0:
                        scores['normal']['source'].append(mse_per_item[item_index].cpu().detach().item())
                    else:
                        scores['anomalous']['source'].append(mse_per_item[item_index].cpu().detach().item())
                else:
                    if label[item_index] == 0:
                        scores['normal']['target'].append(mse_per_item[item_index].cpu().detach().item())
                    else:
                        scores['anomalous']['target'].append(mse_per_item[item_index].cpu().detach().item())
            total_test_loss += torch.mean(mse_per_item)
    print(f"Total loss mean: {total_test_loss/steps}")

    return scores


# -------------------------------------
# ------- Main Test Function ----------
# -------------------------------------

def test_model(models_dir, model, test_window_length, machine_name, device):
    # plot the training and val losses
    print(f"[INFO] testing the model for {models_dir}")
    previous_state = torch.load(f"{models_dir}/train_loss_history.pt")
    previous_model = torch.load(f"{models_dir}/best_valid_model")
    model.load_state_dict(previous_model)

    plt.style.use("ggplot")
    H = previous_state

    # Plotting loss on train and evaluation
    plt.figure("total_loss").clear()
    plt.plot(H["train_loss"], label="train_loss", linestyle="solid")
    plt.plot(H["val_loss"], label="validation_loss", linestyle="solid")
    plt.title("Evolutia functiei de cost in timpul antrenarii")
    plt.xlabel("# Epoca")
    plt.ylabel("Cost")
    plt.legend(loc="upper right")
    plt.savefig(f"{models_dir}/train_val_graph_mamba.png")

    # Parameters
    domains = ['source', 'target']  # predefine domains
    p = 0.1  # false positive rate for pAUC

    auc_values = []
    pauc_values = []

    dataset_source, dataset_target = load_dataset('./DCASE2024', ('dev', 'test'), extension='.npy'
                                                  , machineType=machine_name)

    # Calculate AUC and pAUC for each combination of machine type, section, and domain
    dev_test_dataset = AudioAnomalyDataset((dataset_source, dataset_target), window_length=test_window_length,
                                           standardize=False, extension='.npy', is_testing=True)
    print(f'Number of test Samples: {len(dev_test_dataset)}')

    testDataLoader = DataLoader(dev_test_dataset, batch_size=30)

    testSteps = len(testDataLoader.dataset) // 30

    scores = test_loop(model, testDataLoader, testSteps, dev_test_dataset.window_length, dev_test_dataset.stride,
                       dev_test_dataset.num_windows_per_file, device)

    # Calculate pAUC across both domains for each machine type
    all_normal_scores = scores['normal']['source'] + scores['normal']['target']
    all_anomalous_scores = scores['anomalous']['source'] + scores['anomalous']['target']
    pauc = calculate_pauc(all_normal_scores, all_anomalous_scores, p)
    pauc_values.append(pauc)

    for d in domains:
        auc = calculate_auc(scores['normal'][d], all_anomalous_scores)
        auc_values.append(auc)

    np.save(f'./{models_dir}/auc_values.npy', auc_values)
    np.save(f'./{models_dir}/pauc_values.npy', pauc_values)
    # Calculate the total score
    total_score = calculate_total_score(auc_values, pauc_values)

    print(f'Total Score (Ω): {total_score}')
    np.save(f'./{models_dir}/total_score.npy', total_score)

