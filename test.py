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
def test_loop(model, data_loader, steps, device):
    scores = {
        'normal': {'source': [], 'target': []},
        'anomalous': {'source': [], 'target': []}
    }
    # set the model in eval mode
    model.eval()  # Set model to evaluation mode
    total_test_loss = 0
    with torch.no_grad():
        for real_signal, target_signal, tags, origin in data_loader:
            hidden = None  # Reset hidden state for each file
            batch_loss = 0
            label = tags[0].item()
            origin = origin[0]
            for fragment_index in range(len(real_signal)):  # Sequential windows per file
                # Move data to device
                real_signal_fragment = real_signal[fragment_index].unsqueeze(0).to(device)
                target_signal_fragment = target_signal[fragment_index].unsqueeze(0).to(device)
                # Forward pass
                res = model(real_signal_fragment, hidden=hidden)  # Pass hidden state
                reconstructed = res["x"]  # Reconstructed output
                hidden = res["hidden"]  # Update hidden state for the next window

                batch_loss += mse_loss(reconstructed, target_signal_fragment)
            batch_loss = batch_loss / 10
            total_test_loss += batch_loss
            if origin == 'source':
                if label == 0:
                    scores['normal']['source'].append(batch_loss)
                else:
                    scores['anomalous']['source'].append(batch_loss)
            else:
                if label == 0:
                    scores['normal']['target'].append(batch_loss)
                else:
                    scores['anomalous']['target'].append(batch_loss)

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

    testDataLoader = DataLoader(dev_test_dataset, batch_size=9)

    testSteps = len(testDataLoader.dataset) // 9

    scores = test_loop(model, testDataLoader, testSteps, device)

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

