import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import torch.nn as nn
# from activation_models.models import *
import wandb
from easydict import EasyDict as edict
import argparse
import os
import csv
import matplotlib.pyplot as plt
from source_dataset_training.resnet import *
from collections import defaultdict
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision
from torchvision import models, transforms
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, auc
import torch
from collections import defaultdict
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import random
import time
from easydict import EasyDict as edict
import json
import logging
import timm
from pprint import pprint

def update_best_metrics(eval_dict_val, eval_dict_test, avg_loss_val, avg_loss_test,
                        best_metrics):
    """
    Update the best metrics if the current evaluation metrics are better.

    Parameters:
    - eval_dict_val (dict): Dictionary containing the validation evaluation metrics.
    - eval_dict_test (dict): Dictionary containing the test evaluation metrics.
    - avg_loss_val (float): Average validation loss.
    - avg_loss_test (float): Average test loss.
    - best_metrics (dict): Dictionary containing the best metrics to be updated.

    Returns:
    - best_metrics (dict): Updated dictionary containing the best metrics.
    """

    # Check and update the best validation and test AUROC
    if eval_dict_val["AUROC"] >= best_metrics["val_auroc"]:
        best_metrics["val_auroc"] = eval_dict_val["AUROC"]
        best_metrics["test_auroc"] = eval_dict_test["AUROC"]

    # Check and update the best validation and test AUPR_1
    if eval_dict_val["AUPR_1"] >= best_metrics["val_aupr_1"]:
        best_metrics["val_aupr_1"] = eval_dict_val["AUPR_1"]
        best_metrics["test_aupr_1"] = eval_dict_test["AUPR_1"]

    # Check and update the best validation and test AUPR_0
    if eval_dict_val["AUPR_0"] >= best_metrics["val_aupr_0"]:
        best_metrics["val_aupr_0"] = eval_dict_val["AUPR_0"]
        best_metrics["test_aupr_0"] = eval_dict_test["AUPR_0"]

    # Check and update the best validation and test AURC
    # assuming lower is better for AURC
    if eval_dict_val["AURC"] <= best_metrics["val_aurc"]:
        best_metrics["val_aurc"] = eval_dict_val["AURC"]
        best_metrics["test_aurc"] = eval_dict_test["AURC"]

    # Check and update the best validation and test loss
    # assuming lower is better for loss
    if avg_loss_val <= best_metrics["val_loss"]:
        best_metrics["val_loss"] = avg_loss_val
        best_metrics["test_loss"] = avg_loss_test

    return best_metrics


def save_dict_to_json(dictionary, base_dir="outputs/Cifar10", filename="results.json", run_id=None):
    """
    Saves a dictionary to a JSON file inside a specified folder.

    Parameters:
    - dictionary (dict): The dictionary to save.
    - folder_name (str): The name of the folder where the file will be saved.
    - filename (str): Optional. The name of the JSON file. Default is 'results.json'.
    - run_id (str or None): Optional. If provided, creates a subdirectory with this name under `folder_name`.
    """

    # If run_id is provided, create a subdirectory with this name
    if run_id:
        folder_path = os.path.join(base_dir, str(run_id))
    else:
        folder_path = os.path.join(base_dir)

    # Create the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Construct file path
    file_path = os.path.join(folder_path, filename)

    # Save the dictionary as JSON in the file
    with open(file_path, 'w') as f:
        json.dump(dictionary, f, indent=4)

    print(f"Dictionary saved to {file_path}")

def evaluate_classification_metrics(ys, SRs_list):
    """
    Evaluate and print AUROC and AUPRC for both possible labelings.
    
    Parameters:
    ys (list or np.ndarray): True binary labels.
    SRs_list (list or np.ndarray): Predicted scores.
    
    Returns:
    dict: Dictionary containing AUROC, AUPRC for original and flipped labels.
    """
    ys = np.array(ys)
    SRs_list = np.array(SRs_list)

    # Calculate AUROC
    auroc = roc_auc_score(ys, SRs_list)
    logging.info(f'AUROC: {auroc}')

    # Calculate AUPRC for the original labeling
    auprc_1 = average_precision_score(ys, SRs_list)
    logging.info(f'AUPR_1 (1 is correct): {auprc_1}')
    

    # Calculate AUPRC for the flipped labeling
    auprc_0 = average_precision_score(1 - ys, 1 - SRs_list)
    logging.info(f'AUPR_0 (0 is correct): {auprc_0}')
    
    # SRs_list_sorted = np.sort(SRs_list)[::-1]
    SRs_list_sorted_indices = np.argsort(SRs_list)[::-1]
    ys = 1 - ys[SRs_list_sorted_indices] # now 0 means correct, and 1 means a mistake
    
    coverages = []
    selective_risks = []
    for i in range(10):
        coverage = (i + 1) / 10
        coverages.append(coverage)
        
        selective_risk = np.mean(ys[:int(coverage * len(ys))])
        selective_risks.append(selective_risk)
    
    aurc = auc(coverages, selective_risks)
    logging.info(f'AURC: {aurc}')
    
    return {
        'AUROC': auroc,
        'AUPR_1': auprc_1,
        'AUPR_0': auprc_0,
        'AURC': aurc,
        'coverage': coverages,
        'selective_risk': selective_risks
    }


def generate_10_digit_random_number():
    # Seed the random number generator with the current time
    random.seed(time.time())
    return ''.join([str(random.randint(0, 9)) for _ in range(10)])


def register_hooks(cfg, model, H_W_dim=28):
    activations = defaultdict(list)
    def get_activation(name):
        def hook(model, input, output):
            # print(f"{output.shape=}")
            if H_W_dim == 28:
                # print(f"{output.shape=}")
                found_tensor = True
                if output.shape[2:] == torch.Size([56, 56]):
                    new_tensor = F.max_pool2d(output, kernel_size=2, stride=2)
                    # print(new_tensor.shape)
                elif output.shape[2:] == torch.Size([28, 28]):
                    new_tensor = output
                    # print(new_tensor.shape)
                elif output.shape[1:] == torch.Size([512]) or output.shape[1:] == torch.Size([1000]):
                    output = output.reshape(1, output.shape[1:][0], 1, 1)
                    new_tensor = F.interpolate(output, size=(
                        H_W_dim, H_W_dim), mode='nearest')
                    # print(new_tensor.shape)
                else: # output.shape[2:] == torch.Size([14, 14]):
                    new_tensor = F.interpolate(output, size=(H_W_dim, H_W_dim), mode='nearest')
                    # print(new_tensor.shape)
            if found_tensor:
                activations[name].append(new_tensor.detach().cpu())
        return hook

    layers_to_hook = [
        # 'layer1.0.act2',
        'layer1.1.bn2',
        'layer2.1.bn2',
        'layer3.1.bn2',
        'layer4.1.bn2',
        'avgpool',
        'fc'
    ]

    idx = 0
    for name, layer in model.named_modules():
        print(name)
        if name in layers_to_hook:
            layer.register_forward_hook(get_activation(f"{idx}_{name}"))
            idx += 1
        # else:
        #     layer.register_forward_hook(get_activation(f"{idx}_{name}"))
        #     idx += 1
    return activations


def plot_tsne(embeddings, labels, save_path="xxx.png"):
    """
    Plots the t-SNE of given embeddings with different colors for different labels.
    
    Parameters:
    embeddings (list or np.ndarray): List or array of embeddings.
    labels (list or np.ndarray): Corresponding labels for the embeddings.
    """

    # Convert to numpy arrays if not already
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        idx = labels == label
        plt.scatter(tsne_results[idx, 0], tsne_results[idx,
                    1], label=f'Label {label}', alpha=0.6)

    plt.legend()
    plt.title('t-SNE of Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(save_path)
    plt.show()


def plot_histograms(heuristic_values, labels, save_path="yyy.png"):
    """
    Plots two histograms based on heuristic values and corresponding labels.
    
    Parameters:
    heuristic_values (list or numpy array): The heuristic values.
    labels (list or numpy array): The corresponding labels (0 or 1).
    """
    # Convert to numpy arrays for easier indexing
    heuristic_values = np.array(heuristic_values)
    labels = np.array(labels)

    # Separate the heuristic values based on labels
    values_for_0 = heuristic_values[labels == 0]
    values_for_1 = heuristic_values[labels == 1]

    # Create the histograms
    plt.figure(figsize=(12, 6))

    # Histogram for label 0
    plt.hist(values_for_0, bins=100, alpha=0.7, label='Label 0', color='blue')

    # Histogram for label 1
    plt.hist(values_for_1, bins=100, alpha=0.7, label='Label 1', color='orange')
    

    # Add titles and labels
    plt.title('Histograms of Heuristic Values by Label')
    plt.xlabel('Heuristic Values')
    plt.ylabel('Frequency')

    # Set y-axis to logarithmic scale
    plt.yscale('log')
    
    # Add legend
    plt.legend()
    plt.savefig(save_path)
    # Show the plot
    plt.show()
    
class ActivationsDataset(Dataset):
        def __init__(self, activations, correctness_labels):
            self.activations = activations
            self.correctness_labels = correctness_labels
            self.keys = list(activations.keys())
            self.length = len(activations[self.keys[0]])

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # Extract tensors for the current index from each key
            tensors_to_concat = [self.activations[key][idx] for key in self.keys]

            # Concatenate along the second dimension (d)
            concatenated_tensor = torch.cat(tensors_to_concat, dim=1)
            concatenated_tensor = concatenated_tensor.squeeze(0)
            # Get the correctness label
            label = self.correctness_labels[idx]
            self.feature_dim = concatenated_tensor.shape[1]
            
            return concatenated_tensor, label

class ActivationsDataset_MLP(Dataset):
        def __init__(self, activations, correctness_labels):
            self.activations = activations
            self.correctness_labels = correctness_labels
            self.keys = list(activations.keys())
            self.length = len(activations[self.keys[0]])

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # Extract tensors for the current index from each key
            tensors_to_concat = [self.activations[key][idx].reshape(-1) for key in self.keys]

            # Concatenate along the second dimension (d)
            concatenated_tensor = torch.cat(tensors_to_concat, dim=0)
            # Get the correctness label
            label = self.correctness_labels[idx]
            self.feature_dim = concatenated_tensor.shape[0]
            
            return concatenated_tensor, label


def activation_datasets(cfg, model_symmetry = "Cn"):
    # model_names = timm.list_models(pretrained=True)
    # pprint(model_names)
    # exit()
    # original_model = timm.create_model('resnet18.a1_in1k',
    #                           pretrained=True, num_classes=1000).to(cfg.general.device)
    original_model = models.resnet18(
        weights='IMAGENET1K_V1').to(cfg.general.device)
    
    activations_path = './experiments/ImageNet_activations.pt'
    correctness_labels_path = './experiments/ImageNet_correctness_labels.pt'
    if os.path.exists(activations_path):
        activations = torch.load(activations_path)
        if os.path.exists(correctness_labels_path):
            correctness_labels = torch.load(correctness_labels_path)
            print("Activation and correctness lables files loaded successfully.")
    else:

        # transform = timm.data.create_transform(
        #     **timm.data.transforms_factory.resolve_data_config(original_model.pretrained_cfg, model=original_model))
        
        transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

        # Load the training dataset
        imagenet_test_dataset = datasets.ImageFolder(
            root='/home/guy.b/Imgnet/val', transform=transform)


        imagenet_test_dataloader = DataLoader(imagenet_test_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=4
                                            )
        original_model.eval()  # Set the model to evaluation mode

        # Register hooks to collect activations
        activations = register_hooks(None, original_model)
        print("Activation and correctness lables files don't exist, inferencing.")
        # Pass data through the model to collect activations and correctness labels
        correctness_labels = []
        # Pass data through the model to collect activations
        with torch.no_grad():
            # Assuming your DataLoader returns (inputs, labels) tuples
            counter = 0 
            for inputs, labels in tqdm(imagenet_test_dataloader):
                inputs = inputs.to(cfg.general.device)
                outputs = original_model(inputs)
                _, preds = torch.max(outputs, 1)
                correctness = (preds == labels.to(cfg.general.device)).float().cpu()  # 1 if correct, 0 if incorrect
                correctness_labels.append(correctness)
                counter += 1
                if counter == 10000:
                    break
        logging.info("Activation and correctness lables files saved successfully.")
        def save_activations(activations, correctness_labels, path):
            torch.save(activations, os.path.join(
                path, 'ImageNet_activations.pt'))
            torch.save(correctness_labels, os.path.join(
                path, 'ImageNet_correctness_labels.pt'))
        save_activations(activations, correctness_labels, './experiments')
    logging.info(f"Accuracy of original model on dataset: {sum(correctness_labels) / len(correctness_labels)}")
    # Create the dataset
    if model_symmetry == "Cn":
        activations_dataset = ActivationsDataset(activations, correctness_labels)    
    elif model_symmetry == "none":
        activations_dataset = ActivationsDataset_MLP(
            activations, correctness_labels)
        
    d_0 = activations_dataset[0][0].shape[0]
    activations_shape  = activations_dataset[0][0].shape


    # Logging the values with formatted strings
    logging.info(f"d_0: {d_0}, activations_shape: {activations_shape}")
    
    # Set a fixed seed for reproducibility
    torch.manual_seed(42)

    # Define the lengths for train, val, and test sets
    total_size = len(activations_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Randomly split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        activations_dataset, [train_size, val_size, test_size])
    
    # subset_indices = list(range(1000))
    # train_dataset = Subset(train_dataset, subset_indices)
    
    # Concatenate the datasets
    train_and_val_dataset = ConcatDataset([train_dataset, val_dataset])

    # Check the lengths of the splits
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    activations_loader_train = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4)
    activations_loader_val = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4)
    activations_loader_test = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    activations_loader_train_and_val = DataLoader(
        train_and_val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    return activations_dataset, activations_loader_train, activations_loader_val, activations_loader_test, activations_loader_train_and_val, d_0, original_model

def get_SR_baseline(results_dict, activations_loader_test):
    embed_preds_list = []
    softmaxed_preds_list = []
    SRs_list = []
    ys = []

    for batch in tqdm(activations_loader_test):
        x, y = batch[0], batch[1]
        model_pred = x[:, -100:, 15, 15]
        embed = x[:, -522:-100, 15, 15]
        # Apply softmax to the predictions
        softmaxed_preds = torch.nn.functional.softmax(model_pred, dim=1)

        # Collect the softmaxed predictions and labels
        softmaxed_preds_list.append(softmaxed_preds)
        embed_preds_list.append(embed)

        SRs, _ = torch.max(torch.nn.functional.softmax(
            model_pred, dim=1), dim=1)
        SRs = SRs.tolist()
        SRs_list = SRs_list + SRs
        ys = ys + y.squeeze(1).tolist()

    eval_dict = evaluate_classification_metrics(
        ys, SRs_list)

    results_dict["SR baseline"] = eval_dict
    wandb.log({"epoch": 0,
                "AUROC SR baseline": eval_dict["AUROC"],
                "AUPR_1 SR baseline": eval_dict["AUPR_1"],
                "AUPR_0 SR baseline": eval_dict["AUPR_0"],
                "AURC SR baseline": eval_dict["AURC"]})

    # Concatenate the lists along the batch dimension

    embed_preds_tensor = torch.cat(embed_preds_list, dim=0)
    ys_tensor = torch.tensor(ys)

    plot_tsne(embed_preds_tensor, ys_tensor)
    plot_histograms(SRs_list, ys)



def train_loop(model, loader, optimizer, criterion, device, mlp_model=False):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_auroc = 0
    num_batches = len(loader)

    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        if mlp_model:
            # x = x.reshape(batch_size, -1)
            pred = model(x=x)
        else:
            pred, features = model(x)
        loss = criterion(pred, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    # Average metrics over all batches
    avg_accuracy = total_accuracy / num_batches
    avg_auroc = total_auroc / num_batches
    avg_loss = total_loss / num_batches

    return avg_loss, avg_auroc, avg_accuracy

def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    # total_accuracy = 0
    # total_auroc = 0
    num_batches = len(loader)
    preds = []
    ys = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            ys.append(y)
            pred, features = model(x)
            preds.append(pred)
            loss = criterion(pred, y.float())
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    preds = torch.cat(preds, dim=0).reshape(-1)
    ys = torch.cat(ys, dim=0).reshape(-1)
    return avg_loss, preds.cpu().tolist(), ys.cpu().tolist()


class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        temperature_scaled_logits = logits / self.temperature
        return temperature_scaled_logits

def set_temperature(model, validation_loader):
    """
    Tune the temperature of the model using the validation set.
    """
    model.eval()
    temperature_scaling = TemperatureScaling().to(next(model.parameters()).device)
    nll_criterion = nn.CrossEntropyLoss().to(next(model.parameters()).device)

    # optimizer = torch.optim.LBFGS(
    #     [temperature_scaling.temperature], lr=0.01, max_iter=50)
    # optimizer = torch.optim.Adam([temperature_scaling.temperature], lr=0.01)
    # optimizer = torch.optim.SGD(
    #     [temperature_scaling.temperature], lr=0.01, momentum=0.9)

    optimizer = torch.optim.Adagrad(
        [temperature_scaling.temperature], lr=0.01)

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in validation_loader:
            logits = inputs[:, -10:, 1, 1]
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list).to(next(model.parameters()).device)
    labels = torch.cat(labels_list).to(next(model.parameters()).device)
    for _ in tqdm(range(1)):
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(temperature_scaling(
                logits), labels.squeeze().long())
            loss.backward()
            return loss

        optimizer.step(eval)
    logging.info(f'Optimal temperature: {temperature_scaling.temperature.item()}')

    return temperature_scaling

def get_tmp_scale_baseline(cfg, results_dict, activations_loader_test, temperature_scaling_model):
    SRs_list = []
    ys = []
    for batch in tqdm(activations_loader_test):
        x, y = batch[0], batch[1]
        model_pred = x[:, -100:, 15, 15].to(cfg.general.device)
        model_pred = temperature_scaling_model(model_pred)
        SRs, _ = torch.max(torch.nn.functional.softmax(
            model_pred, dim=1), dim=1)
        SRs = SRs.tolist()
        SRs_list = SRs_list + SRs
        ys = ys + y.squeeze(1).tolist()

    eval_dict = evaluate_classification_metrics(
        ys, SRs_list)

    results_dict["SR + tmp scale baseline"] = eval_dict
    wandb.log({"epoch": 0,
                "AUROC SR + tmp scale baseline": eval_dict["AUROC"],
                "AUPR_1 SR + tmp scale baseline": eval_dict["AUPR_1"],
                "AUPR_0 SR + tmp scale baseline": eval_dict["AUPR_0"],
                "AURC SR + tmp scale baseline": eval_dict["AURC"]})

class SimpleResNet(nn.Module):
    def __init__(self, input_channels, hidden_dim=256, num_classes=1):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim , kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        self.conv4 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=2, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(hidden_dim)
        
        self.conv5 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=2, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(hidden_dim)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # max_softmax = max(F.softmax(x[:, -100:, 15, 15], dim=1))
        
        # Block 1
        # identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        # x += identity  # Adding residual 
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        # identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        # x += identity  # Adding residual
        x = F.relu(x)
        x = self.pool(x)

        # Block 3
        # identity = x
        x = self.conv3(x)
        x = self.bn3(x)
        # x += identity  # Adding residual
        x = F.relu(x)
        x = self.pool(x)

        # Block 4
        # identity = x
        x = self.conv4(x)
        x = self.bn4(x)
        # x += identity  # Adding residual
        x = F.relu(x)
        x = self.pool(x)
        
        # # Block 5
        # # identity = x
        # x = self.conv5(x)
        # x = self.bn5(x)
        # # x += identity  # Adding residual
        # x = F.relu(x)
        # x = self.pool(x)
            
        assert x.shape[2] == 1 and x.shape[3] == 1, f"Expected spatial dimensions to be 1x1, but got {x.shape[2:]} instead."

        features = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(features)
        
        # x = x + max_softmax
        x = self.sigmoid(x)
        return x, features

class SimpleMLP(nn.Module):
    def __init__(self, input_channels, hidden_dim=256, num_classes=1):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        self.fc_final = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.fc_final(x)

        x = self.sigmoid(x)

        return x, x


def run(cfg):
    
    # Configure logging settings if not already configured
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


    # Construct argument parser
    parser = argparse.ArgumentParser()

    # Add arguments based on cfg keys
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size)
    parser.add_argument('--dataset', type=str, default=cfg.dataset)
    parser.add_argument('--model_symmetry', type=str,
                        default=cfg.model_symmetry)
    parser.add_argument('--hidden_dim', type=int, default=cfg.hidden_dim)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    parser.add_argument('--weight_decay', type=float, default=cfg.weight_decay)
    parser.add_argument('--num_epochs', type=int, default=cfg.num_epochs)
    parser.add_argument('--seed', type=int, default=cfg.seed)

    # Parse arguments
    args = parser.parse_args()

    # Access parsed arguments
    logging.info(f"Parsed arguments: {args}")


    cfg = edict(vars(args))
    cfg.general = edict({})
    cfg.general.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    results_dict = edict({})
    
    run_id = generate_10_digit_random_number()
    cfg.general.run_id = run_id
    tag = f"run_id_{run_id}||model_{cfg.model_symmetry}||batch_size_{cfg.batch_size}||hidden_dim_{cfg.hidden_dim}||lr_{cfg.lr}||weight_decay_{cfg.weight_decay}||num_epochs_{cfg.num_epochs}"

    wandb.init(settings=wandb.Settings(
        start_method='thread'), project="Synthetic experiment Neurons", name=tag, config=cfg)


    logging.info("""
    #######################################
    Getting activations dataset.
    #######################################
    """)

    activations_dataset, activations_loader_train, activations_loader_val, activations_loader_test, activations_loader_train_and_val, d_0, original_model = activation_datasets(cfg=cfg, model_symmetry="Cn")
    
    logging.info("""
    #######################################
    Getting SR baseline.
    #######################################
    """)
    get_SR_baseline(results_dict=results_dict, activations_loader_test=activations_loader_test)

    logging.info("""
    #######################################
    Getting Tmp Scale baseline.
    #######################################
    """)
    # Tune the temperature on the validation set
    temperature_scaling_model = set_temperature(
        model=original_model, validation_loader=activations_loader_train_and_val)

    get_tmp_scale_baseline(cfg=cfg,
        results_dict=results_dict, activations_loader_test=activations_loader_test, temperature_scaling_model=temperature_scaling_model)

    logging.info("""
    #######################################
    Neural Network approach.
    #######################################
    """)
    
    if cfg.model_symmetry == "Cn":
        torch.manual_seed(cfg.seed)
        model = SimpleResNet(input_channels=d_0, hidden_dim=cfg.hidden_dim
                             ,num_classes=1).to(
            cfg.general.device)

        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Number of parameter of Cn-based: {total_params}")
    
    elif cfg.model_symmetry == "none":
        activations_dataset, activations_loader_train, activations_loader_val, activations_loader_test, activations_loader_train_and_val, d_0, original_model = activation_datasets(cfg=cfg, model_symmetry="none")
        torch.manual_seed(cfg.seed)
        model = SimpleMLP(input_channels=d_0, hidden_dim=cfg.hidden_dim, num_classes=1).to(
            cfg.general.device)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Number of parameter of MLP: {total_params}")
    else:
        pass
    
    optim = torch.optim.Adam(model.parameters(), 
                             # lr=0.01,
                             lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    
    sched = torch.optim.lr_scheduler.StepLR(
        optim, step_size=5, gamma=0.5)
    
    crit = nn.BCELoss()

    best_metrics = {
    "val_auroc": 0,
    "test_auroc": 0,
    
    "val_aupr_1": 0,
    "test_aupr_1": 0,
    
    "val_aupr_0": 0,
    "test_aupr_0": 0,
    
    "val_aurc": float('inf'),
    "test_aurc": float('inf'),
    
    "val_loss": float('inf'),
    "test_loss": float('inf')
}
    
    for epoch in tqdm(range(cfg.num_epochs)):
        avg_loss, total_auroc, total_accuracy = train_loop(
            model=model, loader=activations_loader_train, optimizer=optim, criterion=crit, device=cfg.general.device)
        wandb.log({"epoch": epoch+1,
                   "Train Loss": avg_loss,
                   "Train AUROC": total_auroc,
                   "Train Accuracy": total_accuracy})

        avg_loss_val, preds_val, ys_val = eval_loop(
            model=model, loader=activations_loader_val, criterion=crit, device=cfg.general.device)
        eval_dict_val = evaluate_classification_metrics(ys_val, preds_val)
        wandb.log({"epoch": epoch+1,
                   "Val Loss": avg_loss_val,
                   "Val AUROC": eval_dict_val["AUROC"],
                   "Val AUPR_1": eval_dict_val["AUPR_1"],
                   "VAL AUPR_0": eval_dict_val["AUPR_0"],
                   "Val AURC": eval_dict_val["AURC"]
                   })

        avg_loss_test, preds_test, ys_test = eval_loop(
            model=model, loader=activations_loader_test, criterion=crit, device=cfg.general.device)
        eval_dict_test = evaluate_classification_metrics(ys_test, preds_test)
        wandb.log({"epoch": epoch+1,
                   "Test Loss": avg_loss_test,
                   "Test AUROC": eval_dict_test["AUROC"],
                   "Test AUPR_1": eval_dict_test["AUPR_1"],
                   "Test AUPR_0": eval_dict_test["AUPR_0"],
                   "Test AURC": eval_dict_test["AURC"]
                   })
        

        update_best_metrics(eval_dict_val=eval_dict_val, eval_dict_test=eval_dict_test,
                            avg_loss_val=avg_loss_val, avg_loss_test=avg_loss_test, best_metrics=best_metrics)
        
        wandb.log({"epoch": epoch+1,
                   "Best Val Loss": best_metrics["val_loss"],
                   "Best Val AUROC": best_metrics["val_auroc"],
                   "Best Val AUPR_1": best_metrics["val_aupr_1"],
                   "Best Val AUPR_0": best_metrics["val_aupr_0"],
                   "Best Val AURC": best_metrics["val_aurc"],
                   
                   "Best Test Loss": best_metrics["test_loss"],
                   "Best Test AUROC": best_metrics["test_auroc"],
                   "Best Test AUPR_1": best_metrics["test_aupr_1"],
                   "Best Test AUPR_0": best_metrics["test_aupr_0"],
                   "Best Test AURC": best_metrics["test_aurc"]})

        sched.step()
    
    results_dict["Ours"] = best_metrics
    save_dict_to_json(dictionary=results_dict, base_dir="outputs/Cifar10",
                      filename="metrics.json", run_id=run_id)
    wandb.finish()
    
    exit()

    
