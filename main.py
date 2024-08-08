
"""
Description: This script performs the following tasks on the Zinc12k dataset:

0) Train a GNN model on the dataset.
1) Load activations from a pretrained model.
2) Create a dataset from these activations:
   a) Each data point is a tuple (Activations_i, gap_from_groundtruth_i, ground_truth_i).
3) Split the dataset randomly.
4) Train a new model on top of these activations to:
   a) Predict whether the prediction is correct or not.
   b) Predict the gap from the ground truth.
5) Evaluate the new model using:
   a) AUROC, AUPR0, AUPR1, AURC
   b) MAE (Mean Absolute Error)
"""
# TODO: Add support of config file
# TODO: Add support to wandb sweep
# TODO: add baselines
import matplotlib.pyplot as plt
from PIL import Image
from easydict import EasyDict as edict
import argparse
import yaml
from tqdm import tqdm
import pandas as pd
import pickle
import csv
import os
import wandb
import torch
from torchvision import models
# from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, auc
from torch_geometric.nn import GINEConv, global_add_pool
import torch
import numpy as np
import random
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import torch_geometric.data as data
from torch.utils.data import Subset

##########################################################
#                      Config                            #
##########################################################


# Define a function to load configurations from the YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args(config):
    parser = argparse.ArgumentParser(
        description='Process configuration parameters.')

    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', type=str_to_bool, default=value)
        elif isinstance(value, int):
            parser.add_argument(f'--{key}', type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(f'--{key}', type=float, default=value)
        else:
            parser.add_argument(f'--{key}', type=str, default=value)

    args = parser.parse_args()
    return args


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

##########################################################
#                     Data saver                         #
##########################################################

def save_metrics_to_csv(metrics, filename='metrics.csv'):
    """
    Saves the metrics to a CSV file.

    Parameters:
    - metrics (dict): Dictionary of metrics to save.
    - filename (str): The name of the CSV file to save the metrics. Default is 'metrics.csv'.
    """
    fieldnames = metrics.keys()
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)

##########################################################
#                Train original model                    #
##########################################################

class CustomGINE(torch.nn.Module):
    """
    Custom GINE layer for the GNN network.

    Parameters:
    - in_dim (int): Input dimension.
    - emb_dim (int): Embedding dimension.
    - track_running_stats (bool): Whether to track running stats in BatchNorm layers.
    - num_edge_emb (int): Number of edge embeddings. Default is 4.
    """
    def __init__(self, in_dim, emb_dim, track_running_stats, num_edge_emb=4):
        super().__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(
                emb_dim, track_running_stats=track_running_stats),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.layer = GINEConv(nn=mlp, train_eps=True)
        self.edge_embedding = torch.nn.Embedding(
            num_embeddings=num_edge_emb, embedding_dim=in_dim
        )
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, self.edge_embedding(edge_attr))


class GNNnetwork(torch.nn.Module):
    """
    GNN network consisting of multiple GINE layers and BatchNorm layers.

    Parameters:
    - num_layers (int): Number of layers.
    - in_dim (int): Input dimension.
    - emb_dim (int): Embedding dimension.
    - add_residual (bool): Whether to add residual connections. Default is False.
    - track_running_stats (bool): Whether to track running stats in BatchNorm layers.
    - num_tasks (int): Number of tasks. Default is None.
    """
    def __init__(
        self,
        num_layers,
        in_dim,
        emb_dim,
        add_residual=False,
        track_running_stats=True,
        num_tasks: int = None,
    ):

        super().__init__()

        self.emb_dim = emb_dim

        self.feature_encoder = torch.nn.Embedding(
            num_embeddings=in_dim, embedding_dim=emb_dim
        )

        self.gnn_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                CustomGINE(
                    emb_dim,
                    emb_dim,
                    track_running_stats=track_running_stats,
                )
            )
            self.bn_layers.append(
                torch.nn.BatchNorm1d(
                    emb_dim, track_running_stats=track_running_stats)
            )

        self.add_residual = add_residual
        self.pool = global_add_pool

        self.final_layers = None
        if num_tasks is not None:
            emb_dim = emb_dim
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 * emb_dim,
                                out_features=num_tasks),
            )

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.feature_encoder(x.squeeze())

        for gnn, bn in zip(self.gnn_layers, self.bn_layers):
            h = torch.relu(bn(gnn(x, edge_index, edge_attr)))

            if self.add_residual:
                x = h + x
            else:
                x = h

        x = self.pool(x, batch)
        out = self.final_layers(x)

        return out


def train(model, optimizer, loader, device):
    """
    Trains the model for one epoch.

    Parameters:
    - model (torch.nn.Module): The model to train.
    - optimizer (torch.optim.Optimizer): The optimizer for training.
    - loader (DataLoader): The data loader for training data.
    - device (torch.device): The device to run the training on.

    Returns:
    - float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    """
    Evaluates the model.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - loader (DataLoader): The data loader for evaluation data.
    - device (torch.device): The device to run the evaluation on.

    Returns:
    - float: The average error for the evaluation.
    """
    model.eval()
    total_error = 0
    for data in tqdm(loader, desc="Testing"):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)

##########################################################
def train_from_scratch(cfg, use_split_indices=False):
    """
    Trains the GNN model from scratch.

    Parameters:
    - use_split_indices (bool): Whether to use pre-split indices for training. Default is False.
    - model_name (str): The name of the model for saving purposes. Default is 'zinc-model'.
    - epochs (int): The number of epochs to train the model. Default is 2.

    Returns:
    - dict: Metrics from the best model during training.
    """
    model_name = cfg.original_model__model_name


    path = "/tmp"

    train_dataset = ZINC(path, subset=True, split='train')
    if use_split_indices == True:
        model_name = model_name + '_trained_on_5k_only'
        # Get indices for the split
        train_indices, _ = load_train_split_indices()
        # Create subsets using the indices
        train_dataset = Subset(train_dataset, train_indices)



    val_dataset = ZINC(path, subset=True, split='val')
    test_dataset = ZINC(path, subset=True, split='test')

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.original_model__batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.original_model__batch_size)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.original_model__batch_size)


    device = torch.device(f'cuda:{cfg.device}' if torch.cuda.is_available() else 'cpu')
    

    model = GNNnetwork(
        num_layers=cfg.original_model__num_layers,
        in_dim=cfg.original_model__in_dim,
        emb_dim=cfg.original_model__emb_dim,
        add_residual=cfg.original_model__add_residual,
        track_running_stats=cfg.original_model__track_running_stats,
        num_tasks=cfg.original_model__num_tasks,
    ).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    cfg.original_model_parmas = total_params
    if os.path.isfile(model_name + '.pt'):
        print(f"{model_name} exists. Inferencing using that model...")
        model.load_state_dict(torch.load(model_name + '.pt'))
        val_mae = test(model, val_loader, device)
        test_mae = test(model, test_loader, device)
        print(f'Final Test Loss: {test_mae:.4f}')
        metrics = {
            'model': model_name,
            'best_val_loss': val_mae,
            'best_test_loss': test_mae,
        }
        return metrics
    else:
        print(f"{model_name} does not exist. Will train...")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.original_model__lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=cfg.original_model__step_size,
                                                gamma=cfg.original_model__gamma,
                                                )
    best_val_mae = test_at_best_val = 1e9
    for epoch in range(0, cfg.original_model__epochs):
        loss = train(model, optimizer, train_loader, device)
        val_mae = test(model, val_loader, device)
        test_mae = test(model, test_loader, device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            test_at_best_val = test_mae
            metrics = {
                'model': model_name,
                'epoch': epoch + 1,
                'train_loss': loss,
                'val_loss': val_mae,
                'test_loss': test_mae,
                'best_val_loss': best_val_mae,
                'best_test_loss': test_at_best_val,
            }
            torch.save(model.state_dict(), model_name + '.pt')

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
              f'Test: {test_mae:.4f}, Best Val: {best_val_mae:.4f}, Test @ Best Val: {test_at_best_val:.4f}')
    
    print("Loading the best model for final evaluation...")
    # model.load_state_dict(torch.load(model_name + '.pt'))
    val_mae = test(model, val_loader, device)
    test_mae = test(model, test_loader, device)
    print(f'Final Test Loss: {test_mae:.4f}')
    metrics = {
                'model': model_name,
                'best_val_loss': val_mae,
                'best_test_loss': test_mae,
            }

    return metrics
##########################################################

##########################################################
#                   Neurons model                        #
##########################################################

class GNNnetwork_no_feature_encoding(torch.nn.Module):
    """
    GNN network for knowledge distillation.

    Parameters:
    - num_layers (int): Number of layers.
    - in_dim (int): Input dimension.
    - emb_dim (int): Embedding dimension.
    - add_residual (bool): Whether to add residual connections. Default is False.
    - track_running_stats (bool): Whether to track running stats in BatchNorm layers.
    - num_tasks (int): Number of tasks. Default is None.
    """

    def __init__(
        self,
        num_layers,
        in_dim,
        emb_dim,
        add_residual=False,
        track_running_stats=True,
        num_tasks: int = None,
    ):

        super().__init__()

        self.emb_dim = emb_dim

        self.gnn_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(
                CustomGINE(
                    in_dim,
                    emb_dim,
                    track_running_stats=track_running_stats,
                )
            )
            else:          
                self.gnn_layers.append(
                    CustomGINE(
                        emb_dim,
                        emb_dim,
                        track_running_stats=track_running_stats,
                    )
                )
            self.bn_layers.append(
                torch.nn.BatchNorm1d(
                    emb_dim, track_running_stats=track_running_stats)
            )

        self.add_residual = add_residual
        self.pool = global_add_pool

        self.final_layers = None
        if num_tasks is not None:
            emb_dim = emb_dim
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=emb_dim,
                                out_features=2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 * emb_dim,
                                out_features=num_tasks),
            )

    def forward(self, x, edge_index, edge_attr, batch):

        for gnn, bn in zip(self.gnn_layers, self.bn_layers):
            h = torch.relu(bn(gnn(x, edge_index, edge_attr)))

            if self.add_residual:
                if x.shape[1] == h.shape[1]:
                    x = h + x
                else:
                    x = h
            else:
                x = h

        x = self.pool(x, batch)
        out = self.final_layers(x)

        return out


##########################################################
#                Activations loader                      #
##########################################################

def load_pretrained_original_gnn_model(path='zinc-model_to_boost'):
    """
    Loads a pretrained GNN model.

    Parameters:
    - path (str): Path to the model file. Default is 'zinc-model_to_boost'.

    Returns:
    - GNNnetwork: The loaded model.
    """
    path = path + '.pt'
    model = GNNnetwork(
        num_layers=6,
        in_dim=21,
        emb_dim=128,
        add_residual=True,
        track_running_stats=True,
        num_tasks=1,
    )
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Function to extract activations, true labels, and pred labels
def get_all_activations_datasets(model, loader, device):
    """
    Extracts activations, true labels, and predicted labels from the model.

    Parameters:
    - model (GNNnetwork): The model to extract activations from.
    - loader (DataLoader): The data loader for extracting activations.
    - device (torch.device): The device to run the extraction on.

    Returns:
    - dict: Activations from all layers.
    - Tensor: True labels.
    - Tensor: Predicted labels.
    - list: Edge indices.
    - list: Edge attributes.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize a dictionary to store activations from all layers
    activations = {f'layer_{i}': [] for i in range(len(model.gnn_layers) + 1)}

    # Lists to store true labels and pred labels
    true_labels = []
    pred_labels = []
    edge_indices = []
    edge_attrs = []

    # Define a hook function to capture activations
    def hook_fn(module, input, output, layer_name):
        # Append the output activations to the corresponding layer entry
        activations[layer_name].append(output.detach().cpu().numpy())

    hooks = []
    # Register hooks for each GNN layer
    for i, layer in enumerate(model.gnn_layers):
        hooks.append(layer.register_forward_hook(lambda module, input,
                     output, i=i: hook_fn(module, input, output, f'layer_{i}')))

    # No gradient calculation needed
    with torch.no_grad():
        for data in tqdm(loader, desc="Collecting activations"):
            data = data.to(device)

            # Forward pass through the model
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)

            # Collect edge info
            edge_indices.append(data.edge_index.cpu().numpy())
            edge_attrs.append(data.edge_attr.cpu().numpy())

            # Collect true labels and predicted outputs
            true_labels.append(data.y.cpu().numpy())
            pred_labels.append(output.cpu().numpy())

    # Remove all hooks
    for hook in hooks:
        hook.remove()

    # Convert lists of numpy arrays into single torch tensors
    true_labels = torch.from_numpy(np.concatenate(true_labels, axis=0))
    pred_labels = torch.from_numpy(
        np.concatenate(pred_labels, axis=0)).squeeze()
    # edge_indices = torch.from_numpy(np.concatenate(edge_indices, axis=1))
    # edge_attrs = torch.from_numpy(np.concatenate(edge_attrs, axis=1))

    # Calculate the total error
    total_error = (pred_labels -
                   true_labels).abs().sum().item() / len(true_labels)
    print("Total error is: ", total_error)

    # Summary of outputs:
    # activations: a dictionary where each key corresponds to a layer and the value is a list of numpy arrays
    # - For example, activations['layer_0'] is a list of length <num_samples> and each entry activations['layer_0'][0] is a numpy array of shape (num_nodes, num_features_of_layer_0)
    # true_labels: a torch tensor of shape (num_samples, 1)
    # pred_labels: a torch tensor of shape (num_samples, 1)
    return activations, true_labels, pred_labels, edge_indices, edge_attrs


def create_geometric_dataset(activations, true_labels, pred_labels, edge_indices, edge_attrs, last_layer_idx=5):
    """
    Creates a geometric dataset from activations.

    Parameters:
    - activations (dict): Activations from the model.
    - true_labels (Tensor): True labels.
    - pred_labels (Tensor): Predicted labels.
    - edge_indices (list): Edge indices.
    - edge_attrs (list): Edge attributes.
    - last_layer_idx (int): Index of the last layer. Default is 5.

    Returns:
    - My_CustomDataset: The created geometric dataset.
    """
    data_list = []

    # Number of samples
    num_samples = true_labels.shape[0]

    for i in range(num_samples):
        # Concatenate all layer activations for the i-th sample
        sample_activations = [torch.tensor(
            activations[f'layer_{j}'][i]) for j in range(last_layer_idx)]
        # Concatenate along the feature dimension
        concatenated_activations = torch.cat(sample_activations, dim=1)

        # Create the data object for the i-th sample
        data = Data(
            x=concatenated_activations,
            true_label=true_labels[i],
            predicted_label=pred_labels[i],
            gap=true_labels[i] - pred_labels[i],
            edge_index=torch.from_numpy(edge_indices[i]),
            edge_attr=torch.from_numpy(edge_attrs[i])
        )
        data_list.append(data)
    return My_CustomDataset(data_list)


class My_CustomDataset(Dataset):
    """
    Custom dataset for geometric data.

    Parameters:
    - data_list (list): List of data objects.
    """
    def __init__(self, data_list):
        super().__init__() 
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def __getitem__(self, idx):
        return self.get(idx)
    
    def save(self, file_path):
        """
        Save the dataset to a file.

        Parameters:
        - file_path (str): Path to the file where the dataset will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.data_list, f)

    @classmethod
    def load(cls, file_path):
        """
        Load the dataset from a file.

        Parameters:
        - file_path (str): Path to the file from which the dataset will be loaded.

        Returns:
        - My_CustomDataset: An instance of My_CustomDataset with the loaded data.
        """
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)
        return cls(data_list)

def get_all_zinc12k_activations(use_split_indices=False,  split='train', model_name='zinc-model'):
    """
    Gets all activations for the specified split of the Zinc12k dataset.

    Parameters:
    - split (str): The dataset split ('train', 'val', 'test'). Default is 'train'.
    - model_name (str): The name of the model to load. Default is 'zinc-model'.

    Returns:
    - dict: Activations from all layers.
    - Tensor: True labels.
    - Tensor: Predicted labels.
    - list: Edge indices.
    - list: Edge attributes.
    """
    device = torch.device(f'cuda:{cfg.device}' if torch.cuda.is_available() else 'cpu')
    path = "/tmp"

    # Load dataset
    if split == 'val':
        loader = ZINC(path, subset=True, split=split)
        loader = DataLoader(loader, batch_size=1)
    elif split == 'test':
        loader = ZINC(path, subset=True, split=split)
        loader = DataLoader(loader, batch_size=1)
    elif split == 'train':
        loader = ZINC(path, subset=True, split=split)
        if use_split_indices == True:
            # Get indices for the split
            _, train_indices = load_train_split_indices()
            # Create subsets using the indices
            loader = Subset(loader, train_indices)
        loader = DataLoader(loader, batch_size=1)

    # Load model
    if use_split_indices == True:
        model_name = model_name + '_trained_on_5k_only'
        model = load_pretrained_original_gnn_model(model_name)
    else:
        model = load_pretrained_original_gnn_model(model_name)
    model = model.to(device)

    # Get all activations
    activations, true_labels, pred_labels, edge_indices, edge_attrs = get_all_activations_datasets(
        model, loader, device)
    return activations, true_labels, pred_labels, edge_indices, edge_attrs

##########################################################
def get_all_activations_geometric_datasets(use_split_indices=False):
    """
    Gets all activations datasets for train, val, and test splits.

    Returns:
    - My_CustomDataset: Train dataset with activations.
    - My_CustomDataset: Validation dataset with activations.
    - My_CustomDataset: Test dataset with activations.
    """
    print("Collecting train...")
    if use_split_indices == True:
        print("Loading the second 5000 samples")
    activations_train, true_labels_train, pred_labels_train, edge_indices_train, edge_attrs_train = get_all_zinc12k_activations(split='train', use_split_indices=use_split_indices)
    print("Done!")
    print("Collecting val...")
    activations_val, true_labels_val, pred_labels_val, edge_indices_val, edge_attrs_val = get_all_zinc12k_activations(
        split='val', use_split_indices=use_split_indices)
    print("Done!")
    print("Collecting test...")
    activations_test, true_labels_test, pred_labels_test, edge_indices_test, edge_attrs_test = get_all_zinc12k_activations(
        split='test', use_split_indices=use_split_indices)
    print("Done!")

    neurons_dataset_train = create_geometric_dataset(
        activations_train, true_labels_train, pred_labels_train, edge_indices_train, edge_attrs_train)
    neurons_dataset_val = create_geometric_dataset(
        activations_val, true_labels_val, pred_labels_val, edge_indices_val, edge_attrs_val)
    neurons_dataset_test = create_geometric_dataset(
        activations_test, true_labels_test, pred_labels_test, edge_indices_test, edge_attrs_test)
    return neurons_dataset_train, neurons_dataset_val, neurons_dataset_test
##########################################################


##########################################################
#                Knowledge distilation                   #
##########################################################


##########################################################
def distill_knowledge_to_neurons_model(cfg):
    """
    Distills knowledge from the original model to a new model using activations datasets.

    Returns:
    - dict: Metrics from the best model during distillation.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    print("Starting training of the original model...")
    metrics_zinc_model = train_from_scratch(cfg, use_split_indices=False)
    print("Original model training completed.")
    
    print("Collecting activations...")   
    def load_or_generate_datasets():
        train_file = './activations_for_knowledge_distillation/train_dataset.pkl'
        val_file = './activations_for_knowledge_distillation/val_dataset.pkl'
        test_file = './activations_for_knowledge_distillation/test_dataset.pkl'

        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            print("Loading existing activations datasets.")
            train_dataset = My_CustomDataset.load(train_file)
            val_dataset = My_CustomDataset.load(val_file)
            test_dataset = My_CustomDataset.load(test_file)
        else:
            print("Inferencing to get activations datasets.")
            train_dataset, val_dataset, test_dataset = get_all_activations_geometric_datasets()
            os.makedirs('./activations_for_knowledge_distillation', exist_ok=True)
            train_dataset.save(train_file)
            val_dataset.save(val_file)
            test_dataset.save(test_file)

        return train_dataset, val_dataset, test_dataset
    
    train_dataset, val_dataset, test_dataset = load_or_generate_datasets()  
    print("Activations collected.")
    print("Logging into Weights & Biases...")
    
    wandb.login(key='1b14383181f638ed8622970eb48b622a876f45dd')
    
    tag = f"Task_{cfg.task}||Epochs_{cfg.knowledge_distillation__epochs}||Num_layers_{cfg.knowledge_distillation__num_layers}||Learning_rate_{cfg.knowledge_distillation__learning_rate}||Num_layers_{cfg.knowledge_distillation__num_layers}||Emb_dim_{cfg.knowledge_distillation__emb_dim}||Add_residual_{cfg.knowledge_distillation__add_residual}||Step_size_{cfg.knowledge_distillation__step_size}||Gamma_{cfg.knowledge_distillation__gamma}"
    
    wandb.init(project=cfg.wandb__project, entity="guybs", name=tag)
    
    neurons_train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.knowledge_distillation__batch_size, shuffle=True)
    
    neurons_val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.knowledge_distillation__batch_size, shuffle=False)
    
    neurons_test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.knowledge_distillation__batch_size, shuffle=False)

    device = torch.device(f'cuda:{cfg.device}' if torch.cuda.is_available() else 'cpu')


    # Initialize the model
    model = GNNnetwork_no_feature_encoding(
        num_layers=cfg.knowledge_distillation__num_layers,
        in_dim=640,
        emb_dim=cfg.knowledge_distillation__emb_dim,
        add_residual=cfg.knowledge_distillation__add_residual,
        track_running_stats=cfg.knowledge_distillation__track_running_stats,
        num_tasks=cfg.knowledge_distillation__num_tasks,
    ).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    cfg.distill_knowledge_model_parmas = total_params
    wandb.config = cfg
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.knowledge_distillation__learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=cfg.knowledge_distillation__step_size,
                                                gamma=cfg.knowledge_distillation__gamma,
                                                )
    best_val_loss = best_test_loss = 1e9

    def train(model, optimizer, loader, device):
        """
        Trains the distillation model for one epoch.

        Parameters:
        - model (torch.nn.Module): The model to train.
        - optimizer (torch.optim.Optimizer): The optimizer for training.
        - loader (DataLoader): The data loader for training data.
        - device (torch.device): The device to run the training on.

        Returns:
        - float: The average loss for the epoch.
        """
        model.train()
        total_loss = 0
        for data in tqdm(loader, desc="Training"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = (out.squeeze() - data.true_label).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def test(model, loader, device):
        """
        Evaluates the distillation model.

        Parameters:
        - model (torch.nn.Module): The model to evaluate.
        - loader (DataLoader): The data loader for evaluation data.
        - device (torch.device): The device to run the evaluation on.

        Returns:
        - float: The average error for the evaluation.
        """
        model.eval()
        total_error = 0
        for data in tqdm(loader, desc="Testing"):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_error += (out.squeeze() - data.true_label).abs().sum().item()
        return total_error / len(loader.dataset)
    
    metrics = {}
    print("Starting training of the distilled model...")
    for epoch in tqdm(range(cfg.knowledge_distillation__epochs), desc="Distilling knowledge"):
        
        train_loss = train(model, optimizer, neurons_train_dataloader, device)
        
        val_loss = test(
            model, neurons_val_dataloader, device)
        
        test_loss = test(
            model, neurons_test_dataloader, device)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            # torch.save(model.state_dict(), cfg.knowledge_distillation__model_name + '.pt')
            metrics = {
                'epoch': epoch + 1,
                'train_MAE': train_loss,
                'validation_MAE': val_loss,
                'test_MAE': test_loss,
                'best_validation_MAE': best_val_loss,
                'best_test_MAE_based_on_val': best_test_loss,
                'original_model_best_validation_MAE': metrics_zinc_model['best_val_loss'],
                'original_model_best_test_MAE_based_on_val': metrics_zinc_model['best_test_loss'],
            }

        # Log metrics to wandb
        wandb.log({
                'epoch': epoch + 1,
                'train_MAE': train_loss,
                'validation_MAE': val_loss,
                'test_MAE': test_loss,
                'best_validation_MAE': best_val_loss,
                'best_test_MAE_based_on_val': best_test_loss,
                'original_model_best_validation_MAE': metrics_zinc_model['best_val_loss'],
                'original_model_best_test_MAE_based_on_val': metrics_zinc_model['best_test_loss'],
            })

        print(
            f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, , Test Loss: {test_loss:.4f}, Best Test Loss: {best_test_loss:.4f}')
    
    print("Loading the best distilled model for final evaluation...")
    # model.load_state_dict(torch.load(
    #     cfg.knowledge_distillation__model_name + '.pt'))
    val_loss = test(model, neurons_val_dataloader, device)
    test_loss = test(model, neurons_test_dataloader, device)
    print(f'Final Test Loss: {test_loss:.4f}')

    print("Saving all metrics to CSV...")
    # Save all metrics to CSV
    save_metrics_to_csv(metrics, filename=cfg.knowledge_distillation__metrics_file + '.csv')
    print("Metrics saved to" + cfg.knowledge_distillation__metrics_file + ".csv.")
    
    return metrics
##########################################################

##########################################################
#                Estimate uncertainty                    #
##########################################################


def calculate_metrics(true_gap, pred_gap, true_labels, pred_labels):
    true_gap = true_gap.cpu().numpy()
    pred_gap = pred_gap.cpu().numpy()
    # metrics for neurons' model predicting the gap
    mae = mean_absolute_error(true_gap, pred_gap)
    mse = mean_squared_error(true_gap, pred_gap)
    rmse = mean_squared_error(true_gap, pred_gap, squared=False)
    r2 = r2_score(true_gap, pred_gap)
    # metrics for neurons' model goodness in uncertainty
    abs_pred_gap = np.abs(pred_gap)
    abs_true_gap = np.abs(true_gap)

    def rc_curve_calculator(abs_true_gap, abs_pred_gap):
        abs_pred_gap_sorted_indices = np.argsort(abs_pred_gap)
        abs_true_gap = abs_true_gap[abs_pred_gap_sorted_indices]
        coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        risks = []
        for c in coverages:
            risk = np.mean(abs_true_gap[:int(c * len(abs_true_gap))])
            risks.append(risk)
        rcauc = auc(x=coverages, y=risks)
        return coverages, risks, rcauc
    coverages, risks, rcauc = rc_curve_calculator(
        abs_true_gap=abs_true_gap, abs_pred_gap=abs_pred_gap)
    # metrics for neurons' model in improving performance
    original_model_mae = torch.abs(true_labels - pred_labels).mean().item()
    booster_model_mae = torch.abs(
        true_labels - (pred_labels + torch.from_numpy(pred_gap).to(true_labels.device))).mean().item()
    
    plt.figure()
    plt.plot(coverages, risks)
    
    return {
        'Neurons_MAE': mae,
        'Neurons_MSE': mse,
        'Neurons_RMSE': rmse,
        'Neurons_R²': r2,
        'Neurons_rcauc': rcauc,
        'Neurons_coverages': coverages,
        'Neurons_risks': risks,
        'Original_model_MAE': original_model_mae,
        'Booster_model_MAE': booster_model_mae,
        'rc_curve': wandb.Image(plt)
        #

    }


def get_train_split_indices(train_dataset_size=10000, size_of_actual_train=5000, seed=42):
    """
    Splits the given training dataset into indices for Train_dataset and Train_Boost_dataset,
    and saves these indices to a pickle file.
    
    Parameters:
    - train_dataset (Dataset): The original training dataset.
    - size_of_actual_train (int): The size of the Train_dataset.
    - seed (int): The random seed for reproducibility. Default is 42.
    
    Returns:
    - train_indices (numpy.ndarray): Indices for the Train_dataset of size X.
    - boost_indices (numpy.ndarray): Indices for the Train_Boost_dataset of size 10,000 - X.
    """
    np.random.seed(seed)

    # Get indices for the split
    indices = np.random.permutation(train_dataset_size)

    # Split indices into Train_dataset and Train_Boost_dataset
    train_indices = indices[:size_of_actual_train]
    boost_indices = indices[size_of_actual_train:10_000]

    # Save indices to a pickle file
    with open(f'split_indices_{size_of_actual_train}.pkl', 'wb') as f:
        pickle.dump((train_indices, boost_indices), f)

    return train_indices, boost_indices


def load_train_split_indices(size_of_actual_train=5000):
    """
    Loads the split indices from a pickle file.
    
    Parameters:
    - filename (str): The name of the pickle file to load the indices from. Default is 'split_indices.pkl'.
    
    Returns:
    - train_indices (numpy.ndarray): Indices for the Train_dataset.
    - boost_indices (numpy.ndarray): Indices for the Train_Boost_dataset.
    """
    with open(f'split_indices_{size_of_actual_train}.pkl', 'rb') as f:
        train_indices, boost_indices = pickle.load(f)

    return train_indices, boost_indices


##########################################################
def estimate_uncertainty(cfg):
    """
    Estimate uncertainty

    Returns:
    - dict: Metrics of uncertainty estimation
    """
    print("Starting training of the original model...")
    print("Training on 5000 samples (only) instead of 10000 (leaves the other 5000 for training the neurons model for uncertainty)")
    metrics_zinc_model = train_from_scratch(cfg, use_split_indices=True)
    print("Original model training completed.")

    print("Collecting activations...")
    train_dataset, val_dataset, test_dataset = get_all_activations_geometric_datasets(use_split_indices=True)
    print("Activations collected.")

    print("Logging into Weights & Biases...")
    wandb.login(key='1b14383181f638ed8622970eb48b622a876f45dd')
    wandb.init(project=cfg.wandb__project, entity="guybs")

    neurons_train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.uncertainty_estimation__batch_size, shuffle=True)

    neurons_val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.uncertainty_estimation__batch_size, shuffle=False)

    neurons_test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.uncertainty_estimation__batch_size, shuffle=False)

    device = torch.device(f'cuda:{cfg.device}' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = GNNnetwork_no_feature_encoding(
        num_layers=cfg.uncertainty_estimation__num_layers,
        in_dim=640,
        emb_dim=cfg.uncertainty_estimation__emb_dim,
        add_residual=cfg.uncertainty_estimation__add_residual,
        track_running_stats=cfg.uncertainty_estimation__track_running_stats,
        num_tasks=cfg.uncertainty_estimation__num_tasks,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.uncertainty_estimation__learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=cfg.uncertainty_estimation__step_size,
                                                gamma=cfg.uncertainty_estimation__gamma,
                                                )
    best_val_loss = best_test_loss = 1e9

    def train(model, optimizer, loader, device):
        """
        Trains the distillation model for one epoch.

        Parameters:
        - model (torch.nn.Module): The model to train.
        - optimizer (torch.optim.Optimizer): The optimizer for training.
        - loader (DataLoader): The data loader for training data.
        - device (torch.device): The device to run the training on.

        Returns:
        - float: The average loss for the epoch.
        """
        model.train()
        total_loss = 0
        for data in tqdm(loader, desc="Training"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = (out.squeeze() - data.gap).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(loader.dataset)

    
    @torch.no_grad()
    def test(model, loader, device):
        """
        Evaluates the distillation model.

        Parameters:
        - model (torch.nn.Module): The model to evaluate.
        - loader (DataLoader): The data loader for evaluation data.
        - device (torch.device): The device to run the evaluation on.

        Returns:
        - float: The average error for the evaluation.
        """
        model.eval()
        total_error = 0
        
        true_gaps = []
        pred_gaps = []
        true_labels = []
        pred_labels = []
    
        for data in tqdm(loader, desc="Testing"):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_error += (out.squeeze() - data.gap).abs().sum().item()
            
            true_gaps.append(data.gap)
            pred_gaps.append(out.squeeze())
            true_labels.append(data.true_label)
            pred_labels.append(data.predicted_label)
            
        true_gaps = torch.concatenate(true_gaps)
        pred_gaps = torch.concatenate(pred_gaps)
        true_labels = torch.concatenate(true_labels)
        pred_labels = torch.concatenate(pred_labels)
            
        return total_error / len(loader.dataset), true_gaps, pred_gaps, true_labels, pred_labels

    metrics = {}
    print("Starting training of uncertainty estimator model...")
    for epoch in tqdm(range(cfg.uncertainty_estimation__epochs), desc="Training uncertainty model"):

        train_loss = train(model, optimizer, neurons_train_dataloader, device)

        val_loss, val_true_gaps, val_pred_gaps, val_true_labels, val_pred_labels = test(
            model, neurons_val_dataloader, device)
        
        metrics_val = calculate_metrics(
            true_gap=val_true_gaps, pred_gap=val_pred_gaps, true_labels=val_true_labels, pred_labels=val_pred_labels)
        
        test_loss, test_true_gaps, test_pred_gaps, test_true_labels, test_pred_labels = test(
            model, neurons_test_dataloader, device)
        
        metrics_test = calculate_metrics(
            true_gap=test_true_gaps, pred_gap=test_pred_gaps, true_labels=test_true_labels, pred_labels=test_pred_labels)
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            # torch.save(model.state_dict(), cfg.uncertainty_estimation__model_name + '.pt')
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'best_val_loss': best_val_loss,
                'best_test_loss': best_test_loss,
                **{f'best_val_{k}': v for k, v in metrics_val.items()},
                **{f'best_test_{k}': v for k, v in metrics_test.items()}
            }

        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            # 'best_val_loss': best_val_loss,
            # 'best_test_loss': best_test_loss,
            **{f'val_{k}': v for k, v in metrics_val.items()},
            **{f'test_{k}': v for k, v in metrics_test.items()},
            "val_rc_curve": metrics_val['rc_curve'],
            "test_rc_curve": metrics_test['rc_curve'],
        })

        print(
            f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, , Test Loss: {test_loss:.4f}, Best Test Loss: {best_test_loss:.4f}')

    print("Loading the best uncertainty model for final evaluation...")
    # model.load_state_dict(torch.load(cfg.uncertainty_estimation__model_name + '.pt'))
    val_loss, val_true_gaps, val_pred_gaps, val_true_labels, val_pred_labels = test(
        model, neurons_val_dataloader, device)

    metrics_val = calculate_metrics(
        true_gap=val_true_gaps, pred_gap=val_pred_gaps, true_labels=val_true_labels, pred_labels=val_pred_labels)

    test_loss, test_true_gaps, test_pred_gaps, test_true_labels, test_pred_labels = test(
        model, neurons_test_dataloader, device)

    metrics_test = calculate_metrics(
        true_gap=test_true_gaps, pred_gap=test_pred_gaps, true_labels=test_true_labels, pred_labels=test_pred_labels)
    print(f'Final Test Loss: {test_loss:.4f}')

    print("Saving all metrics to CSV...")
    # Save all metrics to CSV
    save_metrics_to_csv(metrics, filename=cfg.uncertainty_estimation__metrics_file + '.csv')
    print("Metrics saved to" + cfg.uncertainty_estimation__metrics_file + ".csv.")

    return metrics
##########################################################

if __name__ == "__main__":
    # Load configuration from the YAML file
    cfg = load_config('config.yaml')

    # Parse command-line arguments
    args = parse_args(cfg)
    cfg = edict(vars(args))

    get_train_split_indices()
    if cfg.task == "uncertainty_estimation":
        estimate_uncertainty(cfg)
    elif cfg.task == "distill_knowledge":
        distill_knowledge_to_neurons_model(cfg)
    
    
    exit()

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################





####################
# Evaluation merics
####################

def save_metrics_to_csv(metrics, filename='metrics.csv'):
    # Ensure all values are converted to strings for writing to CSV
    metrics = {k: (','.join(map(str, v)) if isinstance(v, list) else v)
               for k, v in metrics.items()}
    fieldnames = metrics.keys()

    # Write to CSV file (overwrite mode)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)

####################
# Section 0: Train a GNN model on the train dataset, which is taken from original train dataset
####################



##############################################################################



##############################################################################

####################
# Section 1: Load Activations from Pretrained Model
####################



####################
# Section 2: Create a dataset from these activations
####################


####################
# Section 3: Split the dataset randomly
####################

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert train_ratio + val_ratio + \
        test_ratio == 1, "The sum of the split ratios must be 1."
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_split = int(np.floor(train_ratio * dataset_size))
    val_split = int(np.floor(val_ratio * dataset_size))

    train_indices = indices[:train_split]
    val_indices = indices[train_split:train_split + val_split]
    test_indices = indices[train_split + val_split:]

    train_data_list = [dataset[i] for i in train_indices]
    val_data_list = [dataset[i] for i in val_indices]
    test_data_list = [dataset[i] for i in test_indices]

    train_dataset = My_CustomDataset(train_data_list)
    val_dataset = My_CustomDataset(val_data_list)
    test_dataset = My_CustomDataset(test_data_list)

    return train_dataset, val_dataset, test_dataset

####################
# Section 4: Train a new model on top of these activations
####################

######################################################################################################

def train_neurons_model(train_dataset, val_dataset, test_dataset):
    wandb.login(key='1b14383181f638ed8622970eb48b622a876f45dd')
    # wandb.login()  # Log in to wandb
    # os.environ['WANDB_API_KEY'] = '1b14383181f638ed8622970eb48b622a876f45dd'
    wandb.init(project="neurons_project10", entity="guybs")
    neurons_train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True)
    neurons_val_dataloader = DataLoader(
        val_dataset, batch_size=128, shuffle=False)
    neurons_test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False)

    device = torch.device(f'cuda:{cfg.device}' if torch.cuda.is_available() else 'cpu')


    # Initialize the model
    model = GNNnetwork_no_feature_encoding(
        num_layers=6,
        in_dim=640,
        emb_dim=640,
        add_residual=True,
        track_running_stats=True,
        num_tasks=1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)


    def train(model, optimizer, loader, device):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = (out.squeeze() - data.gap).abs().mean()
            # gap=true_labels[i] - pred_labels[i],
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(model, loader, device):
        model.eval()
        total_loss = 0
        true_gaps = []
        pred_gaps = []
        
        true_labels = []
        pred_labels = []
        
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = (out.squeeze() - data.gap).abs().sum()
            total_loss += loss.item()
            true_gaps.append(data.gap)
            pred_gaps.append(out.squeeze())
            true_labels.append(data.true_label)
            pred_labels.append(data.predicted_label)
        true_gaps = torch.concatenate(true_gaps)
        pred_gaps = torch.concatenate(pred_gaps)
        true_labels = torch.concatenate(true_labels)
        pred_labels = torch.concatenate(pred_labels)
        return total_loss / len(loader.dataset), true_gaps, pred_gaps, true_labels, pred_labels

    best_val_loss = float('inf')
    best_test_loss = float('inf')
    patience, trials = 100, 0


    for epoch in range(3):
        train_loss = train(model, optimizer, neurons_train_dataloader, device)
        val_loss, val_true_gaps, val_pred_gaps, val_true_labels, val_pred_labels = evaluate(
            model, neurons_val_dataloader, device)
        metrics_val = calculate_metrics(
            true_gap=val_true_gaps, pred_gap=val_pred_gaps, true_labels=val_true_labels, pred_labels=val_pred_labels)
        test_loss, test_true_gaps, test_pred_gaps, test_true_labels, test_pred_labels = evaluate(
            model, neurons_test_dataloader, device)
        metrics_test = calculate_metrics(
            true_gap=test_true_gaps, pred_gap=test_pred_gaps, true_labels=test_true_labels, pred_labels=test_pred_labels)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_neurons_model.pt')
            metrics = {
                'model': 'booster_model',
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'best_val_loss': best_val_loss,
                'best_test_loss': best_test_loss,
                **{f'val_{k}': v for k, v in metrics_val.items()},
                **{f'test_{k}': v for k, v in metrics_test.items()}
            }
            trials = 0
        else:
            trials += 1
            
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'best_val_loss': best_val_loss,
            'best_test_loss': best_test_loss,
            **{f'val_{k}': v for k, v in metrics_val.items()},
            **{f'test_{k}': v for k, v in metrics_test.items()}
        })


        print(
            f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, , Test Loss: {test_loss:.4f}, Best Test Loss: {best_test_loss:.4f}')

        if trials >= patience:
            print('Early stopping')
            break

    # model.load_state_dict(torch.load('best_neurons_model.pt'))
    val_loss, _, _, _, _= evaluate(model, neurons_val_dataloader, device)
    test_loss, _, _, _, _ = evaluate(model, neurons_test_dataloader, device)
    print(f'Test Loss: {test_loss:.4f}')

    return metrics


if __name__ == "__main__":
    # knowledge_distilation()
    exit()
    # get_train_split_indices(train_dataset_size=10_000, size_of_actual_train=5000)
    # exit()
    
    ####################
    # Section 0: Train a GNN model on the dataset.
    ####################
    print("Section 0: Train a GNN model on the dataset.")
    metrics_zinc_model = train_from_scratch(use_split_indices=False)
    metrics_zinc_model_to_boost = train_from_scratch(use_split_indices=True)

    ####################
    # Section 1: Load Activations from Pretrained Model
    ####################
    print("Section 1: Load Activations from Pretrained Model")
    activations, true_labels, pred_labels, edge_indices, edge_attrs = get_all_zinc12k_activations()

    ####################
    # Section 2: Create a dataset from these activations
    ####################
    print("Section 2: Create a dataset from these activations")
    neurons_dataset = create_geometric_dataset(
        activations, true_labels, pred_labels, edge_indices, edge_attrs)
    
    ####################
    # Section 3: Split the dataset randomly
    ####################
    print("Section 3: Split the dataset randomly")
    train_dataset, val_dataset, test_dataset = split_dataset(
        neurons_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(
        f"Training set: {len(train_dataset)} samples, Validation set: {len(val_dataset)} samples, Test set: {len(test_dataset)} samples")


   ####################
    # Section 4: Train a new model on top of these activations
    ####################
    print("Section 4: Train a new model on top of these activations")
    metrics_booster = train_neurons_model(
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
    
    results_list = [
        metrics_zinc_model,
        metrics_zinc_model_to_boost,
        metrics_booster
    ]
    # Combine results into a single dataframe
    results_df = pd.DataFrame(results_list)

    # Save the combined results to a CSV file
    results_df.to_csv('model_results.csv', index=False)
    print("Results saved to model_results.csv")
