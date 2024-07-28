"""
Description: This script performs the following tasks on the Zinc12k dataset:

0) Train a GNN model on the dataset.
1) Load activations from a pretrained model.
2) Create a dataset from these activations:
   a) Each data point is a tuple (Activations_i, gap_from_groundtruth_i, ground_truth_i).
3) Train a new model on top of these activations to:
   a) Predict whether the prediction is correct or not.
   b) Predict the gap from the ground truth.
4) Evaluate the new model using:
   a) AUROC, AUPR0, AUPR1, AURC
   b) MAE (Mean Absolute Error)
"""

import torch
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from torch_geometric.nn import  GINEConv, global_add_pool
import torch
import numpy as np
import random
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
####################
# Section 0: Train a GNN model on the dataset.
####################

# The original model over Zinc12k
class CustomGINE(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, track_running_stats, num_edge_emb=4):
        super().__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim, track_running_stats=track_running_stats),
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

        self.feature_encoder =  torch.nn.Embedding(
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
                torch.nn.BatchNorm1d(emb_dim, track_running_stats=track_running_stats)
            )

        self.add_residual = add_residual
        self.pool = global_add_pool

        self.final_layers = None
        if num_tasks is not None:
            emb_dim = emb_dim
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks),
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
    model.train()
    total_loss = 0
    for data in loader:
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
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)

##############################################################################
def train_from_scratch():
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)

    path = "/tmp"

    train_dataset = ZINC(path, subset=True, split='train')
    val_dataset = ZINC(path, subset=True, split='val')
    test_dataset = ZINC(path, subset=True, split='test')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)


    device = torch.device('cuda:2')

    model = GNNnetwork(
        num_layers=6,
        in_dim=21,
        emb_dim=128,
        add_residual=True,
        track_running_stats=True,
        num_tasks=1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=300,
                                                gamma=0.5,
                                                )
    best_val_mae = test_at_best_val = 1e9
    for epoch in range(0, 1_000):
        loss = train(model, optimizer, train_loader, device)
        val_mae = test(model, val_loader, device)
        test_mae = test(model, test_loader, device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            test_at_best_val = test_mae

            torch.save(model.state_dict(), 'zinc-model.pt')


        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
            f'Test: {test_mae:.4f}, Best Val: {best_val_mae:.4f}, Test @ Best Val: {test_at_best_val:.4f}')
##############################################################################

####################
# Section 1: Load Activations from Pretrained Model
####################

# Load your pretrained GNN model
def load_pretrained_gnn_model(path='zinc-model.pt'):
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

# Function to extract activations, true labels, and predicted labels
def get_all_activations(model, loader, device):
    model.eval()
    activations = {f'layer_{i}': [] for i in range(len(model.gnn_layers) + 1)}  # To store activations from all layers
    node_mapping = []

    def hook_fn(module, input, output, layer_name):
        activations[layer_name].append(output.detach().cpu())

    hooks = []
    # Register hooks for each layer
    for i, layer in enumerate(model.gnn_layers):
        hooks.append(layer.register_forward_hook(lambda module, input, output, i=i: hook_fn(module, input, output, f'layer_{i}')))
    # Also register a hook for the final layer
    hooks.append(model.final_layers.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, f'layer_{len(model.gnn_layers)}')))

    with torch.no_grad():
        for graph_idx, data in enumerate(loader):
            data = data.to(device)
            model(data.x, data.edge_index, data.edge_attr, data.batch)
            node_mapping.append(graph_idx)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Convert lists of tensors into single tensors
    for key in activations.keys():
        activations[key] = torch.cat(activations[key], dim=0)
    node_mapping = np.array(node_mapping)

    return activations, node_mapping

# Function to concatenate activations for each node
def concatenate_activations(activations):
    concat_activations = []
    for i in range(len(activations['layer_0'])):
        node_activations = [activations[f'layer_{j}'][i] for j in range(len(activations))]
        concat_activations.append(torch.cat(node_activations, dim=0))
    return torch.stack(concat_activations)

##############################################################################
def get_all_zinc12k_test_activatiions():
    device = torch.device('cuda:2')
    path = "/tmp"

    # Load dataset
    test_dataset = ZINC(path, subset=True, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Load model
    model = load_pretrained_gnn_model()
    model = model.to(device)

    # Get all activations
    activations, node_mapping = get_all_activations(model, test_loader, device)
    
    print("guy")
    # return activations, true_labels, predicted_labels
##############################################################################


####################
# Section 2: Create a dataset from these activations
####################


if __name__ == "__main__":
    ####################
    # Section 0: Train a GNN model on the dataset.
    ####################
    print("Section 0: Train a GNN model on the dataset.")
    # train_from_scratch()
            
    ####################
    # Section 1: Load Activations from Pretrained Model
    ####################
    print("Section 1: Load Activations from Pretrained Model")
    activations, true_labels, predicted_labels = get_all_zinc12k_test_activatiions()
    print("guy")

    ####################
    # Section 2: Create a dataset from these activations
    ####################
    print("Section 2: Create a dataset from these activations")
    