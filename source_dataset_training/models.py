import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
import torch_geometric.nn as tg_nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from easydict import EasyDict as edict
from source_dataset_training import data
from source_dataset_training.resnet import *


class CustomGINE(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, layernorm, track_running_stats, num_edge_emb=4):
        super().__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(
                emb_dim, track_running_stats=track_running_stats)
            if not layernorm
            else torch.nn.LayerNorm(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.layer = GINEConv(nn=mlp, train_eps=True)
        self.edge_embedding = torch.nn.Embedding(
            num_embeddings=num_edge_emb, embedding_dim=in_dim
        )

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, self.edge_embedding(edge_attr))

    def reset_parameters(self):
        self.edge_embedding.reset_parameters()
        self.layer.reset_parameters()


class GNNnetwork(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        emb_dim,
        feature_encoder,
        GNNConv,
        layernorm=False,
        add_residual=False,
        track_running_stats=True,
        num_tasks: int = None,
    ):

        super().__init__()

        self.emb_dim = emb_dim

        # self.feature_encoder = feature_encoder
        self.feature_encoder = nn.Embedding(
            num_embeddings=21, embedding_dim=emb_dim)

        self.gnn_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                GNNConv(
                    emb_dim if i != 0 else in_dim,
                    emb_dim,
                    layernorm,
                    track_running_stats=track_running_stats,
                )
            )
            self.bn_layers.append(
                torch.nn.BatchNorm1d(
                    emb_dim, track_running_stats=track_running_stats)
                if not layernorm
                else torch.nn.LayerNorm(emb_dim)
            )
        self.add_residual = add_residual
        self.final_layers = None
        if num_tasks is not None:
            emb_dim = emb_dim
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 * emb_dim,
                                out_features=num_tasks),
            )

    def forward(self, batched_data):
        x, edge_index, edge_attr = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
        )
        x = self.feature_encoder(x.squeeze())  # (g, node, _)

        for gnn, bn in zip(self.gnn_layers, self.bn_layers):
            h = torch.relu(bn(gnn(x, edge_index, edge_attr)))

            if self.add_residual:
                x = h + x
            else:
                x = h

        
        x_pool = tg_nn.global_add_pool(x, batched_data.batch)
        # x_pool = tg_nn.global_mean_pool(x, batched_data.batch)
        # x = torch_scatter.segment_csr(
        #     src=x,
        #     indptr=batched_data.batch,
        #     reduce="mean",
        # )  # : (g, node, _) -> (g, _)
        out = self.final_layers(x_pool)

        return out




def get_model(cfg):
    if cfg.source_dataset.name == "zinc12k":
        model = GNNnetwork(num_layers=6,
                           in_dim=128,
                           emb_dim=128,
                           feature_encoder=AtomEncoder(
                               128),
                        GNNConv=CustomGINE,
                        layernorm=False,
                           add_residual=True,
                        track_running_stats=True,
                        num_tasks=1)


        cfg.source_dataset.model = edict({})
        cfg.source_dataset.model.dim_embed = 128
        cfg.source_dataset.model.num_layers = 6
        cfg.source_dataset.max_neuron_number = data.get_n_max_of_dataset(cfg)
        
    elif cfg.source_dataset.name == "cifar10":
        model = resnet18(pretrained=True)
        
        cfg.source_dataset.model = edict({})
        cfg.source_dataset.model.dim_embed = 512
        cfg.source_dataset.model.num_layers = 10
        cfg.source_dataset.max_neuron_number = 1
        
        
    else:
        raise NotImplementedError(
            f"Model for source cataset {cfg.source_dataset.name} not implemented"
        )
    return model
