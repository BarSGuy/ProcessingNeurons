import torch
import torch.nn as nn
import torch
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

##############################################################
###################### Main architecure ######################
##############################################################


class NeuronArchitecture_images(nn.Module):
    def __init__(self, cfg) -> None:
        super(NeuronArchitecture_images, self).__init__()
        self.cfg = cfg
        d_0 = 512
        d_hidden = self.cfg.model.dim_embed
        d_final = self.cfg.model.dim_output
        num_layers = self.cfg.model.num_layers

        # self.cfg=cfg
        # d = self.cfg.source_dataset.model.dim_embed
        # L = self.cfg.source_dataset.model.num_layers
        # d_hidden = self.cfg.model.dim_embed
        # d_final = self.cfg.model.dim_output

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Initial layer
        self.layers.append(NeuronEquivDeepSetLayer_translation(
            d_in=d_0, d_out=d_hidden))
        self.bns.append(nn.BatchNorm1d(d_hidden))

        # Hidden layers
        for layer in range(1, num_layers):
            self.layers.append(NeuronEquivDeepSetLayer_translation(
                d_in=d_hidden, d_out=d_hidden))
            self.bns.append(nn.BatchNorm1d(d_hidden))

        # Final layer - pooling
        self.pooling = NeuronInvariantDeepSetLayer_translation(
            d_in=d_hidden, d_out=d_final)
        
        # Output layer for binary classification
        self.output_layer = nn.Linear(d_final, 1)

    def forward(self, batch):
        for layer, bn in zip(self.layers, self.bns):
            batch_x = bn(layer(batch))
            if self.cfg.model.residual and batch.x.shape[-1] == batch_x.shape[-1]:
                batch.x = batch.x + batch_x
            else:
                batch.x = batch_x
        # pooling
        batch.x = self.pooling(batch)
        # Apply the output layer
        batch.x = self.output_layer(batch.x)
        # Apply sigmoid to get probabilities
        batch.x = torch.sigmoid(batch.x)
        return batch.x

    # def forward(self, batch):
    #     for layer in self.layers:
    #         batch.x = layer(batch)
    #     return batch.x


class NeuronArchitecture(nn.Module):
    def __init__(self, cfg) -> None:
        super(NeuronArchitecture, self).__init__()
        self.cfg = cfg
        d_0 = cfg.source_dataset.model.dim_embed * cfg.source_dataset.model.num_layers
        d_hidden = self.cfg.model.dim_embed
        d_final = self.cfg.model.dim_output
        num_layers = self.cfg.model.num_layers
        
        # self.cfg=cfg
        # d = self.cfg.source_dataset.model.dim_embed
        # L = self.cfg.source_dataset.model.num_layers
        # d_hidden = self.cfg.model.dim_embed
        # d_final = self.cfg.model.dim_output
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Initial layer
        self.layers.append(NeuronEquivDeepSetLayer(
            d_in=d_0, d_out=d_hidden))
        self.bns.append(nn.BatchNorm1d(d_hidden))

        # Hidden layers
        for layer in range(1, num_layers):
            self.layers.append(NeuronEquivDeepSetLayer(
                d_in=d_hidden, d_out=d_hidden))
            self.bns.append(nn.BatchNorm1d(d_hidden))

        # Final layer - pooling
        self.pooling = NeuronInvariantDeepSetLayer(
            d_in=d_hidden, d_out=d_final)

    def forward(self, batch):
        for layer, bn in zip(self.layers, self.bns):
            batch_x = bn(layer(batch))
            if self.cfg.model.residual and batch.x.shape[-1] == batch_x.shape[-1]:
                batch.x = batch.x + batch_x
            else:
                batch.x = batch_x
        # pooling
        batch.x = self.pooling(batch)
        return batch.x
    
    # def forward(self, batch):
    #     for layer in self.layers:
    #         batch.x = layer(batch)
    #     return batch.x

class MLPs(nn.Module):
    def __init__(self, cfg) -> None:
        super(MLPs, self).__init__()
        self.cfg = cfg
        d_0 = self.cfg.source_dataset.max_nodes * \
            cfg.source_dataset.model.num_layers * cfg.source_dataset.model.dim_embed
        d_hidden = self.cfg.model.dim_embed
        d_final = self.cfg.model.dim_output
        
        # d = self.cfg.source_dataset.model.dim_embed
        # num_layers = self.cfg.model.num_layers
        # L = self.cfg.source_dataset.model.num_layers
        # d_hidden = self.cfg.model.dim_embed
        # d_final = self.cfg.model.dim_output
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Create the layers
        
        # layer 1:
        self.layers.append(nn.Linear(d_0, d_hidden))
        self.bns.append(nn.BatchNorm1d(d_hidden))
        # layers 2 - L:
        for _ in range(1, cfg.model.num_layers):
            self.layers.append(nn.Linear(d_hidden, d_hidden))
            self.bns.append(nn.BatchNorm1d(d_hidden))
        # output layer:
        self.layers.append(nn.Linear(d_hidden, d_final))
        
    def forward(self, batch):
        x = batch.x
        for layer, bn in zip(self.layers, self.bns):
            x_ = F.relu(bn(layer(x)))
            if self.cfg.model.residual and x.shape[-1] == x_.shape[-1]:
                x = x + x_
            else:
                x = x_
        # pooling
        x = self.layers[-1](x)
        x = global_mean_pool(x, batch.batch)
        return x
##############################################################
######################### Helpers ############################
##############################################################


class NeuronEquivDeepSetLayer_translation(nn.Module):
    def __init__(self, d_in, d_out):
        super(NeuronEquivDeepSetLayer_translation, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out)
        )
        self.rho = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out)
        )

    def forward(self, batch):
        # # Apply phi to each element in the set
        x_phi = self.phi(batch.x)

        # # Sum over the elements in the set
        act_indices = batch.activation_idx + batch.batch * batch.num_layers[0]
        x_sum_brodcasted = self.get_x_sum_brod(x=batch.x, idx=act_indices)
        x_sum = self.rho(x_sum_brodcasted)

        # # sum them
        x = x_phi + x_sum
        return x

    def get_x_sum_brod(self, x, idx):
        x_sum = scatter(src=x, index=idx.long(),
                        dim=0, reduce='sum')
        x_sum_brodcasted = x_sum[idx.long()]
        return x_sum_brodcasted


class NeuronInvariantDeepSetLayer_translation(nn.Module):
    def __init__(self, d_in, d_out):
        super(NeuronInvariantDeepSetLayer_translation, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_in)
        )
        self.rho = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_out)
        )

    def forward(self, batch):
        # Apply phi to each element in the set
        x_phi = self.phi(batch.x)

        # Sum over the elements in the set
        act_indices = batch.activation_idx + batch.batch * batch.num_layers[0]
        x_sum = self.get_x_sum(x=x_phi, idx=act_indices)

        # Apply rho to the summed result
        x_sum = self.rho(x_sum)
        batch_size = torch.tensor(batch.batch.max().item() + 1)
        batch_size = batch_size.to(x_sum.device)
        indices = torch.repeat_interleave(torch.arange(
            batch_size).to(x_sum.device), batch.num_layers[0])
        x_sum = self.get_x_sum(x=x_sum, idx=indices)
        return x_sum

    def get_x_sum(self, x, idx):
        x_sum = scatter(src=x, index=idx.long(),
                        dim=0, reduce='sum')
        return x_sum




class NeuronEquivDeepSetLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super(NeuronEquivDeepSetLayer, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out)
        )
        self.rho = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out)
        )

    def forward(self, batch):
        # # Apply phi to each element in the set
        x_phi = self.phi(batch.x)

        # # Sum over the elements in the set
        x_sum_brodcasted = self.get_x_sum_brod(x=batch.x, idx=batch.batch)
        x_sum = self.rho(x_sum_brodcasted)

        # # sum them
        x = x_phi + x_sum
        return x

    def get_x_sum_brod(self, x, idx):
        x_sum = scatter(src=x, index=idx,
                        dim=0, reduce='sum')
        x_sum_brodcasted = x_sum[idx]
        return x_sum_brodcasted

class NeuronInvariantDeepSetLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super(NeuronInvariantDeepSetLayer, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_in)
        )
        self.rho = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Linear(d_in, d_out)
        )

    def forward(self, batch):
        # Apply phi to each element in the set
        x_phi = self.phi(batch.x)

        # Sum over the elements in the set
        x_sum = self.get_x_sum(x=x_phi, idx=batch.batch)

        # Apply rho to the summed result
        x_sum = self.rho(x_sum)

        return x_sum

    def get_x_sum(self, x, idx):
        x_sum = scatter(src=x, index=idx,
                        dim=0, reduce='sum')
        return x_sum



# class NeuronEquivDeepSetLayer(nn.Module):
#     def __init__(self, d=128, L=6, d_hidden=32, d_output=32):
#         super(NeuronEquivDeepSetLayer, self).__init__()
#         self.phi = nn.Sequential(
#             nn.Linear(d*L, d_hidden*L),
#             nn.ReLU(),
#             nn.Linear(d_hidden*L, d_output*L)
#         )
#         self.rho = nn.Sequential(
#             nn.Linear(d*L, d_hidden*L),
#             nn.ReLU(),
#             nn.Linear(d_hidden*L, d_output*L)
#         )

#     def forward(self, batch):
#         # # Apply phi to each element in the set
#         x_phi = self.phi(batch.x)

#         # # Sum over the elements in the set
#         x_sum_brodcasted = self.get_x_sum_brod(x=batch.x, idx=batch.batch)
#         x_sum = self.rho(x_sum_brodcasted)

#         # # sum them
#         x = x_phi + x_sum
#         return x

#     def get_x_sum_brod(self, x, idx):
#         x_sum = scatter(src=x, index=idx,
#                         dim=0, reduce='sum')
#         x_sum_brodcasted = x_sum[idx]
#         return x_sum_brodcasted
#         # # Apply phi to each element in the set
#         # x_phi = self.phi(x)

#         # x_sum = x_phi.sum(dim=1)
#         # # Apply rho to the aggregated result
#         # output = self.rho(x_sum)
#         # return output


# class NeuronInvariantDeepSetLayer(nn.Module):
#     def __init__(self, d=128, L=6, d_hidden=32, d_output=1):
#         super(NeuronInvariantDeepSetLayer, self).__init__()
#         self.phi = nn.Sequential(
#             nn.Linear(d*L, d_hidden*L),
#             nn.ReLU(),
#             nn.Linear(d_hidden*L, d_hidden*L)
#         )
#         self.rho = nn.Sequential(
#             nn.Linear(d_hidden*L, d_output*L),
#             nn.ReLU(),
#             nn.Linear(d_output*L, d_output)
#         )

#     def forward(self, batch):
#         # Apply phi to each element in the set
#         x_phi = self.phi(batch.x)

#         # Sum over the elements in the set
#         x_sum = self.get_x_sum(x=x_phi, idx=batch.batch)

#         # Apply rho to the summed result
#         x_sum = self.rho(x_sum)

#         return x_sum

#     def get_x_sum(self, x, idx):
#         x_sum = scatter(src=x, index=idx,
#                         dim=0, reduce='sum')
#         return x_sum


def get_model(cfg):
    if cfg.model.symmetry == "set":
        if cfg.source_dataset.name == "zinc12k":
            return NeuronArchitecture(cfg=cfg)
        elif cfg.source_dataset.name == "cifar10":
            return NeuronArchitecture_images(cfg=cfg)
        else:
            raise NotImplementedError(f"Model for source dataset {cfg.source_dataset.name} not implemented")
    elif cfg.model.symmetry == "none":
        return MLPs(cfg=cfg)
    else:
        raise ValueError(f"Invalid symmetry type: {cfg.model.symmetry}. Expected 'set' or 'none'.")