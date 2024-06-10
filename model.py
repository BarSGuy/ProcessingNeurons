import torch
import torch.nn as nn
import torch
from torch_scatter import scatter


##############################################################
###################### Main architecure ######################
##############################################################
class NeuronArchitecture(nn.Module):
    def __init__(self, d=128, L=6, d_hidden=32, d_final=1, 
                 num_layers=4) -> None:
        super(NeuronArchitecture, self).__init__()
        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(NeuronEquivDeepSetLayer(
            d=d, L=L, d_hidden=d_hidden, d_output=d_hidden))

        # Hidden layers
        for layer in range(1, num_layers - 1):
            self.layers.append(NeuronEquivDeepSetLayer(
                d=d_hidden, L=L, d_hidden=d_hidden, d_output=d_hidden))

        # Final layer
        self.layers.append(NeuronInvariantDeepSetLayer(
            d=d_hidden, L=L, d_hidden=d_hidden, d_output=d_final))
        


    def forward(self, batch):
        for layer in self.layers:
            batch.x = layer(batch)
        return batch.x
            


##############################################################
######################### Helpers ############################
##############################################################
class NeuronEquivDeepSetLayer(nn.Module):
    def __init__(self, d=128, L=6, d_hidden=32, d_output=32):
        super(NeuronEquivDeepSetLayer, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(d*L, d_hidden*L),
            nn.ReLU(),
            nn.Linear(d_hidden*L, d_output*L)
        )
        self.rho = nn.Sequential(
            nn.Linear(d*L, d_hidden*L),
            nn.ReLU(),
            nn.Linear(d_hidden*L, d_output*L)
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
        # # Apply phi to each element in the set
        # x_phi = self.phi(x)

        # x_sum = x_phi.sum(dim=1)
        # # Apply rho to the aggregated result
        # output = self.rho(x_sum)
        # return output


class NeuronInvariantDeepSetLayer(nn.Module):
    def __init__(self, d=128, L=6, d_hidden=32, d_output=1):
        super(NeuronInvariantDeepSetLayer, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(d*L, d_hidden*L),
            nn.ReLU(),
            nn.Linear(d_hidden*L, d_hidden*L)
        )
        self.rho = nn.Sequential(
            nn.Linear(d_hidden*L, d_output*L),
            nn.ReLU(),
            nn.Linear(d_output*L, d_output)
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
