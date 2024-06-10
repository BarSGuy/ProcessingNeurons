import torch
import torch.nn as nn
from collections import defaultdict
import torch_geometric.data as data
from torch_geometric.data import DataLoader, InMemoryDataset, Data
import os.path as osp

def register_hooks(model):
    activations = defaultdict(list)

    def get_activation(name):
        def hook(model, input, output):
            activations[name].append(output.detach().cpu())
        return hook

    for idx, layer in enumerate(model.bn_layers):  # Customize for specific layers
        # if isinstance(layer, (nn.BatchNorm1d)):  
        layer.register_forward_hook(get_activation(idx))

    return activations


class ActivationDataset(InMemoryDataset):
    def __init__(self, root, model, original_dataset, transform=None, pre_transform=None):
        self.model = model
        self.original_dataset = original_dataset

        # Register hooks to capture activations
        self.hooks = register_hooks(self.model)

        super().__init__(root, transform, pre_transform)
        self._data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'activation_data.pt'

    def process(self):
        self.model.eval()
        data_list = []
        count = 0
        with torch.no_grad():
            for batch in self.original_dataset:
                batch = batch.to(self.model.device)
                y = self.model(batch)
                correctness = torch.abs(y - batch.y).float().cpu() # low means better

                # Collect activations
                layer_activations = {k: torch.cat(
                    v, 0) for k, v in self.hooks.items()}

                # Concatenate activations across the feature dimension
                activations_concat = torch.cat(
                    [act for act in layer_activations.values()], dim=1)

                # Clear the activations for the next data point
                for k in self.hooks.keys():
                    self.hooks[k].clear()


                # # src dst for deep sets
                # deep_sets_indices_all_layers = []
                # current_index = 0
                # for layer, act in layer_activations.items():
                #     layer_size = act.shape[0]
                #     deep_sets_indices = torch.repeat_interleave(
                #         torch.tensor([current_index]), layer_size).reshape(-1, 1)
                #     deep_sets_indices_all_layers.append(deep_sets_indices)
                #     current_index += 1
                # num_layers = torch.tensor(current_index)

                # Create a Data object
                x = activations_concat
                # make a list of tensors a torch tensor 
                # deep_sets_indices_all_layers = torch.cat(
                #     deep_sets_indices_all_layers, dim=0)
                # data_obj = Data(x=x, activation_idx=deep_sets_indices_all_layers,
                #                 y=correct, num_layers=num_layers)
                data_obj = Data(x=x,
                                y=correctness)
                data_list.append(data_obj)
                count += 1
                print(f"Processed {count} elements")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print(f"Data length before saving: {len(data_list)}")
        torch.save((data, slices), self.processed_paths[0])
        print(f"Total processed elements: {count}")
        print(f"Data list length: {len(data_list)}")

    def len(self):
        return self.slices['x'].size(0)-1

    def get(self, idx):
        data = self._data.__class__()

        for key in self._data.keys:
            item, slices = self._data[key], self.slices[key]
            s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data


def create_custom_dataloader(activation_dataset, batch_size=128, num_workers=4):
    return DataLoader(activation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, follow_batch=['x', 'deep_sets_indices'])

if __name__ == "__main__":

    pass
    
# import torch

# def create_dataset(pretrain_model, dataloader):
#     device = torch.device(f"cuda:0")
#     model = model.to(device)
#     # Register hooks
#     activations = {}

#     def get_activation(name):
#         def hook(model, input, output):
#             activations[name] = output.detach()
#         return hook

#     # Register hooks on GNN layers and batch normalization layers
#     for i, bn_layer in enumerate(pretrain_model.bn_layers):
#         bn_layer.register_forward_hook(get_activation(f'bn_layer_{i}'))
        
#     pretrain_model.eval()
#     for batch in dataloader:
#         batch = batch.to(device)
#         with torch.no_grad():
#             num_nodes = batch.num_nodes
#             _ = model(batch)
#             # Print the activations
#             for layer_name, activation in activations.items():
#                 print(f'{layer_name} activation shape: {activation.shape}')
#             exit()

    
        