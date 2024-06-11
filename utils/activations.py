from torch_geometric.data import DataLoader
from collections import defaultdict
import torch
from torch_geometric.data import DataLoader, InMemoryDataset, Data
from source_dataset_training import data
import logging
from tqdm import tqdm
import torch.nn as nn
import json


def register_hooks(cfg, model):
    activations = defaultdict(list)

    def get_activation(name):
        def hook(model, input, output):
            activations[name].append(output.detach().cpu())
        return hook

    if cfg.source_dataset.name == "zinc12k":
        for idx, layer in enumerate(model.bn_layers):  # Customize for specific layers
            # if isinstance(layer, (nn.BatchNorm1d)):
            layer.register_forward_hook(get_activation(idx))
    elif cfg.source_dataset.name == "cifar10":
        idx = 0
        for name, layer in model.named_modules():
            # Collect BatchNorm2d at the end of each block in the main layers
            if any(sub in name for sub in ['layer1', 'layer2', 'layer3', 'layer4']):
                if isinstance(layer, nn.BatchNorm2d) and \
                   ('bn2' in name or (name.endswith('bn1') and 'downsample' in name)):
                    layer.register_forward_hook(get_activation(idx))
            # Collect AdaptiveAvgPool2d and Linear after layer4
            elif name in ['avgpool', 'fc']:
                layer.register_forward_hook(get_activation(idx))
            idx = idx + 1
    else:
        raise Exception (f"Invalid source dataset: {cfg.source_dataset.name}.")

    return activations


class ActivationDataset(InMemoryDataset):
    def __init__(self, cfg, root, model, original_dataset, transform=None, pre_transform=None):
        self.cfg=cfg
        self.model = model
        self.original_dataset = original_dataset

        # Register hooks to capture activations
        self.hooks = register_hooks(cfg=self.cfg, model=self.model)
        self.max_neuron_number = cfg.source_dataset.max_neuron_number
        super().__init__(root, transform, pre_transform)
        self._data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'activation_data.pt'

    def process(self):
        if self.cfg.source_dataset.name == "zinc12k":
            self.process_graphs()
        elif self.cfg.source_dataset.name == "cifar10":
            self.process_images()
        else:
            raise Exception (f"Invalid source dataset: {self.cfg.source_dataset.name}.")

    def process_graphs(self):
        self.model.eval()
        data_list = []
        # count = 0
        # TODO: currently working only for regression tasks!
        
        with torch.no_grad():
            for batch in tqdm(self.original_dataset, desc="Processing batches", unit="batch"):
                batch = batch.to(self.cfg.general.device)
                y = self.model(batch)
                correctness = torch.abs(
                    y - batch.y).float().cpu()  # low means better

                # Collect activations
                layer_activations = {k: torch.cat(
                    v, 0) for k, v in self.hooks.items()}
                self.cfg.source_dataset.num_layers = len(layer_activations)
                # Concatenate activations across the feature dimension
                activations_concat = torch.cat(
                    [act for act in layer_activations.values()], dim=1)  # shape n x d*L
                if self.cfg.model.symmetry == "I": # adding padding
                    n, d = activations_concat.shape
                    if n > self.max_neuron_number:
                        raise ValueError(
                            "self.max_neuron_number must be greater than or equal to n")
                    activations_concat = torch.cat([activations_concat, torch.zeros(
                        (self.max_neuron_number - activations_concat.size(0), activations_concat.size(1)))], dim=0)
                    activations_concat = activations_concat.reshape(1, -1)
                elif self.cfg.model.symmetry == "S_n":
                    pass
                else:
                    raise ValueError(
                        f"Invalid symmetry type: {self.cfg.model.symmetry}. Expected 'set' or 'none'.")
                
                # Clear the activations for the next data point
                for k in self.hooks.keys():
                    self.hooks[k].clear()


                # Create a Data object
                x = activations_concat

                data_obj = Data(x=x,
                                y=correctness)
                data_list.append(data_obj)


        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # print(f"Data length before saving: {len(data_list)}")
        torch.save((data, slices), self.processed_paths[0])
        # print(f"Total processed elements: {count}")
        logging.info(f"Data list length: {len(data_list)}")
    
    def process_images(self):
        self.model.eval()
        data_list = []
        with torch.no_grad():
            for batch in tqdm(self.original_dataset, desc="Processing batches", unit="batch"):
                img, label = batch[0].to(self.cfg.general.device), batch[1].to(
                    self.cfg.general.device)
                y = self.model(img)
                correctness = torch.tensor([y.argmax().item() == label[0].item()])  # low means better
                # Collect activations
                layer_activations = {k: torch.cat(
                    v, 0) for k, v in self.hooks.items()}

                total_activations = torch.tensor([])
                layers = torch.tensor([])
                layer = 0
                d_max = self.cfg.source_dataset.model.dim_embed
                for layer_name, activations in layer_activations.items():
                    # print(f"Layer: {layer_name}")
                    # print(f"Activations: {activations}")
                    if layer_name == 67: # softmax layer
                        b, d = activations.shape
                        num_pixels = 1
                    else:
                        b, d, n_1, n_2 = activations.shape
                        num_pixels = n_1 * n_2
                    assert b == 1, "The batch size should be 1!"
                    activations = activations.reshape(num_pixels, d)
                    # Pad with zeros to reach the shape (512, d)
                    padding = torch.zeros((num_pixels, d_max - d),
                        device=activations.device, dtype=activations.dtype)
                    activations = torch.cat((activations, padding), dim=1)
                    
                    total_activations = torch.cat(
                        (total_activations, activations), dim=0)
                    layers = torch.cat(
                        (layers, 
                         torch.tensor([layer] * num_pixels)
                        ), dim=0)
                    layer += 1
                for k in self.hooks.keys():
                    self.hooks[k].clear()
                x = total_activations
                if self.cfg.model.symmetry == "I":  # adding padding
                    x = x.reshape(1, -1)
                indices = layers

                data_obj = Data(x=x,
                                num_layers = torch.tensor([layer]),
                                activation_idx=indices,
                                y=correctness)
                data_list.append(data_obj)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # print(f"Data length before saving: {len(data_list)}")
        torch.save((data, slices), self.processed_paths[0])
        # print(f"Total processed elements: {count}")
        logging.info(f"Data list length: {len(data_list)}")
    
    def len(self):
        return self.slices['x'].size(0)-1

    def get(self, idx):
        data = self._data.__class__()
        for key in self._data.keys():
            item, slices = self._data[key], self.slices[key]
            s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    def shuffle(self):
        pass
        # dataset.shuffle()
        # shuffled_data = self._data.__class__()
        # generator = torch.Generator().manual_seed(42)
        # indices = torch.randperm(
        #     len(shuffled_data), generator=generator).tolist()
        # # print(indices[:10])
        # # path = f'./source_dataset_training/{self.cfg.source_dataset.name}_indices.json'
        # shuffled_data = shuffled_data[indices]
        # self._data = shuffled_data

        # with open(path, 'r') as file:
        #     indices = json.load(file)

        # # Create a new data object to store shuffled data
        # shuffled_data = self._data.__class__()
        # for key in self._data.keys():
        #     item = self._data[key]
        #     slices = self.slices[key]
        #     # Create a list to store the shuffled items
        #     shuffled_items = []
        #     for idx in indices:
        #         s = slice(slices[idx], slices[idx + 1])
        #         shuffled_items.append(item[s])
        #     # Concatenate the shuffled items back into a single tensor
        #     shuffled_data[key] = torch.cat(shuffled_items, dim=0)

        # self._data = shuffled_data
    


def create_custom_dataloader(cfg, activation_dataset, manual_shuffle = False):
    manual_shuffle = False
    if manual_shuffle:
        activation_dataset.shuffle()
    return DataLoader(activation_dataset, batch_size=cfg.neuron_dataset.bs, shuffle=True, num_workers=cfg.neuron_dataset.num_workers, follow_batch=['x', 'deep_sets_indices'])


def get_activation_dataloaders(cfg, pre_trained_model):
    logging.info(
        f"Inferencing the {cfg.source_dataset.name} dataset to get the activations")

    dataloader, _ = data.get_dataloader(cfg=cfg, batch_size=1)
    activations_dataset_path = f'./activations_dataset/{cfg.source_dataset.name}'

    def prepare_activation_dataloader(cfg, dataset_type, dataset):
        logging.info(
            f"Preparing the activations of {cfg.source_dataset.name}, the {dataset_type} dataset: {len(dataset.dataset)} samples")
        
        activation_dataset = ActivationDataset(
            cfg=cfg, root=f"{activations_dataset_path}_{dataset_type}_{cfg.model.symmetry}", model=pre_trained_model, original_dataset=dataset)
        logging.info(
            
            f"Done!")
        
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(
            len(activation_dataset), generator=generator).tolist()
        # # print(indices[:10])
        # # path = f'./source_dataset_training/{self.cfg.source_dataset.name}_indices.json'
        activation_dataset = activation_dataset[indices]
        # activation_dataset.shuffle()
        activation_dataset = activation_dataset[:
                                                cfg.training.train_set_size_cap]
        return create_custom_dataloader(cfg=cfg, activation_dataset=activation_dataset)
    
    if cfg.source_dataset.name == "zinc12k":
        activation_train_dataloader = prepare_activation_dataloader(cfg=cfg,
            dataset_type="val", dataset=dataloader["val"])
        activation_val_test_dataloader = prepare_activation_dataloader(cfg=cfg,
            dataset_type="test", dataset=dataloader["test"])

        logging.info(
            "Dataset splitting:\n"
            "-------------------\n"
            "1. Using the validation set as the training set.\n"
            "2. Splitting the test set into validation and test subsets with the following ratios:\n"
            f"   - Validation subset: {cfg.neuron_dataset.val_ratio * 100:.2f}%\n"
            f"   - Test subset: {(1 - cfg.neuron_dataset.val_ratio) * 100:.2f}%"
        )
        activation_val_dataloader, activation_test_dataloader = split_dataloader(dataloader=activation_val_test_dataloader,val_size=cfg.neuron_dataset.val_ratio, shuffle=True, seed=42)

        return activation_train_dataloader, activation_val_dataloader, activation_test_dataloader
    
    elif cfg.source_dataset.name == "cifar10":

        logging.info(
            "Dataset splitting:\n"
            "-------------------\n"
            "2. Splitting the test set into training and validation subsets with the following ratios:\n"
            f"   - Training subset: {(1 - cfg.neuron_dataset.val_ratio) * 100:.2f}%\n"
            f"   - Validation subset: {cfg.neuron_dataset.val_ratio * 100:.2f}%\n"
            "3. Splitting the validation subset into validation and test subsets with the following ratios:\n"
            f"   - Validation subset: {cfg.neuron_dataset.val_ratio * cfg.neuron_dataset.val_ratio * 100:.2f}%\n"
            f"   - Test subset: {(cfg.neuron_dataset.val_ratio * cfg.neuron_dataset.val_ratio) * 100:.2f}%"
        )

        activation_test_dataloader = prepare_activation_dataloader(cfg,
            "test", dataloader["test"])
        activation_train_dataloader, activation_val_dataloader = split_dataloader(
            dataloader=activation_test_dataloader, val_size=cfg.neuron_dataset.val_ratio, shuffle=False, seed=cfg.general.seed)
        
        activation_val_dataloader, activation_test_dataloader = split_dataloader(
            dataloader=activation_val_dataloader, val_size=cfg.neuron_dataset.val_ratio, shuffle=False, seed=cfg.general.seed)


        return activation_train_dataloader, activation_val_dataloader, activation_test_dataloader




def split_dataloader(dataloader, val_size=0.2, shuffle=True, seed=42):
    """
    Splits a PyTorch Geometric DataLoader into validation and test DataLoaders.

    Args:
        dataloader (torch_geometric.data.DataLoader): The original PyTorch Geometric DataLoader.
        val_size (float, optional): The proportion of the dataset to include in the validation set. Default is 0.2.
        shuffle (bool, optional): Whether to shuffle the dataset before splitting. Default is True.
        seed (int, optional): The random seed to use for shuffling. Default is 42.

    Returns:
        tuple: A tuple containing two DataLoaders:
            - val_loader (torch_geometric.data.DataLoader): The validation DataLoader.
            - test_loader (torch_geometric.data.DataLoader): The test DataLoader.
    """
    # Get the dataset from the original DataLoader
    dataset = dataloader.dataset

    # Optionally shuffle the dataset
    if shuffle:
        # dataset.shuffle()
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator).tolist()
        # print(indices[:10])
        dataset = dataset[indices]

    # Calculate the split indices
    val_size = int(val_size * len(dataset))
    train_size = len(dataset) - val_size

    # Split the dataset into train and validation subsets
    val_dataset = dataset[:val_size]
    test_dataset = dataset[val_size:]

    # Create DataLoaders for validation and test sets
    val_loader = DataLoader(
        val_dataset, batch_size=dataloader.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=dataloader.batch_size, shuffle=False)

    return val_loader, test_loader
