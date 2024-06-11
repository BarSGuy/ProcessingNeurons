
import torch_geometric.data as data
import torch_geometric as pyg
import logging
import torch
try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    pass
from torch.utils.data import DataLoader


def get_dataloader(cfg, batch_size=None):
    if batch_size is not None:
        if cfg.source_dataset.name == "zinc12k":
            return get_zinc12k_dataloader(batch_size=batch_size)
        elif cfg.source_dataset.name == "cifar10":
            return get_cifar10_dataloader(batch_size=batch_size)
        else:
            raise NotImplementedError(f"Source dataset {cfg.source_dataset.name} not implemented")
    else:
        if cfg.source_dataset.name == "zinc12k":
            return get_zinc12k_dataloader(batch_size=128)
        else:
            raise NotImplementedError(f"Source dataset {cfg.source_dataset.name} not implemented")


def get_zinc12k_dataloader(batch_size=128):
    zinc_dataloader = {
        name: data.DataLoader(
            pyg.datasets.ZINC(
                split=name,
                subset=True,
                root='./datasets/zinc',
            ),
            batch_size=batch_size,
            num_workers=4,
            shuffle=(name == "train"),
        )
        for name in ["train", "val", "test"]
    }
    num_elements_in_target = 1
    return zinc_dataloader, num_elements_in_target


def get_cifar10_dataloader(batch_size=128):
    # Define the transform to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    ])

    # Create a dictionary of DataLoader for train, validation, and test sets
    cifar10_dataloader = {
        'test': DataLoader(
            torchvision.datasets.CIFAR10(
                root='./datasets/cifar10',
                train=False,
                download=True,
                transform=transform
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
    }
    num_elements_in_target = 10
    return cifar10_dataloader, num_elements_in_target

def get_n_max_of_dataset(cfg):
    if cfg.source_dataset.name == "zinc12k":
        try:
            # Load the ZINC12k dataset for all splits
            zinc_dataset = {
                name: pyg.datasets.ZINC(
                    split=name, subset=True, root='./datasets/zinc')
                for name in ["train", "val", "test"]
            }

            # Combine all datasets and find the maximum number of nodes
            max_nodes = max(
                max(data.num_nodes for data in zinc_dataset["train"]),
                max(data.num_nodes for data in zinc_dataset["val"]),
                max(data.num_nodes for data in zinc_dataset["test"])
            )

            logging.info(
                f"Maximum number of nodes in ZINC12k dataset: {max_nodes}")
            cfg.source_dataset.max_nodes = max_nodes
            return max_nodes

        except Exception as e:
            logging.error(
                f"An error occurred while processing the ZINC12k dataset: {e}")
            raise e
    elif cfg.source_dataset.name == "cifar10":
        cfg.source_dataset.max_nodes = -1
        max_nodes = -1
        return max_nodes
    else:
        raise NotImplementedError(
            f"Source dataset {cfg.source_dataset.name} not implemented")
