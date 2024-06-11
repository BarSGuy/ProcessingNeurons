import torch.nn as nn
import torch
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import average_precision_score
from ogb.graphproppred import Evaluator
import numpy as np
from collections import defaultdict
from tqdm import tqdm 
import logging
import copy
import wandb


def get_num_epochs(cfg):
    dataset_name = cfg.source_dataset.name
    epochs = {
        'zinc12k': 1000,
    }
    if dataset_name not in epochs:
        raise ValueError(
            f"No loss function available for the dataset: {dataset_name}")

    return epochs[dataset_name]
# --------------------------------- loss function -------------------------------- #


def get_loss_func(cfg):
    dataset_name = cfg.source_dataset.name
    dataset_info = {
        'zinc12k': (nn.L1Loss(), 'minimize', 'regression'),
    }

    if dataset_name not in dataset_info:
        raise ValueError(
            f"No loss function available for the dataset: {dataset_name}")
    return dataset_info[dataset_name]

class RMSELoss(nn.MSELoss):
    def forward(self, output, target):
        mse = super().forward(output, target)
        return torch.sqrt(mse)

# --------------------------------- optimizer -------------------------------- #


def get_optim_func(cfg, model):
    dataset_name = cfg.source_dataset.name
    optimizers = {
        'zinc12k': torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=0),

    }
    if dataset_name not in optimizers:
        raise ValueError(
            f"No Optimizer available for the dataset: {dataset_name}")

    return optimizers[dataset_name]


# --------------------------------- scheduler -------------------------------- #


def get_sched_func(cfg, optim):
    dataset_name = cfg.source_dataset.name
    sched = {
        'zinc12k': torch.optim.lr_scheduler.StepLR(optim,
                                                   step_size=300,
                                                   gamma=0.5,
                                                   )
    }

    if dataset_name not in sched:
        raise ValueError(
            f"No Scheduler available for the dataset: {dataset_name}")

    return sched[dataset_name]


# --------------------------------- evaluator -------------------------------- #


def get_evaluator(cfg):
    dataset_name = cfg.source_dataset.name
    evaluators = {
        'zinc12k': ZincLEvaluator(),
    }

    if dataset_name not in evaluators:
        raise ValueError(
            f"No loss function available for the dataset: {dataset_name}")

    return evaluators[dataset_name]



class ZincLEvaluator(nn.L1Loss):
    def forward(self, input_dict):
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        return super().forward(y_pred, y_true)

    def eval(self, input_dict):
        L1_val = self.forward(input_dict)
        L1_val_dict = {
            'L1loss': L1_val.item()
        }
        return L1_val_dict


# --------------------------------- training -------------------------------- #


def train_loop(model, loader, critn, optim, epoch, device, task='regression'):
    model.train()
    loss_list = []
    pbar = tqdm(loader, total=len(loader))
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        optim.zero_grad()
        if task == 'classification':
            is_labeled = batch.y == batch.y
            pred = model(batch)  # pred is 128 x 12

            labeled_y = batch.y.to(torch.float32)[is_labeled]
            labeled_pred = pred.to(torch.float32)[is_labeled]
            # TODO: make sure this 2 lines are ok
            labeled_y = labeled_y.reshape(-1)
            labeled_pred = labeled_pred.reshape(-1)

            assert labeled_y.shape == labeled_pred.shape
            loss = critn(labeled_pred, labeled_y)
        elif task == 'regression':
            pred = model(batch).view(batch.y.shape)
            loss = critn(pred, batch.y)
        else:
            raise ValueError(
                f"Invalid task type: {task}. Expected 'regression' or 'classification'.")

        loss.backward()
        optim.step()

        loss_list.append(loss.item())
        pbar.set_description(
            f"Epoch {epoch} Train Step {i}: Loss = {loss.item()}")
        # wandb.log({"Epoch": epoch, "Train Step": i, "Train Loss": loss.item()})

    return loss_list


def eval_loop(model, loader, eval, device, average_over=1, task='regression'):
    model.eval()
    # TODO: for average_over > 0 this works only if test/val dataloader doesn't shuffle!!!
    input_dict_for_votes = []
    for vote in range(average_over):
        pbar = tqdm(loader, total=len(loader),
                    desc=f"Vote {vote + 1} out of {average_over} votes")
        pred, true = [], []
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            with torch.no_grad():
                if task == 'classification':
                    model_pred = model(batch)
                    true.append(batch.y.view(model_pred.shape).detach().cpu())
                    pred.append(model_pred.detach().cpu())
                elif task == 'regression':
                    true.append(batch.y)
                    pred.append(model(batch).view(batch.y.shape))
                else:
                    raise ValueError(
                        f"Invalid task type: {task}. Expected 'regression' or 'classification'.")

        input_dict = {
            "y_true": torch.cat(true, dim=0),
            "y_pred": torch.cat(pred, dim=0)
        }

        input_dict_for_votes.append(input_dict)
    average_votes_dict = average_dicts(*input_dict_for_votes)
    input_dict = average_votes_dict
    metric = eval.eval(input_dict)
    # TODO: assuming 'metric' is a dictionary with 1 single key!
    metric = list(metric.values())[0]
    return metric



def sched_step(cfg, sched, val_metric):
    dataset_name = cfg.source_dataset.name
    if dataset_name == "zinc12k":
        sched.step()
    else:
        raise NotImplementedError(
            f"Schec step for source dataset {cfg.source_dataset.name} not implemented")

# --------------------------------- training - helpers -------------------------------- #


def average_dicts(*input_dicts):
    # Check if there is at least one dictionary
    if len(input_dicts) == 0:
        raise ValueError("No dictionaries provided.")

    # If only one dictionary is provided, return it as is
    if len(input_dicts) == 1:
        return input_dicts[0]

    # Check if all dictionaries have the same keys
    keys = set(input_dicts[0].keys())
    if not all(keys == set(d.keys()) for d in input_dicts):
        raise ValueError("All dictionaries must have the same keys.")

    averaged_dict = {}

    num_dicts = len(input_dicts)
    for key in keys:
        # Sum the tensors from all dictionaries for the same key
        total_tensor = sum(d[key] for d in input_dicts)

        # Calculate the average
        averaged_tensor = total_tensor / num_dicts
        averaged_dict[key] = averaged_tensor

    return averaged_dict
