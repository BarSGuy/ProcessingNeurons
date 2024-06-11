import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Callable, Type
import logging
from tqdm import tqdm

class Config:
    # Dummy class for type hinting
    training: Any
    
    
    
def get_loss_func(cfg):
    task_to_loss = {
        'regression': nn.L1Loss(),
        'classification': nn.BCELoss()
    }

    if cfg.training.task not in task_to_loss:
        raise ValueError(
            f"No loss function available for the task: {cfg.training.task}")

    return task_to_loss[cfg.training.task]


def get_goal(cfg):
    return "minimize"

def get_eval_func(cfg: Config) -> Callable:
    task_to_eval = {
        'regression': RegressionEval(),
        'classification': ClassificationEval()
    }

    task = cfg.training.task
    if task not in task_to_eval:
        raise ValueError(
            f"No evaluator function available for the task: {task}")

    return task_to_eval[task]


def get_optim_func(cfg: Config, model: nn.Module) -> Optimizer:
    optim_type_to_func = {
        'adam': torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.wd)
    }

    optim_type = cfg.training.optim
    if optim_type not in optim_type_to_func:
        raise ValueError(
            f"No optimizer function available for the optimizer type: {optim_type}")

    return optim_type_to_func[optim_type]


def get_sched_func(cfg: Config, optim: Optimizer) -> _LRScheduler:
    sched_type_to_func = {
        'steplr': torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.training.sched.step_size, gamma=cfg.training.sched.gamma)
    }

    sched_type = cfg.training.sched.type
    if sched_type not in sched_type_to_func:
        raise ValueError(
            f"No scheduler function available for the scheduler type: {sched_type}")

    return sched_type_to_func[sched_type]

class RegressionEval(nn.L1Loss):
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


class ClassificationEval(nn.BCELoss):
    def forward(self, input_dict):
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        return super().forward(y_pred, y_true.to(torch.float32))

    def eval(self, input_dict):
        L1_val = self.forward(input_dict)
        L1_val_dict = {
            'BCELoss': L1_val.item()
        }
        return L1_val_dict

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
    return metric, torch.cat(pred, dim=0), torch.cat(true, dim=0)


def sched_step(cfg, sched, val_metric):
    if cfg.training.sched.type == 'steplr':
        sched.step()
    else:
        raise ValueError(
            f"Invalid scheduler type: {cfg.training.sched.type}")
