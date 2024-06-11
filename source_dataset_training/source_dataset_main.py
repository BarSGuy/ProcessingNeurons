import utils.webiases as u_wandb
import logging
from source_dataset_training import training, models, data
import wandb
import torch
from utils import my_logging as u_log
from tqdm import tqdm
import os
from source_dataset_training.resnet import *




def get_pretrained_model_on_source_dataset(cfg):
    logging.info(f"Loading the pre-trained model on {cfg.source_dataset.name}.")
    device = torch.device(f"cuda:{cfg.general.device}")
    torch.manual_seed(cfg.general.seed)
    
    if cfg.source_dataset.name == 'zinc12k':
        pretrained_model_path = f'./pre_trained_models/{cfg.source_dataset.name}.pth'
        if os.path.exists(pretrained_model_path):
            # Load the pretrained model state dictionary
            model = models.get_model(cfg)
            model = model.to(device)
            model.load_state_dict(torch.load(pretrained_model_path))
            return model
        else:
            logging.info(
                f"The pre-trained model on {cfg.source_dataset.name} doesn't exists! Training from scratch!.")
            train_zinc12k_from_scratch(cfg=cfg)
            model = models.get_model(cfg)
            model = model.to(device)
            return model
    elif cfg.source_dataset.name == 'cifar10':
        model = models.get_model(cfg)
        model = model.to(device)
        return model
    else:
        raise ValueError(f"Invalid source dataset: {cfg.source_dataset.name}.")




##############################################################
################### Train on source datasets #################
##############################################################

def train_zinc12k_from_scratch(cfg):
    device = torch.device(f"cuda:{cfg.general.device}")
    logging.info(f"         Pre-trained model -- Setting wandb.")
    u_wandb.set_wandb(cfg=cfg, on_neurons=False)

    logging.info(
        f"         Pre-trained model -- Loading source dataset {cfg.source_dataset.name}.")
    dataloader, _ = data.get_dataloader(cfg)

    logging.info(f"         Pre-trained model -- Loading the model.")
    model = models.get_model(cfg)

    wandb.watch(model)
    model = model.to(device)

    logging.info(f"         Pre-trained model -- Loading loss function.")

    critn, goal, task = training.get_loss_func(cfg=cfg)
    assert task in ['regression', 'classification'], \
        f"Invalid task type: {task}. Expected 'regression' or 'classification'."

    logging.info(f"         Pre-trained model -- Loading optimizer.")
    optim = training.get_optim_func(cfg=cfg, model=model)
    num_epochs = training.get_num_epochs(cfg=cfg)

    logging.info(f"         Pre-trained model -- Loading schedular.")
    sched = training.get_sched_func(cfg=cfg, optim=optim)

    logging.info(f"         Pre-trained model -- Loading evaluator.")
    eval = training.get_evaluator(cfg=cfg)

    logging.info(f"         Pre-trained model -- Starting Training.")
    best_metrics = u_log.initialize_best_metrics(cfg=cfg, goal=goal, training_on_source=True)

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        logging.info(f"         Pre-trained model -- Train loop.")
        # =========================== Training =========================== #
        loss_list = training.train_loop(
            model=model, loader=dataloader["train"], critn=critn, optim=optim, epoch=epoch, device=device, task=task)
        logging.info(f"         Pre-trained model -- Validation loop.")
        # =========================== Validation =========================== #
        val_metric = training.eval_loop(
            model=model, loader=dataloader["val"], eval=eval, device=device, average_over=1, task=task)
        test_metric = training.eval_loop(
            model=model, loader=dataloader["test"], eval=eval, device=device, average_over=1, task=task)
        # =========================== logging  =========================== #
        best_metrics = u_log.update_best_metrics(cfg=cfg,
                                                 best_metrics=best_metrics, val_metric=val_metric, test_metric=test_metric, epoch=epoch, goal=goal, training_on_source=True)
        u_log.log_wandb(cfg=cfg, epoch=epoch, optim=optim, loss_list=loss_list, val_metric=val_metric,
                        test_metric=test_metric, best_metrics=best_metrics, training_on_source=True)
        u_log.set_posfix(optim=optim, loss_list=loss_list, val_metric=val_metric,
                         test_metric=test_metric, pbar=pbar)
        training.sched_step(cfg=cfg, sched=sched, val_metric=val_metric)

    # Ensure the directory exists
    os.makedirs('./pre_trained_models', exist_ok=True)
    # Save the model state dictionary
    torch.save(model.state_dict(), './pre_trained_models/zinc12k.pth')
    wandb.finish()
