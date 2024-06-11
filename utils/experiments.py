from utils.rc_curve import plot_rc_curve_from_best_metrics
import utils.arguments as u_args
import utils.activations as u_act
import utils.rc_curve as u_rc
from tqdm import tqdm
import logging
import utils.my_logging as u_log
import torch
import utils.webiases as u_wandb
from source_dataset_training import source_dataset_main
from activation_models import models as act_models
import wandb
import utils.training as u_train
import json
import utils.output_saver as u_out



def train_activation_model():
    u_log.setup_logging()
    logging.info(f"Loading arguments.")

    cfg_file__path = './config/config.yaml'
    cfg = u_args.load_config(path=cfg_file__path)
    cfg = u_args.override_config_with_args(cfg=cfg)
    cfg.source_dataset.dir = "./source_datasets/" + cfg.source_dataset.name
    cfg.general.run_number = str(u_out.generate_unique_10_digit_number()) + str(cfg.model.symmetry)
    logging.info(f"Setting device and seed.")


    device = torch.device(f"cuda:{cfg.general.device}")
    torch.manual_seed(cfg.general.seed)
    logging.info(
        f"Getting the pre-trained model on the source dataset {cfg.source_dataset.name}")


    pre_trained_model = source_dataset_main.get_pretrained_model_on_source_dataset(
        cfg=cfg)
    logging.info(
        f"Getting the activation dataset for the source dataset {cfg.source_dataset.name}")


    train_dataloader, val_dataloader, test_dataloader = u_act.get_activation_dataloaders(
        cfg=cfg, pre_trained_model=pre_trained_model)

    logging.info(f"Setting wandb.")
    u_wandb.set_wandb(cfg=cfg, on_neurons=True)
    
    logging.info(f"Loading the model.")
    model = act_models.get_model(cfg=cfg)
    wandb.watch(model)
    model = model.to(device)
    
    logging.info(f"Loading loss function.")
    critn = u_train.get_loss_func(cfg=cfg)
    
    logging.info(f"Loading goal.")
    goal = u_train.get_goal(cfg=cfg)
    
    logging.info(f"Loading optimizer.")
    optim = u_train.get_optim_func(cfg=cfg, model=model)

    logging.info(f"Loading schedular.")
    sched = u_train.get_sched_func(cfg=cfg, optim=optim)

    logging.info(f"Loading evaluator.")
    eval = u_train.get_eval_func(cfg=cfg)


    logging.info(f"Starting Training.")
    best_metrics = u_log.initialize_best_metrics(cfg=cfg, goal=goal)

    pbar = tqdm(range(cfg.training.epochs))
    for epoch in pbar:

        loss_list = u_train.train_loop(
            model=model, loader=train_dataloader, critn=critn, optim=optim, epoch=epoch, device=device, task=cfg.training.task)

        val_metric, val_pred_gap, val_actual_gap = u_train.eval_loop(
            model=model, loader=val_dataloader, eval=eval, device=device, average_over=1, task=cfg.training.task)

        test_metric, test_pred_gap, test_actual_gap = u_train.eval_loop(
            model=model, loader=test_dataloader, eval=eval, device=device, average_over=1, task=cfg.training.task)

        val_risks, val_coverages = u_rc.get_rc_curve(
            pred_gap=val_pred_gap, actual_gap=val_actual_gap)

        test_risks, test_coverages = u_rc.get_rc_curve(
            pred_gap=test_pred_gap, actual_gap=test_actual_gap)

        best_metrics = u_log.update_best_metrics(cfg=cfg,
            best_metrics=best_metrics, val_metric=val_metric, test_metric=test_metric, epoch=epoch, goal=goal, val_risks=val_risks, val_coverages=val_coverages, test_risks=test_risks, test_coverages=test_coverages)

        u_log.log_wandb(cfg=cfg, epoch=epoch, optim=optim, loss_list=loss_list, val_metric=val_metric,
                        test_metric=test_metric, val_risks=val_risks, val_coverages=val_coverages, test_risks=test_risks, test_coverages=test_coverages, best_metrics=best_metrics)

        u_log.set_posfix(optim=optim, loss_list=loss_list, val_metric=val_metric,
                        test_metric=test_metric, pbar=pbar)

        u_train.sched_step(cfg=cfg, sched=sched, val_metric=val_metric)
    
    
    u_out.save_dict_to_path(
        data=best_metrics, path=f'./output/{cfg.model.task}/{cfg.source_dataset.name}/{str(cfg.general.run_number)}')

    wandb.finish()
    if cfg.model.symmetry == "S_n":
        plot_rc_curve_from_best_metrics(
            task=cfg.model.task, source_dataset=cfg.source_dataset.name, best_metrics_set=best_metrics, best_metrics_none=best_metrics, run_number1=cfg.general.run_number, run_number2=None)
    elif cfg.model.symmetry == "I":
        plot_rc_curve_from_best_metrics(
            task=cfg.model.task, source_dataset=cfg.source_dataset.name, best_metrics_set=best_metrics, best_metrics_none=best_metrics, run_number1=None, run_number2=cfg.general.run_number)
    return cfg


def get_certainty_of_activation_model(task, source_dataset_name, run_number1, run_number2):
    u_log.setup_logging()
    logging.info(f"Loading arguments.")

    # cfg_file__path = './config/config.yaml'
    # cfg = u_args.load_config(path=cfg_file__path)
    # cfg = u_args.override_config_with_args(cfg=cfg)
    # cfg.source_dataset.dir = "./source_datasets/" + cfg.source_dataset.name



    set_path = f'./output/{task}/{source_dataset_name}/{str(run_number1)}'
    assert "set" in set_path, f"The path {set_path} is not the set path!"
    best_metrics_set = u_out.load_dict_from_path(
        path=set_path)
    
    none_path = f'./output/{task}/{source_dataset_name}/{str(run_number2)}'
    assert "none" in none_path, f"The path {none_path} is not the none path!"
    best_metrics_none = u_out.load_dict_from_path(
        path=none_path)
    
    plot_rc_curve_from_best_metrics(task=task, source_dataset=source_dataset_name, best_metrics_set=best_metrics_set,
                                    best_metrics_none=best_metrics_none, run_number1=run_number1, run_number2=run_number2)
