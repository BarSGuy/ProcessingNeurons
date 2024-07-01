import matplotlib.pyplot as plt
import pandas as pd
# import utils.synthetic_experiment as se
# import experiments.Cifar10_exp_sets as ce
# import experiments.Cifar10_exp_conv_multigrid as ce
import experiments.Cifar10_exp_conv_cat as ce10
import experiments.Cifar100_exp_conv_cat as ce100
import experiments.ImageNet_exp_conv_cat as im
from easydict import EasyDict as edict

cfg = edict({
    ## data
    'dataset': 'imagenet',
    'batch_size': 128,
    
    ## model
    'model_symmetry': 'Cn',
    'hidden_dim': 128,
    
    ## optimizer
    'lr': 0.01,
    'weight_decay': 0,
    'num_epochs': 50,
    
    ## general
    'seed': 42
})


if cfg.dataset == 'cifar10':
    ce10.run(cfg=cfg)
elif cfg.dataset == 'cifar100':
    ce100.run(cfg=cfg)
elif cfg.dataset == 'imagenet':
    im.run(cfg=cfg)

exit()
# se.run(cfg=cfg)


def plot_best_test_loss_from_csv(csv_file_path='./output/results.csv',
                                 output_image_path='./output/best_test_loss.pdf'):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Plot best_test_loss as a function of num_data_points_train for each model
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        plt.plot(model_df['num_data_points_train'],
                 model_df['best_test_loss'], marker='o', label=model)

    # Add labels and title
    plt.xlabel('Number of Training Data Points')
    plt.ylabel('Best Test Loss')
    plt.title(
        'Best Test Loss vs. Number of Training Data Points for Different Models')
    plt.legend()
    plt.grid(True)
    # Set log scale for x-axis and invert y-axis
    plt.xscale('log')
    plt.gca().invert_yaxis()
    # Save the plot as an image file
    plt.savefig(output_image_path)
    plt.show()


plot_best_test_loss_from_csv()
exit()
# from utils.rc_curve import plot_rc_curve_from_best_metrics
# import utils.arguments as u_args
# import utils.activations as u_act
# import utils.rc_curve as u_rc
# from tqdm import tqdm
# import logging
# import utils.my_logging as u_log
# import torch
# import utils.webiases as u_wandb
# from source_dataset_training import source_dataset_main
# from activation_models import models as act_models
# import wandb
# import utils.training as u_train
# import json
from utils.experiments import *
from utils.rc_curve import *

# plot_mean_std_test_risks(
#     file1='./output/gap_minimization/zinc12k/5622295645I',
#     file2='./output/gap_minimization/zinc12k/3092817727I',
#     file3='./output/gap_minimization/zinc12k/1347824332I',)

plot_comparison_mean_sem_test_risks(
    files1=['./output/gap_minimization/zinc12k/1512913526S_n', './output/gap_minimization/zinc12k/8770004378S_n', './output/gap_minimization/zinc12k/9318345743S_n'], label1='Symmetry-based Model',
    files2=['./output/gap_minimization/zinc12k/1347824332I', './output/gap_minimization/zinc12k/3092817727I', './output/gap_minimization/zinc12k/5622295645I'], label2='MLP',
    )
# train_activation_model()
# get_certainty_of_activation_model(task='certainty', source_dataset_name='zinc12k',
#                                 run_number1='8816900536set', run_number2='3952329103none')
exit()

# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #
# ===================================================================================================================== #


# ===================================================================================================================== #
# ========================================               logging               ======================================== #
# ===================================================================================================================== #
u_log.setup_logging()
# ===================================================================================================================== #
# ========================================               load args             ======================================== #
# ===================================================================================================================== #
logging.info(f"Loading arguments.")

cfg_file__path = './config/config.yaml'
cfg = u_args.load_config(path=cfg_file__path)
cfg = u_args.override_config_with_args(cfg=cfg)
cfg.source_dataset.dir = "./source_datasets/" + cfg.source_dataset.name



# ===================================================================================================================== #
# ===================================               set device and seed             =================================== #
# ===================================================================================================================== #
logging.info(f"Setting device and seed.")

device = torch.device(f"cuda:{cfg.general.device}")
torch.manual_seed(cfg.general.seed)


# ===================================================================================================================== #
# ========================================          get pretrained model        ======================================= #
# ===================================================================================================================== #
logging.info(f"Getting the pre-trained model on the source dataset {cfg.source_dataset.name}")
pre_trained_model = source_dataset_main.get_pretrained_model_on_source_dataset(cfg=cfg)


# ===================================================================================================================== #
# ======================================          get activations dataset        ====================================== #
# ===================================================================================================================== #
logging.info(
    f"Getting the activation dataset for the source dataset {cfg.source_dataset.name}")
train_dataloader, val_dataloader, test_dataloader = u_act.get_activation_dataloaders(
    cfg=cfg, pre_trained_model=pre_trained_model)



# ===================================================================================================================== #
# ========================================               load wandb             ======================================= #
# ===================================================================================================================== #
logging.info(f"Setting wandb.")
u_wandb.set_wandb(cfg=cfg, on_neurons=True)

# ===================================================================================================================== #
# ========================================               model              =========================================== #
# ===================================================================================================================== #
logging.info(f"Loading the model.")
model = act_models.get_model(cfg=cfg)
wandb.watch(model)
model = model.to(device)

# ===================================================================================================================== #
# ========================================              training            =========================================== #
# ===================================================================================================================== #
logging.info(f"Loading loss function.")
critn, goal = u_train.get_loss_func(cfg=cfg)

logging.info(f"Loading optimizer.")
optim = u_train.get_optim_func(cfg=cfg, model=model)

logging.info(f"Loading schedular.")
sched = u_train.get_sched_func(cfg=cfg, optim=optim)

logging.info(f"Loading evaluator.")
eval = u_train.get_eval_func(cfg=cfg)

# ===================================================================================================================== #
# ========================================            starting training         ======================================= #
# ===================================================================================================================== #
logging.info(f"Starting Training.")

best_metrics = u_log.initialize_best_metrics(goal=goal)

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
    
    best_metrics = u_log.update_best_metrics(
        best_metrics=best_metrics, val_metric=val_metric, test_metric=test_metric, epoch=epoch, goal=goal, val_risks=val_risks, val_coverages=val_coverages, test_risks=test_risks, test_coverages=test_coverages)
    
    u_log.log_wandb(epoch=epoch, optim=optim, loss_list=loss_list, val_metric=val_metric,
                    test_metric=test_metric, best_metrics=best_metrics, val_risks=val_risks, val_coverages=val_coverages,test_risks=test_risks, test_coverages=test_coverages)
    
    u_log.set_posfix(optim=optim, loss_list=loss_list, val_metric=val_metric,
                     test_metric=test_metric, pbar=pbar)
    
    u_train.sched_step(cfg=cfg, sched=sched, val_metric=val_metric)

path_for_best_matrics = f'./output/{cfg.model.task}/{cfg.source_dataset.name}/Model_{cfg.model.symmetry}||Layers_{cfg.model.num_layers}||Dim_embed_{cfg.model.dim_embed}||LR_{cfg.training.lr}||Optim_{cfg.training.optim}||Sched_{cfg.training.sched.type}||Step_size_{cfg.training.sched.step_size}||WD_{cfg.training.wd}||Epochs_{cfg.training.epochs}||Residual_{cfg.model.residual}.json'

u_log.save_dict_to_path(
    data=best_metrics, path=path_for_best_matrics)


wandb.finish()


