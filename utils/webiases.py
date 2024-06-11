import wandb
import logging

def set_wandb(cfg, on_neurons = False):
    if on_neurons:
        tag = f"Neurons_of__{cfg.source_dataset.name}||Model_symmetry__{cfg.model.symmetry}||LR_{cfg.training.lr}||Num_layers_{cfg.model.num_layers}||Dim_embed_{cfg.model.dim_embed}||Epochs_{cfg.training.epochs}||Residual_{cfg.model.residual}||Run_number_{cfg.general.run_number}"
    else:
        tag = f"Dataset__{cfg.source_dataset.name}||Not_training_on_neurons"
    logging.info(f"{tag}")
    wandb.init(settings=wandb.Settings(
        start_method='thread'), project=cfg.wandb.project_name, name=tag, config=cfg)
