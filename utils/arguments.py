import yaml
from easydict import EasyDict as edict
import argparse

# --------------------------------------------------------------------------------- #
#                               Main functions                                      #
# --------------------------------------------------------------------------------- #


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


# def override_config_with_args(cfg, return_nested=True):
#     # Create an argument parser
#     parser = argparse.ArgumentParser()

#     # Add arguments for each key in the YAML file
#     def add_argument(key, value, parser):
#         if isinstance(value, dict):
#             for sub_key, sub_value in value.items():
#                 add_argument(f"{key}__{sub_key}", sub_value, parser)
#         else:
#             arg_name = f"--{key.replace('.', '__')}"
#             arg_type = type(value)
#             parser.add_argument(arg_name, type=arg_type,
#                                 default=value, help=f"{key}")

#     for key, value in cfg.items():
#         add_argument(key, value, parser)

#     # Parse the command-line arguments
#     args = parser.parse_args()

#     # Update the YAML configuration with the command-line arguments
#     def update_config(cfg, args, prefix=""):
#         for key, value in cfg.items():
#             arg_name = f"{prefix}__{key}" if prefix else key
#             arg_name = arg_name.replace('.', '__')
#             arg_value = getattr(args, arg_name, None)
#             if arg_value is not None:
#                 if isinstance(value, dict):
#                     update_config(value, args, arg_name)
#                 else:
#                     cfg[key] = arg_value

#     update_config(cfg, args)

#     if return_nested:
#         return cfg
#     else:
#         return vars(args)
    
def override_config_with_args(cfg, return_nested=True):
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--general__seed", type=int,
                        default=cfg['general']['seed'], help="seed")
    parser.add_argument("--general__device", type=int,
                        default=cfg['general']['device'], help="device")

    # source_dataset
    parser.add_argument("--source_dataset__name", type=str,
                        default=cfg['source_dataset']['name'], help="name")

    # neuron_dataset
    parser.add_argument("--neuron_dataset__bs", type=int,
                        default=cfg['neuron_dataset']['bs'], help="neuron_bs")
    parser.add_argument("--neuron_dataset__num_workers", type=int,
                        default=cfg['neuron_dataset']['num_workers'], help="neuron_num_workers")
    parser.add_argument("--neuron_dataset__val_ratio", type=float,
                        default=cfg['neuron_dataset']['val_ratio'], help="neuron_val_ratio")

    # model
    parser.add_argument("--model__symmetry", type=str,
                        default=cfg['model']['symmetry'], help="symmetry")
    parser.add_argument("--model__num_layers", type=int,
                        default=cfg['model']['num_layers'], help="num_layers")
    parser.add_argument("--model__dim_embed", type=int,
                        default=cfg['model']['dim_embed'], help="dim_embed")
    parser.add_argument("--model__residual", type=bool,
                        default=cfg['model']['residual'], help="residual")
    parser.add_argument("--model__task", type=str,
                        default=cfg['model']['task'], help="task")
    parser.add_argument("--model__dim_output", type=int,
                        default=cfg['model']['dim_output'], help="dim_output")

    # training
    parser.add_argument("--training__lr", type=float,
                        default=cfg['training']['lr'], help="lr")
    parser.add_argument("--training__task", type=str,
                        default=cfg['training']['task'], help="task")
    parser.add_argument("--training__optim", type=str,
                        default=cfg['training']['optim'], help="optim")
    parser.add_argument("--training__sched__type", type=str,
                        default=cfg['training']['sched']['type'], help="sched_type")
    parser.add_argument("--training__sched__step_size", type=int,
                        default=cfg['training']['sched']['step_size'], help="sched_step_size")
    parser.add_argument("--training__sched__gamma", type=float,
                        default=cfg['training']['sched']['gamma'], help="sched_gamma")
    parser.add_argument("--training__wd", type=float,
                        default=cfg['training']['wd'], help="wd")
    parser.add_argument("--training__epochs", type=int,
                        default=cfg['training']['epochs'], help="epochs")

    # wandb
    parser.add_argument("--wandb__project_name", type=str,
                        default=cfg['wandb']['project_name'], help="project_name")

    args = parser.parse_args()

    # general
    cfg['general']['seed'] = args.general__seed
    cfg['general']['device'] = args.general__device

    # source_dataset
    cfg['source_dataset']['name'] = args.source_dataset__name


    # neuron_dataset
    cfg['neuron_dataset']['bs'] = args.neuron_dataset__bs
    cfg['neuron_dataset']['num_workers'] = args.neuron_dataset__num_workers
    cfg['neuron_dataset']['val_ratio'] = args.neuron_dataset__val_ratio

    # model
    cfg['model']['symmetry'] = args.model__symmetry
    cfg['model']['num_layers'] = args.model__num_layers
    cfg['model']['dim_embed'] = args.model__dim_embed
    cfg['model']['residual'] = args.model__residual
    cfg['model']['task'] = args.model__task
    cfg['model']['dim_output'] = args.model__dim_output

    # training
    cfg['training']['lr'] = args.training__lr
    cfg['training']['task'] = args.training__task
    cfg['training']['optim'] = args.training__optim
    cfg['training']['sched']['type'] = args.training__sched__type
    cfg['training']['sched']['step_size'] = args.training__sched__step_size
    cfg['training']['sched']['gamma'] = args.training__sched__gamma
    cfg['training']['wd'] = args.training__wd
    cfg['training']['epochs'] = args.training__epochs

    # wandb
    cfg['wandb']['project_name'] = args.wandb__project_name

    if return_nested:
        return cfg
    else:
        return vars(args)
# --------------------------------------------------------------------------------- #
#                           Helpers for Main functions                              #
# --------------------------------------------------------------------------------- #

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
