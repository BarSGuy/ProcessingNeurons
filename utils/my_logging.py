import matplotlib.pyplot as plt
import logging
import wandb
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

# --------------------------------- logging code -------------------------------- #

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Initialized logger")


# --------------------------------- logging wandb -------------------------------- #
    
def initialize_best_metrics(cfg, goal = 'minimize', training_on_source=False):
    assert goal in ["minimize", "maximize"], "Invalid goal: must be either 'minimize' or 'maximize'"
    if training_on_source:
        return {
            "val_loss": float('inf') if goal == "minimize" else float('-inf'),
            "test_loss": float('inf') if goal == "minimize" else float('-inf'),
            "epoch": -1
        }
        
    if cfg.model.task == "gap_minimization":
        return {
            "Val Delta gap": float('inf'),
            "Test Delta gap": float('inf'),
            "val_risks": [],
            "val_coverages": [],
            "test_risks": [],
            "test_coverages": [],
            "Auc Val": 1,
            "Auc Test": 1,
            "epoch": -1,
        }


def update_best_metrics(cfg, best_metrics, val_metric, test_metric, epoch, val_risks=None, val_coverages=None, test_risks=None, test_coverages=None, goal='minimize', training_on_source=False):
    assert goal in ["minimize", "maximize"], "Invalid goal: must be either 'minimize' or 'maximize'"
    if training_on_source:
        if (goal == "minimize" and val_metric < best_metrics["val_loss"]) or \
                (goal == "maximize" and val_metric > best_metrics["val_loss"]):
            best_metrics.update({
                "val_loss": val_metric,
                "test_loss": test_metric,
                "epoch": epoch
            })
        return best_metrics
    
    
    if cfg.model.task == "gap_minimization":
        if (goal == "minimize" and metrics.auc(val_coverages.cpu().tolist(), val_risks) < best_metrics["Auc Val"]):
        # or \(goal == "maximize" and val_metric > best_metrics["val_loss"]):
            try:
                best_metrics.update({
                    "Val Delta gap": val_metric,
                    "Test Delta gap": test_metric,
                    "epoch": epoch,
                    "val_risks": val_risks,
                    "val_coverages": val_coverages.cpu().tolist(),
                    "test_risks": test_risks,
                    "test_coverages": test_coverages.cpu().tolist(),
                    "Auc Val": metrics.auc(val_coverages.cpu().tolist(), val_risks),
                    "Auc Test": metrics.auc(test_coverages.cpu().tolist(), test_risks),
                })
            except AttributeError as e:
                best_metrics.update({
                    "val_loss": val_metric,
                    "test_loss": test_metric,
                    "epoch": epoch
                })
            
        return best_metrics


def log_wandb(cfg, epoch, optim, loss_list, val_metric, test_metric, best_metrics, val_risks=None, val_coverages=None, test_risks=None, test_coverages=None, training_on_source=False):
    try:
        lr = optim.param_groups[0]['lr']
    except (KeyError, IndexError, AttributeError) as e:
        logging.info(f"An error occurred while accessing the learning rate: {e}")
        lr = -1
    if training_on_source:
        wandb.log({
            "Epoch": epoch,
            "Train Loss": np.mean(loss_list),
            "Val Loss": val_metric,
            "Test Loss": test_metric,
            "Learning Rate": lr,  # Log the learning rate
            # unpack best metrics into the lognv
            **{f"best_{key}": value for key, value in best_metrics.items()}
        })
        return

    if cfg.model.task == "gap_minimization":
        wandb.log({
            "Epoch": epoch,
            "Train Delta gap": np.mean(loss_list),
            "Val Delta gap": val_metric,
            "Test Delta gap": test_metric,
            "Auc Val": metrics.auc(val_coverages.cpu().tolist(), val_risks),
            "Auc Test": metrics.auc(test_coverages.cpu().tolist(), test_risks),
            "Learning Rate": lr,  # Log the learning rate
            # unpack best metrics into the lognv
            **{f"best_{key}": value for key, value in best_metrics.items()}
        })


def plot_coverage_risk(coverages, risks):
    # Create the plot
    fig, ax = plt.subplots()

    # Plotting the data
    ax.plot(coverages, risks,
            marker='o', linestyle='-', color='b', label='Risk vs. Coverage')

    # Adding title and labels
    ax.set_title("Risks vs. Coverages")
    ax.set_xlabel("Coverages")
    ax.set_ylabel("Risks")

    # Adding a grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adding a legend
    ax.legend()

    # Adding minor ticks
    ax.minorticks_on()

    # Improving the layout
    fig.tight_layout()

    return fig, ax


def set_posfix(optim, loss_list, val_metric, test_metric, pbar):
    try:
        lr = optim.param_groups[0]['lr']
    except (KeyError, IndexError, AttributeError) as e:
        logging.info(f"An error occurred while accessing the learning rate: {e}")
        lr = -1
    pbar.set_postfix({
        "lr": lr,
        "loss": np.mean(loss_list),
        "val": val_metric,
        "test": test_metric
    })
