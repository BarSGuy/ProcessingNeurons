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
    
def initialize_best_metrics(goal = 'minimize'):
    assert goal in ["minimize", "maximize"], "Invalid goal: must be either 'minimize' or 'maximize'"
    return {
        "val_loss": float('inf') if goal == "minimize" else float('-inf'),
        "test_loss": float('inf') if goal == "minimize" else float('-inf'),
        "epoch": -1
    }


def update_best_metrics(best_metrics, val_metric, test_metric, epoch, val_risks=None, val_coverages=None, test_risks=None, test_coverages=None, goal='minimize'):
    assert goal in ["minimize", "maximize"], "Invalid goal: must be either 'minimize' or 'maximize'"
    if (goal == "minimize" and val_metric < best_metrics["val_loss"]) or \
       (goal == "maximize" and val_metric > best_metrics["val_loss"]):
        try:
            best_metrics.update({
                "val_loss": val_metric,
                "test_loss": test_metric,
                "epoch": epoch,
                "val_risks": val_risks,
                "val_coverages": val_coverages.cpu().tolist(),
                "test_risks": test_risks,
                "test_coverages": test_coverages.cpu().tolist(),
                "auc_val": metrics.auc(val_coverages.cpu().tolist(), val_risks),
                "auc_test": metrics.auc(test_coverages.cpu().tolist(), test_risks),
            })
        except AttributeError as e:
            best_metrics.update({
                "val_loss": val_metric,
                "test_loss": test_metric,
                "epoch": epoch
            })
        
    return best_metrics



def log_wandb(epoch, optim, loss_list, val_metric, test_metric, best_metrics):
    try:
        lr = optim.param_groups[0]['lr']
    except (KeyError, IndexError, AttributeError) as e:
        logging.info(f"An error occurred while accessing the learning rate: {e}")
        lr = -1
    
    wandb.log({
        "Epoch": epoch,
        "Train Loss": np.mean(loss_list),
        "Val Loss": val_metric,
        "Test Loss": test_metric,
        "Learning Rate": lr,  # Log the learning rate
        # unpack best metrics into the lognv
        **{f"best_{key}": value for key, value in best_metrics.items()}
    })
    # if epoch % 10 == 0:
    #     data = [[x, y] for (x, y) in zip(val_coverages.cpu(), val_risks)]
    #     table = wandb.Table(data=data, columns=["x", "y"])
    #     fig, ax = plot_coverage_risk(
    #         coverages=val_coverages.cpu(), risks=val_risks)
    #     wandb.log({f"Rc_curve_val_epoch_{epoch}": wandb.Image(fig)})
    #     data = [[x, y] for (x, y) in zip(val_coverages.cpu(), val_risks)]
    #     table = wandb.Table(data=data, columns=["x", "y"])
    #     wandb.log({f"Val_rc_table_epoch_{epoch}": table})
        
    #     fig, ax = plot_coverage_risk(
    #         coverages=test_coverages.cpu(), risks=test_risks)
    #     wandb.log({f"Rc_curve_test_epoch_{epoch}": wandb.Image(fig)})
    #     data = [[x, y] for (x, y) in zip(test_coverages.cpu(), test_risks)]
    #     table = wandb.Table(data=data, columns=["x", "y"])
    #     wandb.log({f"Test_rc_table_epoch_{epoch}": table})
    # # Create a table for the graph data
    # data = [[x, y] for (x, y) in zip(val_coverages, val_risks)]
    # table = wandb.Table(data=data, columns=["x", "y"])


    # wandb.log(
    #     {
    #         f"{epoch}": wandb.plot.line(
    #             table, "x", "y", title="risk coverage"
    #         )
    #     }
    # )


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
