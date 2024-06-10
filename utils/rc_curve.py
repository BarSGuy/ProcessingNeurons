import torch
import matplotlib.pyplot as plt


def get_rc_curve(pred_gap, actual_gap):
    # Ensure coverages tensor is created on the same device as kappas for compatibility
    coverages = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], device=pred_gap.device)

    # Sort pred_gap and reorder y_true and y_pred accordingly
    pred_gap = pred_gap.reshape(-1)
    pred_gap, indices = torch.sort(pred_gap)
    actual_gap = actual_gap.reshape(-1)
    actual_gap = actual_gap[indices]
    # actual_gap = actual_gap/torch.sum(actual_gap)

    # Calculate accumulated risks for each coverage
    total_length = len(actual_gap)
    accumulated_actual_gaps = [
        actual_gap[:int(c * total_length)].mean().item() for c in coverages]

    return accumulated_actual_gaps, coverages


def plot_rc_curve_from_best_metrics(task, source_dataset, best_metrics_set, best_metrics_none, run_number1, run_number2):
    coverages_set = best_metrics_set["test_coverages"]
    risks_set = best_metrics_set["test_risks"]
    coverages_none = best_metrics_none["test_coverages"]
    risks_none = best_metrics_none["test_risks"]
    save_path = f'./output/{task}/{source_dataset}/{run_number1}_{run_number2}.pdf'
    plot_rc_curve(save_path=save_path, coverages1=coverages_set, risks1=risks_set, legend1="Symmetry-based Model",
                  coverages2=coverages_none, risks2=risks_none, legend2="MLP Model")


def plot_rc_curve(save_path, coverages1, risks1, legend1, coverages2, risks2, legend2):
    # Create the plot
    fig, ax = plt.subplots()

    # Plotting the first dataset
    ax.plot(coverages1, risks1,
            marker='o', linestyle='-', color='b', label=legend1)

    # Plotting the second dataset
    ax.plot(coverages2, risks2,
            marker='s', linestyle='--', color='r', label=legend2)

    # Adding title and labels
    ax.set_title("Risks vs. Coverages")
    ax.set_xlabel("Coverages")
    ax.set_ylabel("Risks")
    
    # Setting x ticks
    ax.set_xticks([i/10 for i in range(1, 11)])
    # Adding legend
    ax.legend()
    # Adding a grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Show the plot
    plt.show()
    
    # Save the figure
    fig.savefig(save_path,
            format='pdf', dpi=300)



if __name__ == '__main__':
    pred_gap = torch.tensor([0.3, 0.1, 0.4, 0.2, 0.5])
    actual_gap = torch.tensor([0.2, 0.4, 0.3, 0.1, 0.5])
    get_rc_curve(pred_gap, actual_gap)
