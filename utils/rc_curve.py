import torch
import matplotlib.pyplot as plt
import json
import numpy as np
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
    if run_number2 == None:
        coverages_set = best_metrics_set["test_coverages"]
        risks_set = best_metrics_set["test_risks"]
        save_path = f'./output/{task}/{source_dataset}/{run_number1}.pdf'
        plot_rc_curve(save_path=save_path, coverages1=coverages_set, risks1=risks_set, legend1="Symmetry-based Model",
                      coverages2=coverages_set, risks2=risks_set, legend2="Symmetry-based Model")
    elif run_number1 == None:
        coverages_none = best_metrics_none["test_coverages"]
        risks_none = best_metrics_none["test_risks"]
        save_path = f'./output/{task}/{source_dataset}/{run_number2}.pdf'
        plot_rc_curve(save_path=save_path, coverages1=coverages_none, risks1=risks_none, legend1="MLP Model",
                      coverages2=coverages_none, risks2=risks_none, legend2="MLP Model")
    else:
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


def plot_mean_std_test_risks(file1, file2, file3, output_file = './output/mean_std_test_risks.pdf'):
    # Load the data from the JSON files
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    with open(file3, 'r') as f:
        data3 = json.load(f)

    # Extract test_coverages (assuming they are the same across all files)
    test_coverages = data1['test_coverages']

    # Extract test_risks
    test_risks_1 = data1['test_risks']
    test_risks_2 = data2['test_risks']
    test_risks_3 = data3['test_risks']

    # Calculate mean and standard deviation of test_risks
    test_risks_all = np.array([test_risks_1, test_risks_2, test_risks_3])
    mean_test_risks = np.mean(test_risks_all, axis=0)
    std_test_risks = np.std(test_risks_all, axis=0)
    sem_test_risks = std_test_risks / np.sqrt(test_risks_all.shape[0])
    
    # Plot the mean test risks with standard deviation
    plt.figure(figsize=(10, 6))
    plt.errorbar(test_coverages, mean_test_risks, yerr=sem_test_risks,
                 fmt='-o', capsize=5, label='Mean Test Risks with STD')
    plt.xlabel('Test Coverages')
    plt.ylabel('Mean Test Risks')
    plt.title('Mean Test Risks as a Function of Test Coverages')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.show()


def plot_comparison_mean_sem_test_risks(files1, label1, files2, label2, output_file='./output/mean_std_test_risks_two_methods.pdf'):
    def load_and_compute_mean_sem(files):
        # Load the data from the JSON files
        datasets = [json.load(open(file, 'r')) for file in files]

        # Extract test_coverages (assuming they are the same across all files)
        test_coverages = datasets[0]['test_coverages']

        # Extract test_risks
        test_risks_all = np.array([data['test_risks'] for data in datasets])

        # Calculate mean and SEM of test_risks
        mean_test_risks = np.mean(test_risks_all, axis=0)
        std_test_risks = np.std(test_risks_all, axis=0)
        sem_test_risks = std_test_risks / np.sqrt(test_risks_all.shape[0])

        return test_coverages, mean_test_risks, sem_test_risks

    # Load and compute mean and SEM for the first set of files
    test_coverages1, mean_test_risks1, sem_test_risks1 = load_and_compute_mean_sem(
        files1)

    # Load and compute mean and SEM for the second set of files
    test_coverages2, mean_test_risks2, sem_test_risks2 = load_and_compute_mean_sem(
        files2)

    # Plot the mean test risks with SEM for both runs
    plt.figure(figsize=(10, 6))
    plt.errorbar(test_coverages1, mean_test_risks1,
                 yerr=sem_test_risks1, fmt='-o', capsize=6, label=label1)
    plt.errorbar(test_coverages2, mean_test_risks2,
                 yerr=sem_test_risks2, fmt='-s', capsize=6, label=label2)

    plt.xlabel('Coverage', fontsize=16)
    plt.ylabel('Selective Risk', fontsize=16)
    # plt.title('Comparison of Mean Test Risks as a Function of Test Coverages')
    # Increase the font size of the ticks
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()

if __name__ == '__main__':
    pred_gap = torch.tensor([0.3, 0.1, 0.4, 0.2, 0.5])
    actual_gap = torch.tensor([0.2, 0.4, 0.3, 0.1, 0.5])
    get_rc_curve(pred_gap, actual_gap)

    # plot_mean_std_test_risks(
    #     file1='./output/gap_minimization/zinc12k/5622295645I',
    #     file2='./output/gap_minimization/zinc12k/3092817727I',
    #     file3='./output/gap_minimization/zinc12k/1347824332I',)

    plot_comparison_mean_sem_test_risks(
        files1=['./output/gap_minimization/zinc12k/1512913526S_n', './output/gap_minimization/zinc12k/8770004378S_n', './output/gap_minimization/zinc12k/9318345743S_n'], label1='Symmetry-based Model',
        files2=['./output/gap_minimization/zinc12k/1347824332I', './output/gap_minimization/zinc12k/3092817727I', './output/gap_minimization/zinc12k/5622295645I'], label2='MLP',
    )
