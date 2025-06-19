import matplotlib.pyplot as plt
import torch
# import numpy as np
import os
import os.path as osp

dataset = 'nturgbd'
evaluation = 'CV'
# dataset = 'ucf101'
# evaluation = '2'

checkpoint_base_path = f'results/{dataset}/{evaluation}/{dataset}_{evaluation}_base.pt'
checkpoint_abs_path = f'results/{dataset}/{evaluation}/{dataset}_{evaluation}_abs.pt'
checkpoint_avg_path = f'results/{dataset}/{evaluation}/{dataset}_{evaluation}_avg.pt'
checkpoint_cnn_path = f'results/{dataset}/{evaluation}/{dataset}_{evaluation}_cnn.pt'

checkpoint_base = torch.load(checkpoint_base_path, map_location='cpu')
checkpoint_abs = torch.load(checkpoint_abs_path, map_location='cpu')
checkpoint_avg = torch.load(checkpoint_avg_path, map_location='cpu')
checkpoint_cnn = torch.load(checkpoint_cnn_path, map_location='cpu')

results_base = checkpoint_base['results']
results_abs = checkpoint_abs['results']
results_avg = checkpoint_avg['results']
results_cnn = checkpoint_cnn['results']

# Get the epochs as the x-axis
x = results_base['epoch']

# Plot root path
plot_root = 'results/plots/'
os.makedirs(plot_root, exist_ok=True) # Make dir if it doesn't exist

# Evaluation string and Dataset string dictionaries for plot titles
eval_str_dict = {'CS': 'Cross Subject', 'CV': 'Cross View', 
                 '1': 'Evaluation 1', '2': 'Evaluation 2', '3': 'Evaluation 3'}
dataset_str_dict = {'nturgbd': 'NTU RGB+D', 'nturgbd120': 'NTU RGB+D 120', 'ucf101':'UCF-101'}

fig_title = f'{dataset_str_dict[dataset]} - {eval_str_dict[evaluation]}'
print(f'Plotting results for {dataset_str_dict[dataset]} - {eval_str_dict[evaluation]} evaluation')

# Create a directory, ignoring if it already exists
save_root = osp.join(plot_root, f'{dataset}-{evaluation}/')
os.makedirs(save_root, exist_ok=True)

# ------------------------------
#   Loss comparison graphing
# ------------------------------

def plot_loss_comp(
        fig_title: str, x, save_root,
        results_base=None, 
        results_abs=None, 
        results_avg=None, 
        results_cnn=None
        ):
    """
    Plot the loss comparison graph for different models.
    Args:
        fig_title (str): Title of the figure.
        x (array): Epochs.
        results_base (dict): Results for the base model.
        results_abs (dict): Results for the absolute flow model.
        results_avg (dict): Results for the average flow model.
        results_cnn (dict): Results for the CNN model.
    """
    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(25,7))
    fig.suptitle(f'Loss Comparison - {fig_title}', fontsize=30)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Loss', fontsize=20)

    results = [results_base, results_abs, results_avg, results_cnn]
    plot_params = [
        {'label': 'Base', 'color': 'tab:olive'},
        {'label': 'Absolute flow', 'color': 'tab:green'},
        {'label':  'Average flow', 'color': 'tab:pink'},
        {'label': 'CNN learning', 'color': 'tab:blue'}
    ]

    # Plot class loss
    ax1.set_title('Classification Loss', fontsize = 25)
    ax2.set_title('Feature Loss', fontsize = 25)
    ax3.set_title('Reconstruction Loss', fontsize = 25)

    for axis, loss_type in {ax1: 'cls', ax2: 'feature', ax3:'recon'}.items():
        for result, plot_param in zip(results, plot_params):
            try:
                axis.plot(x, result[f'train_{loss_type}_loss'],
                         **plot_param)
            except TypeError:
                print(f"Failed to plot {plot_param['label']} {loss_type} loss")

    ax1.legend()
    plt.tight_layout()
    os.makedirs(osp.join(save_root,'loss_comparison'), exist_ok=True)
    plt.savefig(osp.join(save_root,'loss_comparison/Loss_comparison.png'))

# ------------------------------
#   Accuracy comparison graphing
# ------------------------------

# Convert results to percentage
def to_percentage(result):
    return [r*100 for r in result]

def plot_accuracy(
        fig_title: str, x, save_root,
        results_base=None, 
        results_abs=None, 
        results_avg=None, 
        results_cnn=None
        ):
    """
    Plot the accuracy comparison graph for different models.
    Args:
        fig_title (str): Title of the figure.
        x (array): Epochs.
        save_root (str): Path to save the plot.
        results_base (dict): Results for the base model.
        results_abs (dict): Results for the absolute flow model.
        results_avg (dict): Results for the average flow model.
        results_cnn (dict): Results for the CNN model.
    """
    # Create figure and axes
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,8))
    fig.suptitle(f'Accuracy Comparison - {fig_title}', fontsize=30)
    for ax in (ax1, ax2):
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylim(0, 100) # 0-100% accuracy
        ax.set_xlim(0, 70)  # 0-70 epochs
        ax.axvline(x=50, color='black', linestyle='--', alpha=0.2) # Epoch 50
        ax.axvline(x=60, color='black', linestyle='--', alpha=0.2) # Epoch 60
        # ax.set_xscale('log')
    ax1.set_ylabel('Classification accuracy %', fontsize=20)

    results = [results_base, results_abs, results_avg, results_cnn]
    plot_params = [
        {'label': 'Base', 'color': 'tab:olive'},
        {'label': 'Absolute flow', 'color': 'tab:green'},
        {'label':  'Average flow', 'color': 'tab:pink'},
        {'label': 'CNN learning', 'color': 'tab:blue'}
    ]


    for result, plot_param in zip(results, plot_params):
        try:
            # Plot train and test AUC
            ax1.plot(x, to_percentage(result['train_AUC']),
                     **plot_param)
            ax2.plot(x, to_percentage(result['test_AUC']),
                     **plot_param)
        except TypeError:
            print(f"Failed to plot {plot_param['label']}")

    # Set titles and legends
    ax1.set_title('Train accuracy', fontsize=25)
    ax2.set_title('Test accuracy', fontsize=25)
    ax1.legend(loc=4, prop={'size':20})

    # Print max AUC for each model
    for name, results in {'Base': results_base, 
                        'Absolute': results_abs, 
                        'Average': results_avg, 
                        'CNN': results_cnn}.items():
        print(f"{name}: {to_percentage(results['test_ACC'][-1])[-1]:.2f}%")
        # print(f"{name}: {max(to_percentage(results['test_AUC'])):.2f}%")

    os.makedirs(osp.join(save_root,'accuracy'), exist_ok=True)
    plt.savefig(osp.join(save_root,'accuracy/Acc_comparison_simple.png'))


# ------------------------------
#   Classification accuracy for observation ratios
# ------------------------------ 

def plot_obs_acc(fig_title: str, save_root,
                  results_base=None, 
                  results_abs=None, 
                  results_avg=None,
                  results_cnn=None):
    """
    Plot the accuray for the observation ratios 
    Network provides a classification per-frame (64 frame input)
    Args:
        fig_title (str): Title of the figure.
        x (array): Epochs.
        results_base (dict): Results for the base model.
        results_abs (dict): Results for the absolute flow model.
        results_avg (dict): Results for the average flow model.
        results_cnn (dict): Results for the CNN model.
    """
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(25,10))
    fig.suptitle(f'Loss Comparison - {fig_title}', fontsize=30)
    for ax in (ax1, ax2):
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('Observation Ratio', fontsize=20)
    ax1.set_ylabel('Accuracy', fontsize=20)

    results = [results_base, results_abs, results_avg, results_cnn]
    plot_params = [
        {'label': 'Base', 'color': 'tab:olive'},
        {'label': 'Absolute flow', 'color': 'tab:green'},
        {'label':  'Average flow', 'color': 'tab:pink'},
        {'label': 'CNN learning', 'color': 'tab:blue'}
    ]
    percents = ((torch.arange(64)+1)*(100/64)).int()

    # Plot class loss
    ax1.set_title('Train Accuracy', fontsize = 25)
    ax2.set_title('Test Accuracy', fontsize = 25)

    for result, plot_param in zip(results, plot_params):
        try:
            ax1.plot(percents, torch.tensor(result['train_ACC'][-1]),
                        **plot_param)
            ax2.plot(percents, torch.tensor(result['test_ACC'][-1]),
                        **plot_param)
        except TypeError:
            print(f"Failed to plot {plot_param['label']}")

    ax1.legend(loc='lower right', fontsize='large')
    plt.tight_layout()
    os.makedirs(osp.join(save_root,'accuracy'), exist_ok=True)
    plt.savefig(osp.join(save_root,'accuracy/Observation_Acc.png'))

# # ------------------------------
# #   Individual loss graphing
# # ------------------------------

# statistics = ['cls_loss', 'recon_loss', 'feature_loss']
# statistic_str_dict = {'cls_loss': 'Classification Loss', 
#                  'recon_loss': 'Reconstruction Loss',
#                  'feature_loss': 'Feature Loss'}


# # Plot reconstruction loss
# # ax.plot(x, results_base[f'train_{statistic}'], label='Base model', color='tab:olive')
# # ax.plot(x, results_cnn[f'train_{statistic}'], label='CNN', color='tab:blue')
# # ax.plot(x, results_lin[f'train_{statistic}'], label='Linear', color='tab:red')
# # ax.plot(x, results_abs[f'train_{statistic}'], label='Flow magnitude averaged', color='tab:green')
# # ax.plot(x, results_avg[f'train_{statistic}'], label='Average flow direction (x,y)', color='tab:pink')
# # ax.legend()

# def base_comparison(results_dict, statistic: str, state: str = 'train'):
#     comparison = (np.array(results_dict[f'{state}_{statistic}']) - np.array(results_base[f'{state}_{statistic}'])) / np.array(results_base[f'{state}_{statistic}'])
#     return comparison

# for statistic in statistics:
#     # Plot each statistic's loss with comparison to base
#     fig, ax = plt.subplots( figsize=(10, 10))
#     # ax.set_title(f'{statistic_str_dict[statistic]} - Base comparison (Train)\n{dataset_str_dict[dataset]} - {eval_str_dict[evaluation]}', fontsize=20)
#     ax.set_title(f'{statistic_str_dict[statistic]} - Base comparison (Train)\n{dataset_str_dict[dataset]}', fontsize=20) # UCF101
#     ax.tick_params(axis='both', which='major', labelsize=15)
#     ax.set_xlabel('Epoch', fontsize=15)
#     ax.set_ylabel('Loss', fontsize=15)
#     # ax.set_ylim(0.0, 0.3) # 0-0.5 loss
#     # ax.set_xlim(0, 60)  # 0-60 epochs
#     # ax.set_xscale('log')
#     ax.plot(x, base_comparison(results_cnn, statistic), label='CNN', color='tab:blue')
#     ax.plot(x, base_comparison(results_abs, statistic), label='Flow magnitude averaged', color='tab:green')
#     ax.plot(x, base_comparison(results_avg, statistic), label='Average flow direction (x,y)', color='tab:pink')
#     ax.hlines(0, 0, 70, colors='black', linestyles='--', label='Base model loss', color='tab:olive')
#     ax.legend()

#     # Adjust the bottom margin to make more space for the caption
#     plt.subplots_adjust(bottom=0.15)

#     caption = f"Comparison of {statistic_str_dict[statistic].lower()} to base model." + \
#     "\nLoss=$\\frac{\\text{Model loss} - \\text{Base model loss}}{\\text{Base model loss}}$"
#     fig.text(.5, .03, caption, ha='center', fontsize=10)

#     # plt.savefig(f'results/plots/{dataset}-{evaluation}/loss_individual/{statistic}-base_comparison.png')
#     plt.savefig(f'results/plots/{dataset}/loss_individual/{statistic}-base_comparison.png')


# # ------------------------------
# #   conf matrix testing
# # ------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# # Assuming y_true and y_pred are your true labels and predicted labels
# y_true = np.random.randint(0, 101, size=1000)  # Example data
# y_pred = np.random.randint(0, 101, size=1000)  # Example data

# cm = confusion_matrix(y_true, y_pred)

# plt.figure(figsize=(20, 20))
# sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()


# # ------------------------------
# # Training time
# # ------------------------------


if __name__ == '__main__':
    # Plot the loss comparison
    plot_loss_comp(
        fig_title, x, save_root, 
        results_base=results_base, 
        results_abs=results_abs,
        results_avg=results_avg,
        results_cnn=results_cnn
        )
    
    # Plot the accuracy comparison
    plot_accuracy(
        fig_title, x, save_root, 
        results_base=results_base, 
        results_abs=results_abs,
        results_avg=results_avg,
        results_cnn=results_cnn
        )

    # Plot the accuracy comparison across different observation ratios
    plot_obs_acc(
        fig_title, save_root,
        results_base=results_base, 
        results_abs=results_abs,
        results_avg=results_avg,
        results_cnn=results_cnn
        )
    # plot_accuracy(fig_title, x, save_root, results_base, results_abs, results_avg, results_cnn)
    # plt.show()
