import os
import os.path as osp

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay
# import numpy as np

# ----------------------------------------------------
# USAGE:
# - edit the `dataset` and `evaluation` variables below
# - Set values of plot_* as needed, if True, that plot will be generated
# - edit the `plot_data` for specific extensions (keys) and plot params
#      python ./results/visualisations/results_vis.py

dataset = 'ntu'  # ntu, ntu120, ucf101
evaluation = 'CV'  # CS/CV, CSub/CSet, 1/2/3
dilation = 3

# In case you didn't want to
loss = True
acc = True
obs = True
confusion = False # REQUIRES EVAL
cls_pred = False # REQUIRES EVAL

# ----------------------------------------------------

# Where to load the results dictionaries (checkpoints) from
results_root = f"results/{dataset}/{evaluation}/train"
# We use the train results dict for now

# Create a directory, ignoring if it already exists
save_root = osp.join('results/plots/', f'{dataset}/{evaluation}/')
os.makedirs(save_root, exist_ok=True)

# Define plotting parameters, the extension is the name of the checkpoint
# e.g.   extension = 'base_NO_MASK'
#        filename = f"{results_root}/nturgbd_CV_{extension}.pt
# Change plot_params such as what you want the plot label to be, color etc.
plot_data = {
    'base': {'plot_params': {'label': 'Base', 'color': 'tab:olive'}},
    f'abs_D{dilation}': {'plot_params': {'label': 'Absolute flow', 'color': 'tab:green'}},
    f'avg_D{dilation}': {'plot_params': {'label':  'Average flow', 'color': 'tab:pink'}},
    f'cnn_D{dilation}': {'plot_params': {'label': 'CNN learning', 'color': 'tab:blue'}},
    # 'base_no-Z': {'plot_params': {'label': 'Base no-Z coordinate', 'color': 'tab:olive'}},
    # 'cnn_D3_no-z': {'plot_params': {'label': 'CNN no-Z coordinate', 'color': 'tab:red'}},
}

for extension, data in plot_data.items():
    filename = osp.join(results_root, f"{dataset}_{evaluation}_{extension}.pt")
    if os.path.exists(filename):
        data['results'] = torch.load(filename, map_location='cpu')['results']
        plot_data[extension] = data


# Get the epochs as the x-axis
x = plot_data[list(plot_data.keys())[0]]['results']['epoch']

# Evaluation string and Dataset string dictionaries for plot titles
eval_str_dict = {'CS': 'Cross Subject', 'CV': 'Cross View',
                 'CSub': 'Cross Subject', 'CSet': 'Cross Setting',
                 '1': 'Evaluation 1', '2': 'Evaluation 2', '3': 'Evaluation 3'}
dataset_str_dict = {
    'ntu': 'NTU RGB+D',
    'ntu120': 'NTU RGB+D 120',
    'ucf101': 'UCF-101'
}

# Figure titles!
fig_title = f'{dataset_str_dict[dataset]} - {eval_str_dict[evaluation]}'
print(
    f'Plotting results for {dataset_str_dict[dataset]} - {eval_str_dict[evaluation]} evaluation')

# ------------------------------
#   Loss comparison graphing
# ------------------------------


def plot_loss_comp(
        fig_title: str, x, save_root, plot_data
):
    """
    Plot the loss comparison graph for different models.
    Args:
        fig_title (str): Title of the figure.
        x (array): Epochs.
        plot_data: (dict) containing plotting parameters and plotting data.
            {"extension": {"plot_params": {...}, "results": {...}}}
    """
    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 7))
    fig.suptitle(f'Loss Comparison - {fig_title}', fontsize=30)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Loss', fontsize=20)

    # Plot class loss
    ax1.set_title('Classification Loss', fontsize=25)
    ax2.set_title('Feature Loss', fontsize=25)
    ax3.set_title('Reconstruction Loss', fontsize=25)

    for axis, loss_type in {ax1: 'cls', ax2: 'feature', ax3: 'recon'}.items():
        for data in plot_data.values():
            # data = {'plot_params': {...}, 'results': {...}}
            try:
                axis.plot(x, data['results'][f'train_{loss_type}_loss'],
                          **data['plot_params'])
            except TypeError:
                print(
                    f"Failed to plot {data['plot_params']['label']} {loss_type} loss")

    ax1.legend()
    plt.tight_layout()
    os.makedirs(osp.join(save_root, 'loss_comparison'), exist_ok=True)
    plt.savefig(osp.join(save_root, 'loss_comparison/Loss_comparison.png'))

# ------------------------------
#   Accuracy comparison graphing
# ------------------------------

# Convert results to percentage


def to_percentage(result):
    return [r*100 for r in result]


def plot_accuracy(
        fig_title: str, x, save_root, plot_data
):
    """
    Plot the accuracy comparison graph for different models.
    Args:
        fig_title (str): Title of the figure.
        x (array): Epochs.
        save_root (str): Path to save the plot.
        plot_data: (dict) containing plotting parameters and plotting data.
            {"extension": {"plot_params": {...}, "results": {...}}}
    """
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Accuracy Comparison - {fig_title}', fontsize=30)
    for ax in (ax1, ax2):
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylim(0, 100)  # 0-100% accuracy
        ax.set_xlim(0, 70)  # 0-70 epochs
        ax.axvline(x=50, color='black', linestyle='--', alpha=0.2)  # Epoch 50
        ax.axvline(x=60, color='black', linestyle='--', alpha=0.2)  # Epoch 60
        # ax.set_xscale('log')
    ax1.set_ylabel('Classification accuracy %', fontsize=20)

    for data in plot_data.values():
        try:
            # Plot train and test AUC
            ax1.plot(x, to_percentage(data['results']['train_AUC']),
                     **data['plot_params'])
            ax2.plot(x, to_percentage(data['results']['test_AUC']),
                     **data['plot_params'])
            print(f"\t\t{data['plot_params']['label']}:  "
            f"{to_percentage(data['results']['test_ACC'][-1])[-1]:.2f}%")
        except TypeError:
            print(f"Failed to plot {data['plot_params']['label']}")

    # Set titles and legends
    ax1.set_title('Train accuracy', fontsize=25)
    ax2.set_title('Test accuracy', fontsize=25)
    ax1.legend(loc=4, prop={'size': 20})

    os.makedirs(osp.join(save_root, 'accuracy'), exist_ok=True)
    plt.savefig(osp.join(save_root, 'accuracy/Acc_comparison_simple.png'))


# ------------------------------
#   Classification accuracy for observation ratios
# ------------------------------

def plot_obs_acc(
        fig_title: str, save_root, plot_data
):
    """
    Plot the accuray for the observation ratios
    Network provides a classification per-frame (64 frame input)
    Args:
        fig_title (str): Title of the figure.
        x (array): Epochs.
        plot_data: (dict) containing plotting parameters and plotting data.
            {"extension": {"plot_params": {...}, "results": {...}}}
    """
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
    fig.suptitle(f'Observation accuracy - {fig_title}', fontsize=30)
    for ax in (ax1, ax2):
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('Observation Ratio', fontsize=20)
    ax1.set_ylabel('Accuracy', fontsize=20)

    percents = ((torch.arange(64)+1)*(100/64)).int()

    # Plot class loss
    ax1.set_title('Train Accuracy', fontsize=25)
    ax2.set_title('Test Accuracy', fontsize=25)

    for data in plot_data.values():
        try:
            ax1.plot(percents, torch.tensor(data['results']['train_ACC'][-1]),
                     **data['plot_params'])
            ax2.plot(percents, torch.tensor(data['results']['test_ACC'][-1]),
                     **data['plot_params'])
        except TypeError:
            print(f"Failed to plot {data['plot_params']['label']}")

    ax1.legend(loc='lower right', fontsize='large')
    plt.tight_layout()
    os.makedirs(osp.join(save_root, 'accuracy'), exist_ok=True)
    plt.savefig(osp.join(save_root, 'accuracy/Observation_Acc.png'))


# ------------------------------
#   Confusion matrix
# ------------------------------

def plot_confusion_matrix(
        fig_title: str, save_root, plot_data
):
    for extension, data in plot_data.items():
        filename = osp.join(
            results_root.replace('train', 'eval'),
            f"{dataset}_{evaluation}_{extension}.pt"
        )
        if os.path.exists(filename):
            data['results'] = torch.load(
                filename, map_location='cpu')
            plot_data[extension] = data

    os.makedirs(osp.join(save_root, 'confusion'), exist_ok=True)
    for model_type, data in plot_data.items():
        try:
            # data = {'plot_params': {...}, 'results': {...}}
            y_true = data['results']['truth'][0].mode()[0].int()
            y_pred = data['results']['pred'][0].mode()[0].int()
            fig, ax = plt.subplots(figsize=(25, 20))
            # cm = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred, ax=ax
            )
            ax.set_title(f"Confusion Matrix {fig_title}",
                         fontsize=25)
            ax.set_xlabel("Predicted label", fontsize=20)
            ax.set_ylabel("True label", fontsize=20)
            plt.savefig(
                osp.join(
                    save_root,
                    f"confusion/confusion_matrix_{model_type}.png")
            )
        except TypeError as e:
            print(
                f"Failed to plot {data['plot_params']['label']} confusion matrix"
            )
            print(e)


# ------------------------------
#   Class prediction frequency
# ------------------------------

def plot_class_predictions(
        fig_title: str, save_root, plot_data
):
    for extension, data in plot_data.items():
        filename = osp.join(
            results_root.replace('train', 'eval'),
            f"{dataset}_{evaluation}_{extension}.pt"
        )
        if os.path.exists(filename):
            data['results'] = torch.load(
                filename, map_location='cpu')
            plot_data[extension] = data

    meta_categories = {
        'Object interaction': [0, 1, 2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24],
        'Dynamic body movements': [7, 8, 9],
        'Gestures': [21, 22, 23, 25, 26, 30, 33, 34, 35, 36, 37, 38, 39, 53],
        'Computer/device interaction': [27, 28, 29, 31, 32],
        'Throwing/tossing object': [4, 6],
        'Two-person violent': [49, 50, 51],
        'Two-person friendly': [52, 54, 57],
        'Two-person neutral': [55, 56, 58, 59],
        'Illness/medical': [40, 41, 42, 43, 44, 45, 46, 47, 48]
    }

    os.makedirs(osp.join(save_root, 'class_predictions'), exist_ok=True)
    for model_type, data in plot_data.items():
        try:
            # data = {'plot_params': {...}, 'results': {...}}
            pred_topk = torch.tensor([
                torch.topk(
                    torch.bincount(sample.int()),
                    k=1
                ).indices[0] for sample in data['results']['pred'][0]
            ])
            pred_topk = torch.bincount(pred_topk, minlength=60)
            
            fig, ax = plt.subplots(figsize=(40, 20))

            # Ordering groups according to meta_categories above
            reordered_values = []
            bar_colors = []
            group_labels = []
            group_positions = []
            color_map = plt.cm.get_cmap('tab10')

            x=0
            for i, (category, class_idxs) in enumerate(meta_categories.items()):
                group_values = pred_topk[class_idxs]
                reordered_values.extend(group_values)
                bar_colors.extend([color_map(i)] * len(class_idxs))
                group_labels.append(category)
                group_positions.append(x+len(class_idxs)/2)
                x += len(class_idxs)

            x_ticks = torch.arange(len(reordered_values))
            ax.bar(x_ticks, reordered_values, color=bar_colors)
            ax.tick_params(axis='both', labelsize=20)
            # TODO: Make this median line dynamic for each dataset!
            ax.plot(torch.linspace(0, 60, 60), torch.ones(60)*275, '--',
                    color='red', label="Median class distribution")
            ax.set_title(f"Predicted class label frequency {fig_title}",
                         fontsize=25)
            ax.set_xlabel("Predicted label", fontsize=20)
            ax.set_ylabel("True label", fontsize=20)
            ax.legend(loc='lower right', fontsize='large')
            plt.savefig(
                osp.join(
                    save_root,
                    f"class_predictions/class_frequency_{model_type}.png")
            )
        except TypeError as e:
            print(
                f"Failed to plot {data['plot_params']['label']} confusion matrix"
            )
            print(e)


if __name__ == '__main__':
    if loss:
        # Plot the loss comparison
        print("\tPlotting loss comparison")
        plot_loss_comp(
            fig_title, x, save_root, plot_data
        )

    if acc:
        # Plot the accuracy comparison
        print("\tPlotting accuracy comparison")
        plot_accuracy(
            fig_title, x, save_root, plot_data
        )

    if obs:
        # Plot the accuracy comparison across different observation ratios
        print("\tPlotting accuracy comparison across observation ratios")
        plot_obs_acc(
            fig_title, save_root, plot_data
        )

    if confusion:
        # Plot individual confusion matrices for each model
        print("\tPlotting individual confusion matrices for each model type")
        plot_confusion_matrix(
            fig_title, save_root, plot_data
        )

    if cls_pred:
        print("\tPlotting predicted classes from each model type")
        plot_class_predictions(
            fig_title, save_root, plot_data
        )

    quit()
