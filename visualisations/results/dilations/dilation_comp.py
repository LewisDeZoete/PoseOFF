import os
import os.path as osp

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# ----------------------------------------------------
# USAGE:
# - edit the `dataset` and `evaluation` variables below
# - Set values of plot_* as needed, if True, that plot will be generated
# - edit the `plot_data` for specific extensions (keys) and plot params
#
# - Default save root:
#      results/visualisations/dilation/{dataset}/{evaluation}
# - Creates graphs:
#      {save_root}/Epoch_dilation_acc_comparison.png
#      {save_root}/Bar_dilation_acc_comparison.png
#
# - To run:
#      python ./results/visualisations/dilation_comp.py

# Which graphs are we making?
dataset = "ucf101" # ntu, ntu120, ucf101
evaluation = "1" # CS, CSub, 1/2/3
flow_type = "RAFT" # RAFT/LK/NF
dilation_values = [1, 2, 3] if dataset == "ucf101" else [1,2,3,4,5]

# In case you didn't want to
epoch_dilation_comp = True
bar_dilation_comp = True

# ----------------------------------------------------

# Create a directory, ignoring if it already exists
save_root = 'visualisations/results/dilations'
os.makedirs(save_root, exist_ok=True)

# Define plotting parameters, the extension is the name of the checkpoint
# e.g.   extension = 'base_NO_MASK'
#        filename = f"{results_root}/nturgbd_CV_{extension}.pt
# Change plot_params such as what you want the plot label to be, color etc.
colors = {1:'tab:orange', 2:'tab:green', 3:'tab:pink', 4:'tab:red', 5:'tab:blue'}

# Define models, datasets and evaluations
models = ["msg3d", "stgcn2", "infogcn2"]
evaluations = {
    "ntu": ["CS", "CV"],
    "ntu120": ["CSet", "CSub"],
    "ucf101": ["1", "2", "3"]
}
class_numbers = {"ntu": 60, "ntu120": 120, "ucf101": 101}

# Evaluation string and Dataset string dictionaries for plot titles
eval_str_dict = {'CS': 'Cross Subject', 'CV': 'Cross View',
                 'CSub': 'Cross Subject', 'CSet': 'Cross Setting',
                 '1': 'Evaluation 1', '2': 'Evaluation 2', '3': 'Evaluation 3'}
dataset_str_dict = {
    'ntu': 'NTU RGB+D',
    'ntu120': 'NTU RGB+D 120',
    'ucf101': 'UCF-101'
}


# -------------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------------
def load_dilation_val_results(dataset, evaluation, models,
                              flow_type="RAFT", dilation_values=[1,2,3,4,5]):
    '''Loads the results dictionaries for dataset and evaluation at different dilation values

    Args:
        dataset (str): Dataset to load results for.
        evaluation (str): Evaluation for given dataset (e.g. "CV" for ntu).
        models (list): List containing strings corresponding to model names.
        flow_type (str): What type of motion estimation method to load results for. Defaults to "RAFT".
        dilation_values (list: int): List of dilation values to load. Defaults to [1,2,3,4,5].
    '''

    results_filename = "{model}_{dataset}_{evaluation}_cnn_{flow_type}_D{dilation}.pt"
    dilation_results = {} # Dictionary to be filled with results values

    # Iterate over each model's results
    for model in models:
        dilation_results[model] = {}
        for dilation in dilation_values:
            # Define the path to the .pt file for each model/dilation values
            res_dict_path = osp.join(
                "results",
                model,
                dataset,
                evaluation,
                "train",
                results_filename.format(
                    model=model,
                    dataset=dataset,
                    evaluation=evaluation,
                    flow_type=flow_type,
                    dilation=dilation
                )
            )

            # Check that the results dictionary exists...
            if osp.exists(res_dict_path):
                print(f"loading result for {model}-{flow_type}-D{dilation}")
                checkpoint = torch.load(res_dict_path, map_location="cpu")
                res = checkpoint['results']
                try:
                    # print(f"\t{torch.tensor(res['test_ACC']).max()*100:.2f}")
                    dilation_results[model][dilation] = torch.tensor(res['test_ACC']).max().item()
                except RuntimeError as e:
                    print(f"\tCouldn't get result for {model}-{flow_type}-D{dilation}")
                    print(f"\t{e}")
                    print(f"\tModel hasn't finished training (results['test_ACC']=[])")
                    dilation_results[model][dilation] = 0
            else:
                print(f"DOES NOT EXIST: {model} - D{dilation}")
    return dilation_results


# -------------------------------------------------------------
# GRAPHING FUNCTIONS
# -------------------------------------------------------------
def make_dilation_bars_joined(
        dilation_results,
        dataset,
        evaluation,
        save_root,
        flow_type,
        dilation_values=[1,2,3,4,5]
):
    """Plot the loss comparison graph for different models ON THE SAME AXES.
    Args:
        dilation_results: (dict) containing best achieved accuracy for dilations.
            {"model_name": {
                1: 0.85,
                2: 0.86,
                ...
                }
            }
        dataset (str): Dataset to be plotted.
        evaluation (str): Evaluation for given dataset (e.g. "CV" for ntu).
        flow_type (str): what type of flow estimation are we comparing?
        save_root (str): Root path to where to save figures (saves to save_root/dataset/evaluation)
    """
    # Create figure and axes
    fig, ax = plt.subplots(layout='constrained')
    fig.suptitle(f'Accuracy dilation comparison - {dataset} - {evaluation}', size=15)
    ax.tick_params(axis='both', which='major')

    width = 0.5
    multiplier = 0
    bins = np.arange(1, len(dilation_values)*2, 2)+1

    for model_name, dilation_data in dilation_results.items():
        # dilation_results = {1: 0.85, 2: 0.92, ...}
        offset = width * multiplier
        measurements = [val for val in dilation_data.values()]

        # for dilation_value, data in dilation_data.items():
        #     try:
        #         measurements.append(np.array(data['results']['test_ACC']).max())
        #     except KeyError:
        #         measurements.append(0)

        # Draw the bars (with the offset)
        rects = ax.bar(
            bins + offset, 100*(np.array(measurements).round(4)), width, label=model_name
            )
        # Put a label above each bar showing highest accuracy
        ax.bar_label(rects, np.round(100*(np.array(measurements)), 1), padding=3, fontsize="x-small")
        multiplier += 1

    ax.set_ylabel('Highest test accuracy (%)')
    ax.set_ylim(80, 95)
    ax.set_xlabel('Flow sampling dilation value')
    ax.set_xticks(bins+width, (bins/2).astype(int))
    ax.legend(loc='upper left', ncols=3)

    os.makedirs(osp.join(save_root, 'output'), exist_ok=True)
    plt.savefig(osp.join(save_root, f'output/Bar_dilation_acc_comparison_jointed-{dataset}-{evaluation}-{flow_type}.png'))


def make_dilation_bars_individual(
        dilation_results,
        dataset,
        evaluation,
        save_root,
        flow_type,
        dilation_values=[1,2,3,4,5]
):
    """Plot the loss comparison graph for different models, each with their own set of axes.
    Args:
        dilation_results: (dict) containing best achieved accuracy for dilations.
            {"model_name": {
                1: 0.85,
                2: 0.86,
                ...
                }
            }
        dataset (str): Dataset to be plotted.
        evaluation (str): Evaluation for given dataset (e.g. "CV" for ntu).
        flow_type (str): what type of flow estimation are we comparing?
        save_root (str): Root path to where to save figures (saves to save_root/dataset/evaluation)
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(16,5), layout='constrained')
    fig.suptitle(f'Accuracy dilation comparison - {dataset} - {evaluation}', size=30)

    for ax, (model_name, dilation_data) in zip(axs, dilation_results.items()):
        # Create dilation results bar chart for single model on single dataset
        model_results = 100*np.array(list(dilation_data.values())).round(4)
        label_string = f"Best perfer: dilation {max(dilation_data, key=dilation_data.get)} " \
            f"({np.round(model_results.max(), 1)}%)"
        bars = ax.bar(
            dilation_values,
            model_results,
            width=0.5,
            label=label_string
        )
        ax.bar_label(
            bars,
            np.round(model_results, 1),
            padding=3,
            fontsize="x-small"
        )
        ax.set_ylim(model_results.round().min()-3, model_results.round().max()+1)
        ax.set_title(model_name)
        ax.legend(loc='upper left', ncols=3)

    axs[0].set_ylabel('Highest test accuracy (%)')
    # ax.set_ylim(80, 95)
    axs[1].set_xlabel('Flow sampling dilation value')
    # ax.set_xticks(bins+width, (bins/2).astype(int))

    os.makedirs(osp.join(save_root, 'output'), exist_ok=True)
    outname = osp.join(save_root, "output", f"Bar_dilation_acc_comparison_individual-{dataset}-{evaluation}-{flow_type}.png")
    print(f"Saving to: {outname}")
    plt.savefig(outname)

if __name__=="__main__":
    dilation_results = load_dilation_val_results(
        dataset,
        evaluation,
        models,
        flow_type=flow_type,
        dilation_values=dilation_values
    )
    # make_dilation_bars_joined(
    #         dilation_results,
    #         dataset,
    #         evaluation,
    #         save_root,
    #         flow_type,
    #         dilation_values=[1,2,3,4,5]
    # )
    make_dilation_bars_individual(
            dilation_results,
            dataset,
            evaluation,
            save_root,
            flow_type,
            dilation_values=dilation_values
    )
