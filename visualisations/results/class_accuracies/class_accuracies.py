#!/usr/bin/env python3
import os
import os.path as osp
import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

# -----------------------------------------------------------------------------------
# CHANGE PLOTTING VARIABLES
# -----------------------------------------------------------------------------------
# How to arrange and group classes in graphs
class_split_type = "custom" # default/custom
# Whether to plot the individual model graphs
individual_models = False # True/False
draw_confusion_matrices = False # True/False
# -----------------------------------------------------------------------------------

# Make sure inputs are right...
assert class_split_type in ["custom", "default"]
assert type(individual_models) == bool

# Defining variables for data capture
models = ["msg3d", "stgcn2", "infogcn2"]
datasets = ["ntu", "ntu120", "ucf101"]
evaluations = {
    "ntu": ["CS", "CV"],
    "ntu120": ["CSet", "CSub"],
    "ucf101": ["1", "2", "3"]
}
class_numbers = {"ntu": 60, "ntu120": 120, "ucf101": 101}

# Define the folder structure for outputs
output_root = "./results/visualisations/class_accuracies"
all_models_dir = osp.join(output_root, "all_models")
individual_models_dir = osp.join(output_root, "individual_models")

# Create required directories
os.makedirs(all_models_dir, exist_ok=True)
for model in models:
    os.makedirs(osp.join(individual_models_dir, model), exist_ok=True)

# -------------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------------
def load_truth_pred_results(dataset, model, evaluation):
    '''Simply loads the results for a given dataset-model-evaluation'''
    res_path_poseoff = f"./results/{model}/{dataset}/{evaluation}/eval/{model}_{dataset}_{evaluation}_cnn_D{1 if dataset=='ucf101' else 3}_obs1.0.pt"
    res_path_base = f"./results/{model}/{dataset}/{evaluation}/eval/{model}_{dataset}_{evaluation}_base_obs1.0.pt"

    res_poseoff = torch.load(res_path_poseoff, map_location='cpu')
    res_base = torch.load(res_path_base, map_location='cpu')

    # Infogcn++ returns one class prediction for every frame (64 total)
    if model == "infogcn2":
        eval_truth_poseoff = res_poseoff['truth'][0].mode(-1).values
        eval_pred_poseoff = res_poseoff['pred'][0].mode(-1).values
        eval_truth_base = res_base['truth'][0].mode(-1).values
        eval_pred_base = res_base['pred'][0].mode(-1).values
    else:
        eval_truth_poseoff = res_poseoff['truth'][0]
        eval_pred_poseoff = res_poseoff['pred'][0]
        eval_truth_base = res_base['truth'][0]
        eval_pred_base = res_base['pred'][0]

    results = {
        "poseoff": {"truth": eval_truth_poseoff, "pred": eval_pred_poseoff},
        "base": {"truth": eval_truth_base, "pred": eval_pred_base}
    }
    return results


def get_truth_pred(dataset, models, evaluations):
    '''Loads all the ground truth and predicted class labels for each model for a dataset'''
    truth_poseoff, pred_poseoff, truth_base, pred_base = (torch.empty(0) for i in range(4))
    # {model1: {"poseoff": {"truth": torch.tensor([0,1,...]), "pred": torch.tensor([1,5,...])}}}
    dataset_results = {
        evaluation: {
            model: load_truth_pred_results(dataset, model, evaluation)
            for model in models
        }
        for evaluation in evaluations[dataset]
    }

    return dataset_results


def concat_results(results, level):
    """
    Concatenates tensors across all levels above the given level.

    Structure: {evaluation: {model: {model_type: {truth/pred: tensor}}}}

    level='model'      → {model:      {model_type: {truth/pred: tensor}}}
    level='model_type' → {model_type: {truth/pred: tensor}}
    """
    levels = ['evaluation', 'model', 'model_type']
    assert level in levels, f"level must be one of {levels}"
    target_depth = levels.index(level)

    def collect(d, current_depth):
        """Flatten all keys above target_depth, preserve structure from target_depth down."""
        if current_depth == target_depth:
            return d  # preserve everything from here down as-is

        # Merge across the current level's keys
        merged = {}
        for child in d.values():
            for k, v in child.items():
                if k not in merged:
                    merged[k] = v
                else:
                    merged[k] = merge_dicts(merged[k], v)
        return collect(merged, current_depth + 1)

    def merge_dicts(a, b):
        """Recursively merge two dicts, concatenating tensors at the leaves."""
        if isinstance(a, torch.Tensor):
            return torch.cat([a, b])
        return {k: merge_dicts(a[k], b[k]) for k in a}

    return collect(results, 0)


def get_per_class_scores(
        dataset,
        truth_poseoff,
        pred_poseoff,
        truth_base,
        pred_base,
        draw_confusion_matrices=False
):
    '''Returns two dictionaries, with keys 'recall', 'precision' and 'f1'.
    First dictionary is poseoff, second is base'''
    confmat_poseoff = confusion_matrix(
        truth_poseoff.numpy(),
        pred_poseoff.numpy(),
        labels=list(range(class_numbers[dataset]))
    )
    confmat_base = confusion_matrix(
        truth_base.numpy(),
        pred_base.numpy(),
        labels=list(range(class_numbers[dataset]))
    )

    per_class_recall_poseoff = confmat_poseoff.diagonal() / confmat_poseoff.sum(axis=1)
    per_class_precision_poseoff = confmat_poseoff.diagonal() / confmat_poseoff.sum(axis=0)
    f1_poseoff = 2 * (per_class_precision_poseoff * per_class_recall_poseoff) / (per_class_precision_poseoff + per_class_recall_poseoff + 1e-8)
    poseoff_scores = {"recall": per_class_recall_poseoff, "precision": per_class_precision_poseoff, "f1": f1_poseoff}

    per_class_recall_base = confmat_base.diagonal() / confmat_base.sum(axis=1)
    per_class_precision_base = confmat_base.diagonal() / confmat_base.sum(axis=0)
    f1_base = 2 * (per_class_precision_base * per_class_recall_base) / (per_class_precision_base + per_class_recall_base + 1e-8)
    base_scores = {"recall": per_class_recall_base, "precision": per_class_precision_base, "f1": f1_base}

    return poseoff_scores, base_scores, confmat_poseoff, confmat_base


def write_scores(recall_poseoff, recall_base, dataset, class_numbers, all_models=False, model="", top_k=5):
    '''Write the scores, including best and worst K classes and wilcoxon score to a file'''

    save_path = all_models_dir if all_models else individual_models_dir
    fname = osp.join(
        save_path,
        model,
        f"{'ALL_MODELS' if all_models else model}-{dataset}.txt"
    )

    per_class_recall_improvement = (recall_poseoff - recall_base)
    recall_changes = {
        'improvement': {
            'class_number': np.where(per_class_recall_improvement > 0),
            'amount': [i for i in per_class_recall_improvement if i>0],
            },
        'no_change': {
            'class_number': np.where(per_class_recall_improvement == 0),
            'amount': [i for i in per_class_recall_improvement if i==0],
            },
        'failures': {
            'class_number': np.where(per_class_recall_improvement < 0),
            'amount': [i for i in per_class_recall_improvement if i<0],
            },
    }
    # Wilcoxon signed-rank test
    stat, p_value = wilcoxon(recall_poseoff, recall_base, alternative="greater")

    # Best and worst K classes
    worst_k = np.argsort(per_class_recall_improvement)[:top_k]
    best_k = np.argsort(per_class_recall_improvement)[-top_k:]

    with open(fname, 'w') as f:
        f.write(f"Improved {len(recall_changes['improvement']['amount'])} classes\n")
        f.write(f"No change in {len(recall_changes['no_change']['amount'])} classes\n")
        f.write(f"Accuracy losses in {len(recall_changes['failures']['amount'])} classes\n")

        f.write(f"Percentage of classes improved: {len(recall_changes['improvement']['amount'])/class_numbers[dataset]*100:.2f}%\n")

        f.write(f"Median gain: {np.median(per_class_recall_improvement)}\n")
        f.write(f"Interquartile range (IQR): {np.percentile(per_class_recall_improvement, [25, 75])}\n")


        f.write("BEST:\n")
        for c in np.flip(best_k):
            f.write(f"\tClass {class_names[c]}: recall={per_class_recall_improvement[c]*100:.2f}%\n")
        f.write("WORST:\n")
        for c in worst_k:
            f.write(f"\tClass {class_names[c]}: recall={per_class_recall_improvement[c]*100:.2f}%\n")

        f.write(f"Wilcoxon statistic: {stat}\n\tP-value: {p_value}\n")

        f.write("\nALL CLASSES:\n")
        for class_no, change in enumerate(per_class_recall_improvement):
            f.write(f"\t{class_names[class_no]}: {change*100:.2f}%\n")


def get_class_categories(dataset, split_name="default"):
    all_splits = {
        'ntu': {
            'default': {
                'daily_actions': {
                    'numbers': list(range(40)),
                    'color': '#4C72B0'
                },
                'medical_conditions': {
                    'numbers': list(range(40,49)),
                    'color': '#C44E52'
                },
                'two_person_interactions': {
                    'numbers': list(range(49,60)),
                    'color': '#CCB974'
                }
            },
            'custom': {
                'object_interactions': {
                    'numbers': [0,1,2,3,5,10,11,12,13,14,15,16,17,18,19,20,24],
                    'color': '#4C72B0'
                },
                'dynamic_body_movements': {
                    'numbers': [7,8,9, 25, 26],
                    'color': '#55A868'
                },
                'gestures': {
                    'numbers': [21,22,23,30,33,34,35,36,37,38,39,53],
                    'color': '#C44E52'
                },
                'computer_device_interactions': {
                    'numbers': [27,28,29,31,32],
                    'color': '#8172B2'
                },
                'throwing_objects': {
                    'numbers': [4,6],
                    'color': '#CCB974'
                },
                'two_person_violent': {
                    'numbers': [49,50,51],
                    'color': '#E15759'
                },
                'two_person_friendly': {
                    'numbers': [52,54,57],
                    'color': '#64B5CD'
                },
                'two_person_neutral': {
                    'numbers': [55,56,58,59],
                    'color': '#8C8C8C'
                },
                'medical': {
                    'numbers': [40,41,42,43,44,45,46,47,48],
                    'color': '#B07AA1'
                }
            }
        },
        'ntu120': {
            'default': {
                'daily_actions': {
                    'numbers': list(range(40))+list(range(60,102)),
                    'color': '#4C72B0'
                },
                'medical_conditions': {
                    'numbers': list(range(40,49))+list(range(102,105)),
                    'color': '#C44E52'
                },
                'two_person_interactions': {
                    'numbers': list(range(49,60))+list(range(105,120)),
                    'color': '#CCB974'
                }
            },
            'custom': {
                'object_interactions': {
                    'numbers': [0, 1, 2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 72, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
                    'color': '#4C72B0'
                },
                'dynamic_body_movements':{
                    'numbers': [7, 8, 9, 25, 26, 79, 98, 99, 100, 101],
                    'color': '#55A868'
                },
                'gestures': {
                    'numbers': [21, 22, 23, 30, 33, 34, 35, 36, 37, 38, 39, 53, 66, 67, 68, 69, 70, 71, 92, 94, 95, 96, 97],
                    'color': '#C44E52'
                },
                'computer_device_interactions': {
                    'numbers': [27, 28, 29, 31, 32, 60, 61, 114],
                    'color': '#8172B2'
                },
                'throwing_objects': {
                    'numbers': [4, 6, 62, 63, 64, 65, 80, 93],
                    'color': '#CCB974'
                },
                'two_person_violent': {
                    'numbers': [49, 50, 51, 105, 106, 107, 108, 109, 110],
                    'color': '#E15759'
                },
                'two_person_friendly': {
                    'numbers': [52, 54, 57, 111, 112, 118],
                    'color': '#64B5CD'
                },
                'two_person_neutral': {
                    'numbers': [55, 56, 58, 59, 113, 115, 116, 117, 119],
                    'color': '#8C8C8C'
                },
                'medical': {
                    'numbers': [40, 41, 42, 43, 44, 45, 46, 47, 48, 102, 103, 104],
                    'color': '#B07AA1'
                }
            }
        },
        'ucf101': {
            # Brutally, I'm coping the class categories from here:
            # https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101_CRCV-TR-12-01.pdf
            'default': {
                'human_object_interation': {
                    'numbers': [0,1,12,19,24,35,42,45,46,49,53,54,55,57,77,79,83,94,99,100],
                    'color': '#4C72B0'
                },
                'body_motion_only': {
                    'numbers': [3,13,14,36,37,47,51,69,71,73,74,88,90,93,97,98,],
                    'color': '#55A868'
                },
                'human_human_interactions': {
                    'numbers': [5,33,38,52,76],
                    'color': '#C44E52'
                },
                'playing_musical_instruments': {
                    'numbers': [26,58,59,60,61,62,63,64,65,66],
                    'color': '#CCB974'
                },
                'sports': {
                    'numbers': [2,4,6,7,8,9,10,11,15,16,17,18,20,21,22,23,25,27,28,29,30,31,32,34,39,40,41,43,44,48,50,56,67,68,70,72,75,78,80,81,82,84,85,86,87,89,91,92,95,96],
                    'color': '#8C8C8C'
                }
            },
            'custom': {
                'human_object_interation': {
                    'numbers': [0,1,12,19,24,35,42,45,46,49,53,54,55,57,77,79,83,94,99,100],
                    'color': '#4C72B0'
                },
                'body_motion_only': {
                    'numbers': [3,13,14,36,37,47,51,69,71,73,74,88,90,93,97,98,],
                    'color': '#55A868'
                },
                'human_human_interactions': {
                    'numbers': [5,33,38,52,76],
                    'color': '#C44E52'
                },
                'playing_musical_instruments': {
                    'numbers': [26,58,59,60,61,62,63,64,65,66],
                    'color': '#CCB974'
                },
                'sports': {
                    'numbers': [2,4,6,7,8,9,10,11,15,16,17,18,20,21,22,23,25,27,28,29,30,31,32,34,39,40,41,43,44,48,50,56,67,68,70,72,75,78,80,81,82,84,85,86,87,89,91,92,95,96],
                    'color': '#8C8C8C'
                }
            }
        }
    }
    return all_splits[dataset][split_name]


# -------------------------------------------------------------
# GRAPHING FUNCTIONS
# -------------------------------------------------------------
def make_confmat_graphs(
        dataset,
        confmat_poseoff,
        confmat_base,
        all_models=False,
        model=""
):
    save_root = all_models_dir if all_models else osp.join(individual_models_dir, model)
    save_path = osp.join(save_root, "confusion_matrices")
    os.makedirs(save_path, exist_ok=True)
    for model_type, confmat in zip(("PoseOFF", "Base"), (confmat_poseoff, confmat_base)):
        mask = np.eye(class_numbers[dataset], dtype=bool)
        fig, ax = plt.subplots(figsize=(40,40))
        sns.heatmap(
            confmat,
            ax=ax,
            mask=mask,
            cmap="Blues",
            annot=False,
            fmt="",
            linewidths=0,
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_xlabel("Predicted Label", fontsize=16)
        ax.set_ylabel("True Label", fontsize=16)
        ax.set_title(f"Confusion Matrix - {dataset} - {model_type}", fontsize=20)

        plt.tight_layout()
        plt.savefig(osp.join(save_path, f"{model_type}-{dataset}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()


def make_graph_standard_bars(
        poseoff_scores,
        base_scores,
        class_split,
        all_models=False,
        model="",
        class_split_type='default'
):
    save_path = all_models_dir if all_models else individual_models_dir
    fname = osp.join(
        save_path,
        model,
        f"STANDARD_bars-{dataset}-{class_split_type}.png"
    )

    # calculate per-class recall improvements
    per_class_recall_improvement = poseoff_scores['recall'] - base_scores['recall']

    ordered_indices = []
    ordered_categories = []
    class_colors = {}

    for cat, vals in class_split.items():
        idxs=vals['numbers']
        ordered_indices.extend(sorted(idxs))
        ordered_categories.extend([cat] * len(idxs))
        class_colors[cat]=vals['color']

    # Reorder recall improvements
    values = per_class_recall_improvement[ordered_indices]

    # Assign colors in grouped order
    colors = [class_colors[cat] for cat in ordered_categories]

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(21, 8))

    x = np.arange(len(values))
    ax.bar(x, values*100, color=colors)
    ax.set_xlim(-1, len(values))
    ax.set_xticklabels([])

    # Zero line
    ax.axhline(0, color='black', linewidth=1)

    # Vertical separators between category groups
    group_sizes = [len(class_split[cat]['numbers']) for cat in class_split]
    boundaries = np.cumsum(group_sizes)[:-1]

    for b in boundaries:
        ax.axvline(b - 0.5, color='gray', linestyle='--', alpha=0.4)

    # ------------------------------------------------------------------
    # 4. Legend
    # ------------------------------------------------------------------

    legend_elements = [
        Patch(facecolor=color, label=cat.replace('_', ' ').capitalize())
        for cat, color in class_colors.items()
        if cat in class_split
    ]

    ax.legend(
        handles=legend_elements,
        title="Class Category",
        title_fontsize=20,
        # bbox_to_anchor=(1.02, 1),
        loc='upper right',
        ncols=2 if len(legend_elements)<=5 else 3,
        fontsize=20
    )

    ax.set_ylabel("Recall Improvement (%)", fontsize=35)
    ax.set_xlabel("Class (Grouped by Category)", fontsize=35)

    dataset_names = {'ntu': 'NTU RGB+D', 'ntu120': 'NTU RGB+D120', 'ucf101': 'UCF-101'}
    model_names = {'': 'All models - ', 'msg3d': 'MS-G3D - ', 'stgcn2': 'ST-GCN++ - ', 'infogcn2': 'InfoGCN++ - '}
    # title = f"{model_names[model]}Per-Class Accuracy Changes (PoseOFF-Base) - {dataset_names[dataset]}"
    title = f"Per-Class Accuracy Changes - {dataset_names[dataset]}"
    # plt.title(title, fontsize=30)
    plt.tick_params(labelsize=25, bottom=False)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def make_graph_stacked_bars(
        poseoff_scores,
        base_scores,
        class_split,
        all_models=False,
        model="",
        class_split_type='default'
):
    save_path = all_models_dir if all_models else individual_models_dir
    fname = osp.join(
        save_path,
        model,
        f"STACKED_bars-{dataset}-{class_split_type}.png"
    )

    # calculate per-class recall improvements
    base_recall = base_scores['recall']
    recall_improvement = poseoff_scores['recall'] - base_scores['recall']

    ordered_indices = []
    ordered_categories = []
    class_colors = {}

    for cat, vals in class_split.items():
        idxs=vals['numbers']
        ordered_indices.extend(sorted(idxs))
        ordered_categories.extend([cat] * len(idxs))
        class_colors[cat]=vals['color']

    # Reorder base recall recall
    base_recall = base_scores['recall'][ordered_indices]
    recall_improvement = recall_improvement[ordered_indices]

    # Assign colors in grouped order
    colors = [class_colors[cat] for cat in ordered_categories]

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 5))

    # Set the floor value for the initial bar plots
    bottom = np.zeros(len(base_recall))

    # Create base bar, then graph the improvements at the top of those bars
    x = np.arange(len(base_recall))
    ax.bar(x, base_recall*100, label="Base", color='grey', bottom=bottom)
    bottom += base_recall*100
    colors=['green' if value>=0 else 'red' for value in recall_improvement]
    ax.bar(x, recall_improvement*100, label="PoseOFF difference", color=colors, bottom=bottom)

    ax.set_xlim(-1, len(base_recall))
    ax.set_xticklabels([])

    # Vertical separators between category groups
    group_sizes = [len(class_split[cat]) for cat in class_split]
    boundaries = np.cumsum(group_sizes)[:-1]

    # for b in boundaries:
    #     ax.axvline(b - 0.5, color='gray', linestyle='--', alpha=0.4)

    # ------------------------------------------------------------------
    # 4. Legend
    # ------------------------------------------------------------------

    # legend_elements = [
    #     Patch(facecolor=color, label=cat.replace('_', ' ').capitalize())
    #     for cat, color in class_colors.items()
    #     if cat in class_split
    # ]

    # ax.legend(
    #     handles=legend_elements,
    #     title="Class Category",
    #     bbox_to_anchor=(1.02, 1),
    #     loc='upper left'
    # )

    ax.set_ylabel("Recall (%)")
    ax.set_xlabel("Class (Grouped by Category)")

    dataset_names = {'ntu': 'NTU RGB+D', 'ntu120': 'NTU RGB+D120', 'ucf101': 'UCF-101'}
    model_names = {'': 'All models - ', 'msg3d': 'MS-G3D - ', 'stgcn2': 'ST-GCN++ - ', 'infogcn2': 'InfoGCN++ - '}
    title = f"{model_names[model]}Per-Class Accuracy Changes - {dataset_names[dataset]}"
    plt.title(title)
    plt.tick_params(bottom=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


if __name__ == '__main__':
    # Graph the datasets, both the model performance across ALL models and individual models
    for dataset in datasets:
        # Retrieve the human readable class names file for each dataset
        print(f"Dataset: {dataset.upper()}")
        with open(f"./data/{dataset}/statistics/class_names.yaml", 'r') as file:
            class_names=yaml.safe_load(file)

        # Get the class splits (categorises classes...)
        class_split = get_class_categories(dataset, class_split_type)

        # Get the results for all models for this particular dataset
        raw_dataset_results = get_truth_pred(dataset, models, evaluations)
        dataset_results = concat_results(raw_dataset_results, level="model")

        # Concatenate the results across all models for the all-model comparisons
        dataset_results_ALL = concat_results(raw_dataset_results, level="model_type")

        # Calculate scores (each dictionary contains 'recall', 'precision', 'f1')
        poseoff_scores, base_scores, confmat_poseoff, confmat_base = get_per_class_scores(
            dataset,
            truth_poseoff=dataset_results_ALL["poseoff"]["truth"],
            pred_poseoff=dataset_results_ALL["poseoff"]["pred"],
            truth_base=dataset_results_ALL["base"]["truth"],
            pred_base=dataset_results_ALL["base"]["pred"],
            draw_confusion_matrices=draw_confusion_matrices
        )
        # Write out the scores for a particular dataset/model
        write_scores(
            poseoff_scores['recall'],
            base_scores['recall'],
            dataset,
            class_numbers,
            all_models=True,
            model='',
            top_k=5
        )
        # Create the graphs for the specific dataset, for each model
        make_confmat_graphs(
                dataset,
                confmat_poseoff,
                confmat_base,
                all_models=True,
                model=''
        )
        make_graph_standard_bars(
                poseoff_scores,
                base_scores,
                class_split,
                all_models=True,
                model='',
                class_split_type=class_split_type
        )
        make_graph_stacked_bars(
                poseoff_scores,
                base_scores,
                class_split,
                all_models=True,
                model='',
                class_split_type=class_split_type
        )
        if individual_models:
            for model in models:
                # Calculate scores (each dictionary contains 'recall', 'precision', 'f1')
                poseoff_scores, base_scores, confmat_poseoff, confmat_base = get_per_class_scores(
                    dataset,
                    truth_poseoff=dataset_results[model]["poseoff"]["truth"],
                    pred_poseoff=dataset_results[model]["poseoff"]["pred"],
                    truth_base=dataset_results[model]["base"]["truth"],
                    pred_base=dataset_results[model]["base"]["pred"]
                )

                # Write out the scores for a particular dataset/model
                write_scores(
                    poseoff_scores['recall'],
                    base_scores['recall'],
                    dataset,
                    class_numbers,
                    all_models=False,
                    model=model,
                    top_k=5
                )

                # Create the graphs for the specific dataset, for each model
                make_confmat_graphs(
                    dataset,
                    confmat_poseoff,
                    confmat_base,
                    all_models=False,
                    model=model
                )
                make_graph_standard_bars(
                        poseoff_scores,
                        base_scores,
                        class_split,
                        all_models=False,
                        model=model,
                        class_split_type=class_split_type
                )
                make_graph_stacked_bars(
                        poseoff_scores,
                        base_scores,
                        class_split,
                        all_models=False,
                        model=model,
                        class_split_type=class_split_type
                )
