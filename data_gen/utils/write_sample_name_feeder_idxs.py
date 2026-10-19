#!/usr/bin/env python3

# Create a yaml mapping between the index of the feeders with specific evaluation
# to the actual sample name that corresponds to the RGB video

# Output yaml looks like:
# ntu:
#   CS:
#     train:
#       0: S001C001P001R001A001
#       ...
#
#     test:
#       0: S001C001P002R001A001
#       ...
# ntu120:
#   CSub:
#     train:
#       0: S001C001P001R001A001
#       ...
#
#     test:
#       0: S001C001P002R001A001
#       ...
#
# ucf101:
#   1:
#     train:
#       0: Applying_eye_makeup/ggg555

import os
import os.path as osp
import numpy as np
import yaml
from data_gen.ntu.seq_transformation import get_details
from data_gen.ntu.seq_transformation import get_indices as get_indices_ntu
from data_gen.ucf101.seq_transformation import get_indices as get_indices_ucf

datasets = {
    "ntu": ["CV", "CS"],
    "ntu120": ["CSub", "CSet"],
    "ucf101": ["1", "2", "3"]
}


for dataset, evaluations in datasets.items():
    feeder_idx_to_name = {}
    for evaluation in evaluations:
        feeder_idx_to_name[evaluation] = {"train": {}, "test": {}}
        print(f"{dataset} - {evaluation}")

        # PATHS
        root_path = f'./data/{dataset}'
        stat_path = osp.join(root_path, 'statistics')

        if dataset != "ucf101":
            # Info files and folders
            skes_name_file = osp.join(stat_path, f'ntu_rgbd{120 if dataset == "ntu120" else ""}-available.txt')
            frames_file = osp.join(stat_path, 'frames_cnt.txt')

            # Load data statistics
            frames_cnt = np.loadtxt('./data/ntu/statistics/frames_cnt.txt', dtype=int)
            skes_name = np.loadtxt('./data/ntu/statistics/ntu_rgbd-available.txt', dtype=str)
            if dataset=='ntu120':
                frames_cnt = np.hstack((frames_cnt, np.loadtxt(frames_file, dtype=int)))
                skes_name = np.hstack((skes_name, np.loadtxt(skes_name_file, dtype=str)))
            details = get_details(skes_name, frames_cnt)

            train_indices, test_indices = get_indices_ntu(
                details['Performer'],
                details['Camera'],
                details['Setup'],
                evaluation)
            
            # Append skes_names from the indices to the feeder_idx_to_name dictionary
            feeder_idx_to_name[evaluation]["train"] = [str(skes_name[idx]) for idx in train_indices]
            feeder_idx_to_name[evaluation]["test"] = [str(skes_name[idx]) for idx in test_indices]

        else:
            with open(osp.join(stat_path, 'ucf101_annotations.yaml'), 'r') as f:
                labels = yaml.safe_load(f)

            train_indices, test_indices = get_indices_ucf(labels, evaluation, "./data/ucf101/statistics")

            feeder_idx_to_name[evaluation]["train"] = [list(labels.keys())[idx] for idx in train_indices]
            feeder_idx_to_name[evaluation]["test"] = [list(labels.keys())[idx] for idx in test_indices]


    save_path = osp.join("data", dataset, "statistics", "feeder_idx_to_filename.yaml")
    with open(save_path, "w+") as file:
        yaml.dump(feeder_idx_to_name, file)

    print(f"Data written to: {save_path}")

