#!/usr/bin/env python3
# This file is for graphing 3D skeletons with the joint sizings weighted by the saliency maps extracted using
# skeleton_xai.py

import os
import os.path as osp
from config import ArgClass
from feeders.ntu_rgb_d import Feeder
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import yaml


# ============================================================
# USER CONFIGURATION
# ============================================================

backbone = "infogcn2" # infogcn2, msg3d, stgcn2
dataset = "ntu" # ntu, ntu120, ucf101
evaluation = "CS" # CS/CV, CSub/CSet, 1/2/3
embedding = "cnn" # base, cnn
flow_type = "RAFT" # RAFT, LK, NF
split = "test"

sample_number = 23

save_root = "visualisations/results/decision_explanations/out"
save_root = osp.join(save_root, backbone, dataset, embedding)

RAW_DIR = osp.join(save_root, "raw")
FIG_DIR = osp.join(save_root, "figures", "saliency_maps")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# FEEDER SETUP
# ============================================================

arg = ArgClass(
    f"config/{backbone}/{dataset}/{embedding}.yaml"
)

arg.feeder_args["data_paths"] = {evaluation: "/fred/oz141/ldezoete/PoseOFF_NT/data/ntu/aligned_data/poseoff/RAFT/ntu_CS-poseoff_RAFT_D1_aligned.npz"}
arg.feeder_args["eval"] = evaluation

arg.feeder_args["use_mmap"] = True

arg.feeder_args["random_choose"] = False
arg.feeder_args["random_shift"] = False
arg.feeder_args["random_move"] = False
arg.feeder_args["random_rot"] = False

feeder = Feeder(
    **arg.feeder_args,
    split=split
)
# Get the class names...
stat_path = osp.join("data", dataset, "statistics")
with open(osp.join(stat_path, "class_names.yaml"), 'r') as f:
    class_names = yaml.safe_load(f)
with open(osp.join(stat_path, "feeder_idx_to_filename.yaml"), 'r') as f:
    feeder_filenames = yaml.safe_load(f)
    feeder_filenames = (feeder_filenames[evaluation][split])

data_numpy, label, mask, index = feeder[
    sample_number
]

# Load the saliency map saved after running skeleon_xai.py
saliency_map_pth = osp.join(RAW_DIR, "joint_temporal_saliency.npy")
if osp.isfile(saliency_map_pth):
    saliency_map = np.load(saliency_map_pth)
    assert saliency_map.shape[0] == data_numpy.shape[1]
else:
    print(f"{saliency_map_pth} does not exist.")
    print("Run `python visuaisations/results/decision_explanations/skeleton_xai.py` with appropriate settings.")
    quit()

print(f"Sample name: {feeder_filenames[sample_number]}")
print(f"Class: {label} - {class_names[label]}")
print("\t\t(sample name has class number indexed at 1, label is indexed at 0.)")
print(f"Saliency map shape: {saliency_map.shape}")
print(f"Original input data shape: {data_numpy.shape}")

body_ntu = [
    [0, 1, 20, 2, 3], # Trunk
    [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21], # arms
    [19, 18, 17, 16, 0, 12, 13, 14, 15] # leg_joints
]
# for frame_no in np.linspace(0, data_numpy.shape[1]-1, 10):
for frame_no in range(data_numpy.shape[1]):
    frame_no = int(frame_no)
    print(frame_no)
    skel_frame = rearrange(data_numpy[:, frame_no, :, 0], 'C V -> V C')
    ax = plt.figure().add_subplot(projection="3d")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Graph the joints
    for joint_no, joint in enumerate(skel_frame):
        ax.scatter(joint[0], joint[1], joint[2], marker='o', c='b', s=saliency_map[frame_no, joint_no]*100)
    for bone_group in body_ntu:
        x_plot = skel_frame[bone_group, 0]
        y_plot = skel_frame[bone_group, 1]
        z_plot = skel_frame[bone_group, 2]
        ax.plot(x_plot, y_plot, z_plot, color='r')
    ax.view_init(elev=75, azim=00, roll=90)
    plt.savefig(osp.join(FIG_DIR, f"frame{frame_no}.png"))
    plt.close()
