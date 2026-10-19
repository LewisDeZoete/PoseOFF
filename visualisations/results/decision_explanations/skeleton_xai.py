#!/usr/bin/env python3

"""
skeleton_xai.py

Unified explainability toolkit for:

    - InfoGCN++
    - MS-G3D
    - ST-GCN++

Provides:

    1. Input Saliency Maps
    2. Graph Edge Attribution
    3. Grad-CAM
    4. Attention Visualisation (when available)
    5. Joint Importance Ranking
    6. Frame Importance Ranking
    7. Combined Comparison Figure

============================================================
EXPLANATIONS
============================================================

INPUT SALIENCY
--------------
Computes:

    dLoss / dInput

Shows which joints and frames influence
the final decision.


EDGE ATTRIBUTION
----------------
Computes:

    dLoss / dGraphTopology

Shows which graph edges matter.


GRAD-CAM
--------
Computes:

    dClassScore / dFeatureMaps

Shows which joints and times caused
the final prediction.


ATTENTION MAPS
--------------
Visualises self-attention matrices.

Shows where the model is looking.


COMPARISON FIGURE
-----------------
Displays all explanation methods side-by-side.
"""

import os
import os.path as osp
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from config.argclass import ArgClass
from model_utils import ModelLoader

from feeders.ntu_rgb_d import Feeder
from training.loss import LabelSmoothingCrossEntropy


# ============================================================
# USER CONFIGURATION
# ============================================================

backbone = "stgcn2"     # infogcn2, msg3d, stgcn2
# TODO: TM

dataset = "ntu"
evaluation = "CS"

embedding = "cnn"

flow_type = "RAFT"

dilation = 3


sample_number = 23 # Sometimes this equals class number...

gcn_number = 2

save_root = "visualisations/results/decision_explanations/out"
save_root = osp.join(save_root, backbone, dataset, embedding)

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

RAW_DIR = osp.join(save_root, "raw")
FIG_DIR = osp.join(save_root, "figures")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# INITIALISE LOGGING
# ============================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=osp.join(RAW_DIR, "skel_xia.log"),
    encoding='utf-8',
    filemode='w',
    level=logging.INFO
)
logger.info(f"Model: {backbone}-{embedding} - {flow_type} - D{dilation}")
logger.info(f"Dataset: {dataset} - {evaluation}")
logger.info(f"Sample number: {sample_number}")

# ============================================================
# LOAD MODEL
# ============================================================

arg = ArgClass(
    f"config/{backbone}/{dataset}/{embedding}.yaml"
)

file_end = f"_{flow_type}_D{dilation}.pt" if embedding != "base" else ".pt"
arg.checkpoint_file = (
    f"results/{backbone}/{dataset}/{evaluation}/train/"
    f"{backbone}_{dataset}_{evaluation}_{embedding}" +
    file_end
)

modelLoader = ModelLoader(arg)
model = modelLoader.model

model.train()

# ============================================================
# LOSS
# ============================================================

if backbone == "infogcn2":
    loss_fn = LabelSmoothingCrossEntropy(
        arg.model_args["T"]
    )
else:
    loss_fn = nn.CrossEntropyLoss()

# ============================================================
# FEEDER SETUP
# ============================================================

arg.feeder_args["eval"] = evaluation

arg.feeder_args["use_mmap"] = True

arg.feeder_args["random_choose"] = False
arg.feeder_args["random_shift"] = False
arg.feeder_args["random_move"] = False
arg.feeder_args["random_rot"] = False

test_feeder = Feeder(
    **arg.feeder_args,
    split="test"
)

data_numpy, label, mask, index = test_feeder[
    sample_number
]

data = torch.tensor(
    data_numpy,
    dtype=torch.float32
)

data = data.unsqueeze(0)

# ============================================================
# INPUT SALIENCY
# ============================================================
#
# Computes:
#
#     dLoss / dInput
#
# Gives:
#
#     Joint x Time importance
#
# ============================================================

data.requires_grad_(True)

# ============================================================
# GRAD-CAM HOOKS
# ============================================================

activations = {}
gradients = {}

def forward_hook(module, inputs, output):
    activations["value"] = output

def backward_hook(module, gin, gout):
    gradients["value"] = gout[0]

# ------------------------------------------------------------
# Pick Grad-CAM layer
# ------------------------------------------------------------

if backbone == "infogcn2":

    gradcam_target = model.cls_decoder[-1]

elif backbone == "stgcn2":

    gradcam_target = model.gcn[-1]

elif backbone == "msg3d":

    gradcam_target = model.tcn3

else:

    raise ValueError(backbone)

gradcam_target.register_forward_hook(
    forward_hook
)

gradcam_target.register_full_backward_hook(
    backward_hook
)

# ============================================================
# FORWARD PASS
# ============================================================

if backbone == "infogcn2":

    y_hat, x_hat, z0, zhat, _ = model(data)

    class_scores = y_hat.mean(-1)

else:

    class_scores = model(data)

pred_class = torch.argmax(
    class_scores,
    dim=1
)

logger.info(f"Prediction: {pred_class.item()}")
logger.info(f"Label:      {label}")

# ============================================================
# LOSS
# ============================================================

target_score = class_scores[
    0,
    pred_class.item()
]

model.zero_grad()

target_score.backward(
    retain_graph=True
)

# ============================================================
# INPUT SALIENCY
# ============================================================

input_saliency = torch.abs(
    data.grad
)

saliency_map = input_saliency.mean(
    dim=(0,1,4)
)

saliency_map = saliency_map.detach().cpu().numpy()

np.save(
    osp.join(
        RAW_DIR,
        "joint_temporal_saliency.npy"
    ),
    saliency_map
)

# ============================================================
# JOINT IMPORTANCE
# ============================================================

joint_scores = saliency_map.mean(axis=0)

plt.figure(figsize=(10,6))
plt.bar(
    np.arange(len(joint_scores)),
    joint_scores
)
plt.title("Joint Importance")
plt.xlabel("Joint")
plt.ylabel("Importance")
plt.tight_layout()

plt.savefig(
    osp.join(
        FIG_DIR,
        "joint_importance_barplot.png"
    )
)

plt.close()

# TODO: DELETE THIS
body_ntu = [
    [0, 1, 20, 2, 3], # Trunk
    [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21], # arms
    [19, 18, 17, 16, 0, 12, 13, 14, 15] # leg_joints
]
frame = 10
skel_frame = rearrange(data_numpy[:, frame, :, 0], 'C V -> V C')
ax = plt.figure().add_subplot(projection="3d")
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# Graph the joints
for joint_no, joint in enumerate(skel_frame):
    ax.scatter(joint[0], joint[1], joint[2], marker='o', c='b', s=saliency_map[frame, joint_no]*100)
for bone_group in body_ntu:
    x_plot = skel_frame[bone_group, 0]
    y_plot = skel_frame[bone_group, 1]
    z_plot = skel_frame[bone_group, 2]
    ax.plot(x_plot, y_plot, z_plot, color='r')
ax.view_init(elev=75, azim=00, roll=90)
plt.savefig(osp.join(FIG_DIR, "WEIGHT_SKE_JOINTS.png"))

# ============================================================
# FRAME IMPORTANCE
# ============================================================

frame_scores = saliency_map.mean(axis=1)

plt.figure(figsize=(12,4))
plt.plot(frame_scores)

plt.title("Frame Importance")

plt.xlabel("Frame")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig(
    osp.join(
        FIG_DIR,
        "frame_importance_barplot.png"
    )
)

plt.close()

# ============================================================
# SALIENCY HEATMAP
# ============================================================

plt.figure(figsize=(12,8))

plt.imshow(
    saliency_map.T,
    aspect="auto",
    cmap="hot"
)

plt.colorbar()

plt.xlabel("Frame")
plt.ylabel("Joint")

plt.title(
    "Joint Temporal Saliency"
)

plt.tight_layout()

plt.savefig(
    osp.join(
        FIG_DIR,
        "joint_temporal_saliency.png"
    )
)

plt.close()

# ============================================================
# GRAD-CAM
# ============================================================

if (
    "value" in activations
    and
    "value" in gradients
):

    act = activations["value"]

    grad = gradients["value"]

    while act.ndim > 4:
        act = act.mean(0)

    while grad.ndim > 4:
        grad = grad.mean(0)

    weights = grad.mean(
        dim=(2,3),
        keepdim=True
    )

    cam = (
        weights * act
    ).sum(dim=1)

    cam = F.relu(cam)

    cam = cam[0]

    cam = (
        cam
        /
        (cam.max() + 1e-8)
    )

    cam = cam.detach().cpu().numpy()

    np.save(
        osp.join(
            RAW_DIR,
            "gradcam.npy"
        ),
        cam
    )

    plt.figure(figsize=(12,8))

    plt.imshow(
        cam,
        aspect="auto",
        cmap="jet"
    )

    plt.colorbar()

    plt.title("Grad-CAM")

    plt.xlabel("Joint")

    plt.ylabel("Frame")

    plt.tight_layout()

    plt.savefig(
        osp.join(
            FIG_DIR,
            "gradcam_joint_temporal.png"
        )
    )

    plt.close()

# ============================================================
# EDGE ATTRIBUTION
# ============================================================

edge_importance = None

try:

    if backbone == "infogcn2":

        A = model.cls_decoder[
            gcn_number
        ].shared_topology

        edge_importance = (
            A.grad.abs()
            .detach()
            .cpu()
            .numpy()
        )

    elif backbone == "stgcn2":

        A = model.gcn[
            gcn_number
        ].gcn.PA

        edge_importance = (
            A.grad.abs()
            .detach()
            .cpu()
            .numpy()
        )

except Exception as e:

    logger.info(
        f"Edge attribution unavailable: {e}"
    )

if edge_importance is not None:

    edge_map = edge_importance.mean(0)

    np.save(
        osp.join(
            RAW_DIR,
            "edge_importance.npy"
        ),
        edge_map
    )

    plt.figure(figsize=(8,8))

    plt.imshow(
        edge_map,
        cmap="hot"
    )

    plt.colorbar()

    plt.title(
        "Graph Edge Gradient Importance"
    )

    plt.tight_layout()

    plt.savefig(
        osp.join(
            FIG_DIR,
            "edge_gradient_importance.png"
        )
    )

    plt.close()

# ============================================================
# ATTENTION
# ============================================================

try:

    if hasattr(model, "get_attention"):

        attns = model.get_attention()

        for layer_idx, attn in enumerate(attns):

            attn_np = (
                attn.mean(0)
                .detach()
                .cpu()
                .numpy()
            )

            np.save(
                osp.join(
                    RAW_DIR,
                    f"attention_layer_{layer_idx}.npy"
                ),
                attn_np
            )

            plt.figure(figsize=(8,6))

            plt.imshow(
                attn_np.mean(0),
                aspect="auto",
                cmap="viridis"
            )

            plt.colorbar()

            plt.title(
                f"Attention Layer {layer_idx}"
            )

            plt.tight_layout()

            plt.savefig(
                osp.join(
                    FIG_DIR,
                    f"attention_layer_{layer_idx}.png"
                )
            )

            plt.close()

except Exception as e:

    logger.info(
        f"Attention unavailable: {e}"
    )

# ============================================================
# COMPARISON FIGURE
# ============================================================

try:

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(16,12)
    )

    axs[0,0].imshow(
        saliency_map.T,
        aspect="auto",
        cmap="hot"
    )

    axs[0,0].set_title(
        "Input Saliency"
    )

    if "cam" in locals():

        axs[0,1].imshow(
            cam,
            aspect="auto",
            cmap="jet"
        )

        axs[0,1].set_title(
            "Grad-CAM"
        )

    if edge_importance is not None:

        axs[1,0].imshow(
            edge_importance.mean(0),
            cmap="hot"
        )

        axs[1,0].set_title(
            "Edge Attribution"
        )

    axs[1,1].plot(
        frame_scores
    )

    axs[1,1].set_title(
        "Frame Importance"
    )

    plt.tight_layout()

    plt.savefig(
        osp.join(
            FIG_DIR,
            "explainability_comparison.png"
        )
    )

    plt.close()

except Exception as e:

    logger.info(
        f"Comparison figure failed: {e}"
    )

print("\nXAI complete.")
print(f"Saved results to {save_root}")
logger.info(f"Saved results to {save_root}")
