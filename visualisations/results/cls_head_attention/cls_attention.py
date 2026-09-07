#!/usr/bin/env python3

import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from config.argclass import ArgClass
from model_utils import ModelLoader
import matplotlib.pyplot as plt
from feeders.ntu_rgb_d import Feeder
from training.loss import LabelSmoothingCrossEntropy
from einops import rearrange
# from .data import 3d_skel_display


# ----------------------------------------------------
# Right now, this only works for infogcn++ when you change the reconstruction decoder to SA_GC
# USAGE:
# - edit the `dataset` and `evaluation` variables below

backbone = "infogcn2" # infogcn2, msg3d, stgcn2
dataset = 'ntu'  # ntu, ntu120, ucf101
evaluation = 'CS'  # CS/CV, CSub/CSet, 1/2/3
embedding = 'cnn' # base, abs, avg, cnn
flow_type = 'RAFT' # RAFT, LK, NF
dilation = 3

gcn_number = 1 # There are two gcn layers in the classification head...
class_number = 24 # I think 24 is kicking
# ----------------------------------------------------

# Define and create paths
save_path = f"visualisations/cls_head_attention/output/graphs/"
os.makedirs(save_path, exist_ok=True)
save_name = osp.join(save_path, f"grad_importance_{embedding}_{flow_type}.png")
print(f"Output file: { save_name }")

# Define the arguments
arg = ArgClass(f"config/{backbone}/{dataset}/{embedding}.yaml")
arg.checkpoint_file = f'results/{backbone}/{dataset}/{evaluation}/train/' \
    f'{backbone}_{dataset}_{evaluation}_{embedding}_{flow_type}_D{dilation}.pt'

# Load the model
modelLoader = ModelLoader(arg)
model = modelLoader.model
model.train()

if backbone == "infogcn2":
    # Create loss criterea
    cls_loss = LabelSmoothingCrossEntropy(arg.model_args["T"])
else:
    # Create the loss function(s)
    cls_loss = nn.CrossEntropyLoss()
    loss_funcs = {"cls_loss": cls_loss}

    # Score functions (train+test), accuracy, cls_loss
    score_funcs = ["ACC", "cls_loss"]

# Setup for feeder
arg.feeder_args['eval'] = evaluation
arg.feeder_args['use_mmap'] = True
arg.feeder_args['data_paths'][evaluation]=\
    f"data/{dataset}/aligned_data/poseoff/{flow_type}/{dataset}_{evaluation}-poseoff_{flow_type}_D{dilation}_aligned.npz" \
    if embedding != 'base' else \
    f"data/{dataset}/aligned_data/pose/{dataset}_{evaluation}-pose_aligned.npz"
arg.feeder_args['random_choose']=False
arg.feeder_args['random_shift']=False
arg.feeder_args['random_move']=False
arg.feeder_args['random_rot']=False

# Create the feeder and get the data
test_feeder = Feeder(**arg.feeder_args, split='test')
data_numpy, label, mask, index = test_feeder[class_number]
data = torch.tensor(data_numpy)
data = torch.unsqueeze(data, 0)

# Register hooks
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
# Register the fully connected layer for example
model.fc.register_forward_hook(get_activation('fc'))


# Pass data to model
if backbone == "infogcn2":
    # y_hat (1, 60, 64)
    y_hat, x_hat, z_0, z_hat_shifted, _ = model(data)
    class_preds = y_hat[0].swapaxes(0,1)
    class_preds = torch.tensor([torch.argmax(frame) for frame in class_preds]).mode()
else:
    # y_hat (1, 60)
    y_hat = model(data) 
    class_preds = y_hat[0]

print(f"Predicted class: {class_preds.values}")
print(f"Real class: {label}")

# Reshaping the label array to perform loss calc...
B = 1
label = torch.tensor(label).view(1, B, 1).expand(1, B, y_hat.size(2))
y_hat_ = rearrange(y_hat, "b i t -> (b t) i")
cls_loss = arg.lambda_1 * cls_loss(y_hat_, label.reshape(-1))

# zero the gradients
model.zero_grad()

# Backward pass
cls_loss.backward()

def get_graph_topology_grad_attr():
    """Get gradient topology gradient attribution.
    What is shows:
        - edge importance
        - learned adjacency sensitivity
        - ∂Loss / ∂A 
    What it tells us:
        - which skeleton connections matter most?
    """
    # Now gradients are stored in .grad for each parameter
    A = model.cls_decoder[gcn_number].shared_topology  # gradient of adjacency
    A_grad = A.grad.detach().cpu().numpy()

    grad_importance = np.abs(A_grad)
    np.save(
        f"results/visualisations/cls_head_attention/output/data/grad_importance_{embedding}.npy",
        grad_importance
    )

def get_attn_maps():
    """Get graph attention maps.
    What it shows:
        - attention matrices
        - attention evolution across frames
    What it tells us:
        - which joints attend to which other joints?
    """
    attn0 = model.cls_decoder[0].get_attn()
    attn1 = model.cls_decoder[1].get_attn()

    attn = rearrange(attn1, '(B T o) H I J -> B T o H I J', B=1, T=64, o=2).detach().numpy()
    attn=attn[0].mean(axis=1) # (T, H, V, V)
    mean_axis = 0 # Axis over which we want to calculate the mean...

def get_joint_saliency():
    input.grad
    data.requires_grad_(True)
    joint_saliency = torch.abs(data.grad)
    joint_saliency = joint_saliency.mean(dim=(0,1,4))

def Grad_CAM():
    """Grad-CAM
    What it tells us:
        - which joints at which times caused this action prediction?
    """
    if backbone == "infogcn2":
        target_layer = model.cls_decoder[1]
    elif backbone == "stgcn2":
        target_layer = model.gcn[-1]
    elif backbone == "msg3d":
        target_layer = model.tcn3

    activations = {}
    gradients = {}

    def forward_hook(m,i,o):
        activations["value"] = o
    def backwards_hook(m,gin,gout):
        gradients["value"] = gout[0]

    target_layer.register_forward_hook(...)
    target_layer.register_full_backwards_hook(...)

    weights = gradients.mean(dim=(2,3), keepdim=True)

    cam = torch.sum(
        weights * activations,
        dim=1
    )

    cam = F.relu(cam)
        
fig, axs = plt.subplots(2,4, figsize=(20,10))
fig.suptitle(f"Limb attention for {embedding} model", fontsize=25)
axs[0,0].set_title("Frame 0")
axs[0,0].imshow(np.mean(attn[0], axis=0), cmap="hot")
axs[1,0].bar(np.linspace(1,25,25), np.mean(np.mean(attn[0], axis=0), axis=mean_axis))
axs[1,0].set_ylabel("Attention weight")

axs[0,1].set_title("Frame 10")
axs[0,1].imshow(np.mean(attn[10], axis=0), cmap="hot")
axs[1,1].bar(np.linspace(1,25,25), np.mean(np.mean(attn[10], axis=0), axis=mean_axis))

axs[0,2].set_title("Frame 30")
axs[0,2].imshow(np.mean(attn[30], axis=0), cmap="hot")
axs[1,2].bar(np.linspace(1,25,25), np.mean(np.mean(attn[30], axis=0), axis=mean_axis))

axs[0,3].set_title("Frame 60")
axs[0,3].imshow(np.mean(attn[60], axis=0), cmap="hot")
axs[1,3].bar(np.linspace(1,25,25), np.mean(np.mean(attn[50], axis=0), axis=mean_axis))

# # Set the x_label and y_lims for the bar graphs
# for i in range(4):
#     axs[1,i].set_xlabel("Limb number (1-25)")
#     axs[1,i].set_ylim(0,0.35)
plt.savefig(save_name)
