#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from config.argclass import ArgClass
from model_utils import ModelLoader
import matplotlib.pyplot as plt
from feeders.ntu_rgb_d import Feeder
from training.loss import LabelSmoothingCrossEntropy
from einops import rearrange

# ----------------------------------------------------
# Right now, this only works for infogcn++ when you change the reconstruction decoder to SA_GC
# USAGE:
# - edit the `dataset` and `evaluation` variables below
# - Set values of plot_* as needed, if True, that plot will be generated
# - edit the `plot_data` for specific extensions (keys) and plot params
#      python ./results/visualisations/results_vis.py

backbone = "msg3d" # InfoGCN++
dataset = 'ntu'  # ntu, ntu120, ucf101
evaluation = 'CS'  # CS/CV, CSub/CSet, 1/2/3
embedding = 'base' # base, abs, avg, cnn
flow_type = 'RAFT' # RAFT, LK, norm
dilation = 3

gcn_number = 1 # There are two gcn layers in the classification head...
class_number = 24 # I think 24 is kicking
# ----------------------------------------------------

# Layer 1, layer 3, layer 6, final layer
layers_to_register = {
    "infogcn2": ["classifier_lst", "cls_decoder"],
    "msg3d": ["to_joint_embedding", "gcn3d1", "sgcn1", "tcn3", "fc"]
}

def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
    var1 = np.linalg.norm(X.T @ X, 'fro')
    var2 = np.linalg.norm(Y.T @ Y, 'fro')

    return hsic / (var1 * var2)

def get_activation(modelname, layername):
    def hook(model, input, output):
        activation[modelname][layername] = output.detach()
    return hook

# Define the arguments
arg_base = ArgClass(f"config/msg3d/{dataset}/base.yaml")
arg_poseoff = ArgClass(f"config/msg3d/{dataset}/cnn.yaml")
# arg_base.checkpoint_file = f'results/{backbone}/{evaluation}/train/' \
#     f'{backbone}_{dataset}_{evaluation}_base.pt'
# arg_poseoff.checkpoint_file = f'results/{backbone}/{evaluation}/train/' \
#     f'{backbone}_{dataset}_{evaluation}_{embedding}_{flow_type}_D{dilation}.pt'
arg_base.checkpoint_file = "results/msg3d/ntu/CS/train/msg3d_ntu_CS_base.pt"
arg_poseoff.checkpoint_file = "results/msg3d/ntu/CS/train/msg3d_ntu_CS_cnn_RAFT_D3.pt"
# Load the models
modelLoader_base = ModelLoader(arg_base)
modelLoader_poseoff = ModelLoader(arg_poseoff)

model_base = modelLoader_base.model
model_poseoff = modelLoader_poseoff.model

model_base.train()
model_poseoff.train()

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
arg_base.feeder_args['eval'] = evaluation
arg_base.feeder_args['use_mmap'] = True
arg_base.feeder_args['data_paths'][evaluation]=\
    f"data/{dataset}/aligned_data/poseoff/{flow_type}/{dataset}_{evaluation}-poseoff_{flow_type}_D{dilation}_aligned.npz" \
    if embedding != 'base' else \
    f"data/{dataset}/aligned_data/pose/{dataset}_{evaluation}-pose_aligned.npz"
arg_base.feeder_args['random_choose']=False
arg_base.feeder_args['random_shift']=False
arg_base.feeder_args['random_move']=False
arg_base.feeder_args['random_rot']=False

# Create the feeder and get the data
test_feeder = Feeder(**arg_base.feeder_args, split='test')
test_dataloader = DataLoader(
    test_feeder,
    batch_size=8,
    num_workers=2,
    shuffle=False,
    pin_memory=True
)

# Dict to store the activations during forward pass
activation = {} # {"base": {"fc": [...], "layer1": [...]}, "poseoff": {"fc": [...], "layer1": [...]}}

# Register the fully connected layer for example
for register_layer in layers_to_register[backbone]:
    print(f"Registering layer forward hook: {register_layer}")
    model_base.fc.register_forward_hook(get_activation("base", register_layer))
    model_poseoff.fc.register_forward_hook(get_activation("poseoff", register_layer))
    activation["base"] = {register_layer: []}
    activation["poseoff"] = {register_layer: []}


for x, y, mask, index in test_dataloader:

    # Pass data to model
    if backbone == "infogcn2":
        y_hat, x_hat, z_0, z_hat_shifted, _ = model_base(x)
        y_hat, x_hat, z_0, z_hat_shifted, _ = model_poseoff(x)
    else:
        y_hat = model_base(x)
        y_hat = model_poseoff(x)

    break

# cka_score = linear_cka(
#     activation["base"]["tcn3"],
#     activation["poseoff"]["tcn3"]
# )

print(activation["base"].keys())
print(activation["poseoff"].keys())

# Register the fully connected layer for example
for register_layer in layers_to_register[backbone]:
    cka_score = linear_cka(
        activation["base"][register_layer],
        activation["poseoff"][register_layer]
    )

    print(f"{register_layer}    CKA = {cka_score:.4f}")
