#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from config.argclass import ArgClass
from model.load_model import ModelLoader
import matplotlib.pyplot as plt
from feeders.ntu_rgb_d import Feeder
from training.loss import LabelSmoothingCrossEntropy
from einops import rearrange
# from .data import 3d_skel_display


# ----------------------------------------------------
# USAGE:
# - edit the `dataset` and `evaluation` variables below
# - Set values of plot_* as needed, if True, that plot will be generated
# - edit the `plot_data` for specific extensions (keys) and plot params
#      python ./results/visualisations/results_vis.py

dataset = 'ntu120'  # ntu, ntu120, ucf101
model_type = 'base' # base, abs, avg, cnn
evaluation = 'CSub'  # CS/CV, CSub/CSet, 1/2/3
dilation = 3
modifier = f"D{dilation}" # for loading checkpoint (can be "SA-GC")

gcn_number = 1 # There are two gcn layers in the classification head...
class_number = 24 # I think 24 is kicking
# ----------------------------------------------------

# Define the arguments
arg = ArgClass(f"config/{dataset}/{model_type}.yaml")
arg.checkpoint_file = f'./results/{dataset}/{evaluation}/train/\
{dataset}_{evaluation}_{model_type}_{modifier}.pt'

# Load the model
modelLoader = ModelLoader(arg)
skel_model = modelLoader.model
skel_model.train()

# Create loss criterea
cls_loss = LabelSmoothingCrossEntropy(T=arg.model_args["T"])

# Setup for feeder
arg.feeder_args['eval'] = evaluation
arg.feeder_args['use_mmap'] = True
arg.feeder_args['data_paths'][evaluation]=\
    f"data/{dataset}/aligned_data/{dataset}_{evaluation}-flowpose_D{dilation}_aligned.npz" \
    if model_type != 'base' else \
    f"data/{dataset}/aligned_data/{dataset}_{evaluation}-pose_aligned.npz"
arg.feeder_args['random_choose']=False
arg.feeder_args['random_shift']=False
arg.feeder_args['random_move']=False
arg.feeder_args['random_rot']=False

# Create the feeder and get the data
test_feeder = Feeder(**arg.feeder_args, split='test')
data_numpy, label, mask, index = test_feeder[class_number]
data = torch.tensor(data_numpy)
data = torch.unsqueeze(data, 0)

# Pass data to model
y_hat, x_hat, z_0, z_hat_shifted, _ = skel_model(data)
# y_hat (1, 60, 64)
class_preds = y_hat[0].swapaxes(0,1)
class_preds = torch.tensor([torch.argmax(frame) for frame in class_preds]).mode()
print(f"Predicted class: {class_preds.values}")
# print(f"Predicted class: {y_hat[0,:,-1]}")
print(f"Real class: {label}")

# Reshaping the label array to perform loss calc...
B = 1
label = torch.tensor(label).view(1, B, 1).expand(1, B, y_hat.size(2))
y_hat_ = rearrange(y_hat, "b i t -> (b t) i")
cls_loss = arg.lambda_1 * cls_loss(y_hat_, label.reshape(-1))

# zero the gradients
skel_model.zero_grad()

# Backward pass
cls_loss.backward()

# Now gradients are stored in .grad for each parameter
A = skel_model.cls_decoder[gcn_number].shared_topology  # gradient of adjacency
A_grad = A.grad.detach().cpu().numpy()

grad_importance = np.abs(A_grad)
np.save(f"grad_importance_{model_type}.npy", grad_importance)

attn0 = skel_model.cls_decoder[0].get_attn()
attn1 = skel_model.cls_decoder[1].get_attn()

attn = rearrange(attn1, '(B T o) H I J -> B T o H I J', B=1, T=64, o=2).detach().numpy()
attn=attn[0].mean(axis=1) # (T, H, V, V)
mean_axis = 0 # Axis over which we want to calculate the mean...
print(attn.shape)

fig, axs = plt.subplots(2,4, figsize=(20,10))
fig.suptitle(f"Limb attention for {model_type} model", fontsize=25)
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

# Set the x_label and y_lims for the bar graphs
for i in range(4):
    axs[1,i].set_xlabel("Limb number (1-25)")
    axs[1,i].set_ylim(0,0.35)
# plt.show()
plt.savefig('./TMP.png')




# # Define the matplotlib style
# plt.style.use('./results/visualisations/plot_styles.mplstyles')

# fig, axs = plt.subplots(2,4, figsize=(10,10))
# axs[0,0].imshow(grad_importance[0], cmap="hot")
# axs[1,0].bar(np.linspace(1,25,25), np.mean(grad_importance[0], axis=0))

# axs[0,1].imshow(grad_importance[1], cmap="hot")
# axs[1,1].bar(np.linspace(1,25,25), np.mean(grad_importance[1], axis=0))

# axs[0,2].imshow(grad_importance[2], cmap="hot")
# axs[1,2].bar(np.linspace(1,25,25), np.mean(grad_importance[2], axis=0))

# axs[0,3].imshow(grad_importance.mean(axis=0), cmap="hot")
# axs[1,3].bar(np.linspace(1,25,25), grad_importance.mean(axis=0).mean(axis=0))

# plt.savefig('./TMP.png')
