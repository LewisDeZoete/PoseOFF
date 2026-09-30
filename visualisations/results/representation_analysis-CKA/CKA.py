#!/usr/bin/env python3

# What is CKA?
# Cented Kernel Alignment (CKA) is a similarity index used to analyse the distance between
# the hidden layers of neural networks trained from different inidialisations.
# https://arxiv.org/abs/1905.00414

import os
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
# TODO: Perform CKA after models see the *full* dataset (will require slurm job)
# Right now, this only works for infogcn++ when you change the reconstruction decoder to SA_GC
# USAGE:
# - edit the `backbone`, `dataset`, `evaluation`, `flow_type` and `dilation` variables below
# - edit the `layer_names` as needed
#     - NOTE: You can find the names of layers by running `python model_utils/write_model_orgs.py`
#     - Model outlines are stored in `org/model_outlines/{backbone}/{backbone}_{embedding}.org`
# - run this file `python visualisations/results/representation_analysis-CKA/CKA.py`

backbone = "stgcn2" # infogcn2, msg3d, stgcn2
dataset = "ntu"  # ntu, ntu120, ucf101
evaluation = "CS"  # CS/CV, CSub/CSet, 1/2/3
flow_type = "RAFT" # RAFT, LK, norm
dilation = 3
# ----------------------------------------------------

layer_names = {
    "infogcn2": ["temporal_encoder", "diffeq_solver", "recon_decoder", "cls_decoder"],
    "msg3d": ["to_joint_embedding", "gcn3d1", "sgcn1", "tcn3", "fc"],
    "stgcn2": ["gcn.0", "gcn.3", "gcn.7", "gcn.9", "head"]
}

# ---------------------------------------------------------------------------
# 1. Core CKA math
# ---------------------------------------------------------------------------

def gram_linear(x: torch.Tensor) -> torch.Tensor:
    """x: [N, D] activations -> [N, N] linear Gram matrix."""
    return x @ x.T

def center_gram(gram: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix (double-centering, as in the CKA paper)."""
    n = gram.shape[0]
    unit = torch.ones(n, n, device=gram.device, dtype=gram.dtype)
    identity = torch.eye(n, device=gram.device, dtype=gram.dtype)
    H = identity - unit / n
    return H @ gram @ H
 
 
def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Linear CKA between activations x [N, D1] and y [N, D2].
    N must match (same examples through both models); D1 and D2 can differ freely.
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
 
    gx = center_gram(gram_linear(x))
    gy = center_gram(gram_linear(y))
 
    hsic = (gx * gy).sum()
    norm_x = torch.linalg.norm(gx)
    norm_y = torch.linalg.norm(gy)
    return (hsic / (norm_x * norm_y)).item()

# ---------------------------------------------------------------------------
# 2. Collecting activations via forward hooks
# ---------------------------------------------------------------------------
def pool_activation(out: torch.Tensor) -> torch.Tensor:
    """
    ST-GCN / CTR-GCN style blocks (which SODE's backbone is built on) emit
    [B, C, T, V] tensors: batch, channels, frames, joints.
    Mean-pool over T and V so each example becomes a flat [C] vector instead
    of flattening the whole [C, T, V] tensor. This isn't required for CKA's
    validity (see note above), but it keeps the Gram matrix computation cheap
    and avoids frame-count mismatches between the flow-augmented stream
    (which may have T-1 frames if flow is computed between consecutive
    frames) and the skeleton-only stream.
    """
    out = out.detach().float()
    if out.dim() == 4:          # [B, C, T, V]
        return out.mean(dim=(2, 3))
    elif out.dim() == 3:        # [B, T, C] or [B, C, T] -- pool the time dim
        return out.mean(dim=1)
    else:                       # already [B, C] (e.g. post-GAP embedding)
        return out.reshape(out.shape[0], -1)

def register_hooks(model, layer_names):
    """
    layer_names: list of dotted attribute paths, e.g.
        ["l1", "l4", "l7", "l10"]  or  ["backbone.blocks.9", "fc"]
    Uses model.get_submodule so nested attributes work.
    Returns (store, handles) — store[name] accumulates pooled batches.
    """
    store = {name: [] for name in layer_names}
    handles = []
 
    def make_hook(name):
        def hook(module, inp, out):
            store[name].append(pool_activation(out).cpu())
        return hook
 
    for name in layer_names:
        layer = model.get_submodule(name)
        handles.append(layer.register_forward_hook(make_hook(name)))
 
    return store, handles

def collect_multi_layer_activations(model, layer_names, dataloader, device="cpu"):
    """
    Run `model` over `dataloader`, capturing pooled activations at every
    layer in `layer_names` in a single pass.
    """
    store, handles = register_hooks(model, layer_names)
    model.eval().to(device)
 
    with torch.no_grad():
        for batch_no, (x, y, mask, index) in enumerate(dataloader):
            model(x.float().to(device))
            if batch_no > 20:
                break
 
    for h in handles:
        h.remove()
 
    return {name: torch.cat(acts, dim=0) for name, acts in store.items()}  # each [N, C]

# ---------------------------------------------------------------------------
# 3. Load data, pass to models
# ---------------------------------------------------------------------------

def main():
    # Define the arguments
    arg_base = ArgClass(f"config/{backbone}/{dataset}/base.yaml")
    arg_poseoff = ArgClass(f"config/{backbone}/{dataset}/cnn.yaml")
    arg_base.checkpoint_file = f'results/{backbone}/{dataset}/{evaluation}/train/' \
        f'{backbone}_{dataset}_{evaluation}_base.pt'
    arg_poseoff.checkpoint_file = f'results/{backbone}/{dataset}/{evaluation}/train/' \
        f'{backbone}_{dataset}_{evaluation}_cnn_{flow_type}_D{dilation}.pt'

    # Load the models
    modelLoader_base = ModelLoader(arg_base)
    modelLoader_poseoff = ModelLoader(arg_poseoff)

    model_base = modelLoader_base.model
    model_poseoff = modelLoader_poseoff.model

    # Setup for feeder
    arg_poseoff.feeder_args['eval'] = evaluation
    arg_poseoff.feeder_args['use_mmap'] = True
    arg_poseoff.feeder_args['data_paths'][evaluation]=\
        f"data/{dataset}/aligned_data/poseoff/{flow_type}/{dataset}_{evaluation}-poseoff_{flow_type}_D{dilation}_aligned.npz" \
        # if embedding != 'base' else \
        # f"data/{dataset}/aligned_data/pose/{dataset}_{evaluation}-pose_aligned.npz"
    arg_poseoff.feeder_args['random_choose']=False
    arg_poseoff.feeder_args['random_shift']=False
    arg_poseoff.feeder_args['random_move']=False
    arg_poseoff.feeder_args['random_rot']=False

    # Create the feeder and get the data
    FeederClass = arg_poseoff.import_class(arg_poseoff.feeder)
    test_dataset = FeederClass(**arg_poseoff.feeder_args, split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=2,
        shuffle=False,
        pin_memory=True
    )

    acts_base = collect_multi_layer_activations(model_base, layer_names[backbone], test_dataloader)
    acts_poseoff = collect_multi_layer_activations(model_poseoff, layer_names[backbone], test_dataloader)

    results = {}
    for layer_name in layer_names[backbone]:
        score = linear_cka(acts_base[layer_name], acts_poseoff[layer_name])
        results[f"{layer_name} <-> {layer_name}"] = score
    print(results)
    
if __name__ == '__main__':
    main()
