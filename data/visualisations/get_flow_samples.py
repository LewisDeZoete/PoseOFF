#!/usr/bin/env python3
'''
EXAMPLE:
srun --time=0:05:00 --gres=gpu:1 --mem-per-cpu=16G \
python data/visualisations/get_flow_samples.py --dataset ntu --sample_name S001C001P001R001A055

# --------------------------------------
sss is the setup number
ccc is the camera ID
ppp is the performer (subject) ID
rrr is the replication number (1 or 2)
aaa is the action class label.
# --------------------------------------
'''

from config.argclass import ArgClass
from data_gen.utils import LoadVideo, GetFlow, PoseOFFSampler
import numpy as np
import torch
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.v2 as v2
from einops import rearrange
import argparse
import pickle
import os.path as osp



parser = argparse.ArgumentParser(description='Quick flow data generator')
parser.add_argument(
    '--dataset',
    dest='dataset',
    default='ntu',
    help='Dataset, either `ntu`, `ntu120` or `ucf101` (default=ntu)\
    \nNOTE: ucf101 not yet implemented...')
parser.add_argument(
    '--sample_name',
    dest='sample_name',
    default=None,
    type=str,
    help='Name of the skeleton sample (or video name) you want to load...'
)
parser.add_argument(
    '--out_path',
    dest='out_path',
    default='./data/visualisations/RAW',
    help='Output path for the npz file generated.'
)
parser.add_argument(
    '--debug',
    action='store_true',
    help='If passed, reduce the size and max frames of processed to check it works.'
)
parsed = parser.parse_args()

# Get command line args
dataset = parsed.dataset
sample_name = parsed.sample_name
out_path = parsed.out_path

# Get the arg class
args = ArgClass(arg=f"./config/infogcn2/{dataset}/cnn.yaml", verbose=True)
transform_args = args.extractor
max_frames = 300

# Debug arguments for quick testing
if parsed.debug:
    max_frames=5
    transform_args['flow']['imsize'] = [240, 320]


# Get the correct device...
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Create the optical flow model (pytorch's RAFT)
weights = torch.load(transform_args['flow']['weights'], weights_only=True, map_location=device)
model = raft_large(progress=False)
model.load_state_dict(weights)
model = model.eval().to(device)

# Create the transforms for extracting the video and generating the optical flow...
rgb_transforms = LoadVideo(max_frames=max_frames)
flow_transforms = v2.Compose([
    v2.Resize(size=transform_args['flow']['imsize']),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # map [0, 1] into [-1, 1]
    GetFlow(model=model, device=device, minibatch_size=transform_args['flow']['minibatch_size'])
    # GetFlow(model=model, device=device, minibatch_size=2) # For non-resized videos
    ])
transform_args['poseoff']['norm'] = False
transform_args['poseoff']['match_pose'] = False
transform_args['poseoff']['ntu'] = True

# Create the PoseOFFSampler transform object
poseOFFTransform = PoseOFFSampler(**transform_args['poseoff'])

# Load the names of the skeleton files
dataset_extn = '' if dataset == 'ntu' else '120'
skes_names = np.loadtxt(
    f"./data/ntu{dataset_extn}/statistics/ntu_rgbd{dataset_extn}-available.txt",
    dtype=str
)
ske_number = list(skes_names).index(sample_name)
print(f"Video name: {sample_name}")
print(f"Skeleton number: {ske_number}")

# Example usage
video_path = f"../Datasets/NTU_RGBD{dataset_extn}/nturgb+d_rgb{dataset_extn}/{sample_name}_rgb.avi"
pose_path = f"./data/ntu{dataset_extn}/raw_data/raw_skes_data.pkl"
pose_denoised_path = f"./data/ntu{dataset_extn}/denoised_data/raw_denoised_colors.pkl"

# Get the pose data
with open(pose_denoised_path, 'rb') as f:
    poses = pickle.load(f)

# Get all the data
pose_data = poses[ske_number]
rgb_data = rgb_transforms(video_path).to(device)
flow_data = flow_transforms(rgb_data)
poseoff_data = poseOFFTransform(flow_data, pose_data.copy().transpose(3,0,2,1))

if not parsed.debug:
    save_path = osp.join(out_path, f"{sample_name}.npz")
    np.savez(
        save_path,
        pose=pose_data,
        rgb=rgb_data.to('cpu').numpy(),
        flow=flow_data.to('cpu').numpy(),
        poseoff=poseoff_data
        )
    print(f"Saved: {save_path}")
