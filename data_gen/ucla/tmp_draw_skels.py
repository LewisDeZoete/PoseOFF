#!/usr/bin/env python3

import argparse
import os
import os.path as osp
from ucla_gendata_test import read_xyz
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from einops import rearrange



parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
parser.add_argument(
    '--data_path', default='../Datasets/N-UCLA/multiview_action'
)
parser.add_argument(
    '--label_path', default='./data/ucla/statistics/'
)
parser.add_argument(
    '--ignored_sample_path',
    default='./data/ucla/statistics/NW-UCLA_samples_with_missing_skeletons.txt'
)
parser.add_argument(
    '--out_path',
    default='./data/ucla/pose/'
)

arg = parser.parse_args()

data_path = arg.data_path
if arg.ignored_sample_path != None:
    with open(arg.ignored_sample_path, 'r') as f:
        ignored_samples = [
            line.strip() for line in f.readlines()
        ]

view = 'view_2'
for filename in os.listdir(osp.join(data_path, view)):
    if filename in ignored_samples:
        continue
    sample_folder = osp.join(data_path, view, filename)
    sample_all_files = os.listdir(sample_folder)
    sample_skel_files = \
        [filename for filename in sample_all_files if filename.endswith('skeletons.txt')]
    sample_rgb_files = \
        [filename for filename in sample_all_files if filename.endswith('rgb.jpg')]
    sample_rgb_files.sort()
    break


def project_skeletons(skeletons, fx=525.0, fy=525.0, cx=319.5, cy=239.5):
    """
    Project NW-UCLA skeletons from 3D camera coords to 2D pixel coords.

    Args:
        skeletons: np.ndarray of shape (3, T, V, M)
                   with [x, y, z] in meters.
        fx, fy, cx, cy: intrinsics for Kinect v1 (default approx).

    Returns:
        np.ndarray of shape (2, T, V, M) with [u, v] pixel coordinates.
    """
    X, Y, Z = skeletons[0], skeletons[1], skeletons[2]  # shape (T, V, M)

    # avoid divide by zero
    Z = np.where(Z == 0, 1e-6, Z)

    u = fx * (X / Z) + cx
    v = -fy * (Y / Z) + cy

    return np.stack([u, v], axis=0)  # shape (2, T, V, M)


frame_no = 20

# Get skeleton data...
sample_data = read_xyz(osp.join(data_path, view, filename))
skel = project_skeletons(sample_data['data'])[:, frame_no,...]
C, V, M = skel.shape
skel = rearrange(skel, 'C V M -> M V C', C=C, V=V, M=M)[0]

# Get image data...
im_name = sample_rgb_files[frame_no]
im = Image.open(osp.join(data_path, view, filename, im_name))

# Time to plot it !
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.imshow(im)
for joint in skel:
    ax.plot(joint[0], joint[1], 'bo')

plt.show()
