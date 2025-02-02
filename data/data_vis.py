import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from config.argclass import ArgClass
import decord
from decord import VideoReader, cpu

# ----------------- #
class_no = 20
video_number = 104
frame_number = 5
# ----------------- #



arg = ArgClass(arg='./config/custom_pose/train_joint_infogcn.yaml')
rgb_path = arg.feeder_args['data_paths']['rgb_path']
flow_path = arg.feeder_args['data_paths']['flow_path']
pose_path = arg.feeder_args['data_paths']['pose_path']

class_name = os.listdir(flow_path)[class_no]
video_name = os.listdir(os.path.join(flow_path, class_name))[video_number].split('.')[0]

# Get the paths to the videos
rgb_video_path = os.path.join(rgb_path, class_name, f'{video_name}.avi')
flow_video_path = os.path.join(flow_path, class_name, f'{video_name}.pt')
pose_video_path = os.path.join(pose_path, class_name, f'{video_name}.pt')

# Get the data from the videos
rgb_data = VideoReader(rgb_video_path, ctx=cpu(0))
flow_data = torch.load(flow_video_path, map_location='cpu')
pose_data = torch.load(pose_video_path, map_location='cpu')

# print(flow_data.shape)
# skel = pose_data[:2,...] # (2, 300, 17, 2)
# skel[0] = (skel[0]+0.5)*319
# skel[1] = (skel[1]-0.5)*-239
# skeleton = skel[:,frame_number,:,0].transpose(1,0)

# frame = rgb_data[frame_number].asnumpy()
# print(frame.shape)

# # frame = np.zeros((240,320))
# for keypoint in skeleton:
#     if 0 in keypoint:
#         continue
#     cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 1, 255, -1)

# plt.imshow(frame, cmap='gray')
# plt.savefig('skeleton.png')

def draw_flow_arrows(vector_field, img):
    """
    Draws arrows pointing in the direction of the flow for a given vector field on an RGB image.
    
    Parameters:
    vector_field (numpy.ndarray): A numpy array of shape (2, 240, 320) representing the flow vectors.
    img (numpy.ndarray): An RGB image of shape (240, 320, 3).
    """
    u = vector_field[0]
    v = vector_field[1]
    
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(v.shape[0]))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)  # Display the RGB image
    # plt.quiver(x, y, u, v, color='r', scale=1, scale_units='xy', angles='xy')
    plt.title('Flow Field on Image')
    plt.savefig('flow.png')


draw_flow_arrows(flow_data[frame_number], rgb_data[frame_number].asnumpy())