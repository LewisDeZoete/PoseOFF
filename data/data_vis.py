import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, "..")))

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from config.argclass import ArgClass

# import decord
from decord import VideoReader, cpu
from data_gen.utils.extractors import FlowPoseSampler
import pickle
from einops import rearrange

# # ----------------- #
# class_no = 6
# video_number = 50
# frame_number = 39
# # ----------------- #

# def draw_flow_vectors(flow_data, frame, start_x, start_y, size=5):
#     """
#     Draws a group of optical flow vectors on the image.

#     Parameters:
#     flow_data (numpy.ndarray): The optical flow data.
#     frame (numpy.ndarray): The image frame.
#     start_x (int): The starting x-coordinate for the 5x5 region.
#     start_y (int): The starting y-coordinate for the 5x5 region.
#     size (int): The size of the region (default is 5).
#     """
#     # for i in range(int(-size/2), int(size/2)):
#     #     for j in range(int(-size/2), int(size/2)):
#     #         x = start_x + i
#     #         y = start_y + j
#     #         # if x < flow_data.shape[2] and y < flow_data.shape[1]:
#     #         flow_vector = flow_data[:, j, i]
#     #         end_x = int(x + flow_vector[0])
#     #         end_y = int(y + flow_vector[1])
#     #         cv2.arrowedLine(frame, (end_x, end_y), (x, y), (0, 255, 0), 1, tipLength=0.1)
#     flow = np.mean(flow_data, axis=(1,2))
#     cv2.arrowedLine(frame, (start_x, start_y), (int(start_x+flow[0]), int(start_y+flow[1])), (0, 255, 0), 1, tipLength=0.1)


# arg = ArgClass(arg='./config/custom_pose/train_joint_infogcn.yaml')
# rgb_path = arg.feeder_args['data_paths']['rgb_path']
# flow_path = arg.feeder_args['data_paths']['flow_path']
# pose_path = arg.feeder_args['data_paths']['pose_path']

# class_name = list(arg.classes.keys())[class_no]
# video_name = sorted(os.listdir(os.path.join(rgb_path, class_name)))[video_number].split('.')[0]

# print(class_name)

# # Get the paths to the videos
# rgb_video_path = os.path.join(rgb_path, class_name, f'{video_name}.avi')
# flow_video_path = os.path.join(flow_path, class_name, f'{video_name}.pt')
# pose_video_path = os.path.join(pose_path, class_name, f'{video_name}.npy')

# print(rgb_video_path)

# # Get the data from the videos
# rgb_data = VideoReader(rgb_video_path, ctx=cpu(0))
# flow_data = torch.load(flow_video_path, map_location='cpu')
# pose_data = np.load(pose_video_path)

# # Get the frame data
# window_size = 15
# arg.extractor['flowpose']['window_size'] = window_size
# flowPoseTransform = FlowPoseSampler(**arg.extractor['flowpose'])

# flow_pose = np.array(flowPoseTransform(flow_data, pose_data))
# print(flow_pose.shape)

# skel = flow_pose[:2,...] # (2, 300, 17, 2)
# skel[0] = (skel[0]+0.5)*319
# skel[1] = (skel[1]+0.5)*239
# skeleton = skel[:,frame_number,:,0].transpose(1,0)

# flows = flow_pose[3:,frame_number,:,0].transpose(1,0).reshape((17,2,window_size,window_size))

# frame = rgb_data[frame_number].asnumpy()
# print(frame.shape)

# # frame = np.zeros((240,320))
# for keypoint_number, keypoint in enumerate(skeleton):
#     if 0 in keypoint:
#         continue
#     draw_flow_vectors(flows[keypoint_number], frame, int(keypoint[0]), int(keypoint[1]), window_size)
#     cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 1, 255, -1)


# plt.imshow(frame, cmap='gray')
# plt.savefig('skeleton.png')

# ----------------------------------
# Flow visualization
# ----------------------------------


# def draw_flow_arrows(vector_field, img):
#     """
#     Draws arrows pointing in the direction of the flow for a given vector field on an RGB image.

#     Parameters:
#     vector_field (numpy.ndarray): A numpy array of shape (2, 240, 320) representing the flow vectors.
#     img (numpy.ndarray): An RGB image of shape (240, 320, 3).
#     """
#     u = vector_field[0]
#     v = vector_field[1]

#     # Create a grid of coordinates
#     x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(v.shape[0]))

#     plt.figure(figsize=(10, 10))
#     # plt.imshow(img)  # Display the RGB image
#     plt.quiver(x, y, u, v, color='r', scale=1, scale_units='xy', angles='xy')
#     plt.title('Flow Field on Image')
#     plt.savefig('flow.png')


# draw_flow_arrows(flow_data[frame_number], rgb_data[frame_number].asnumpy())


# ----------------------------------
# NTU-RGB+D Skeleton Visualization
# ----------------------------------
def draw_skeleton_on_frame(video_path, skeleton, frame_index=0):
    """
    Draws skeleton keypoints onto a frame of the video.

    Args:
        video_path (str): Path to the video file.
        skeleton (np.array): Skeleton keypoints of shape (num_frames, num_keypoints, 2).
        frame_index (int): Index of the frame to draw the skeleton on. Default is 0.

    Returns:
        np.array: The frame with the skeleton keypoints drawn.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the skeleton for corresponding frame
    skeleton_frame = skeleton[:, frame_index, ...]

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None

    # Draw the skeleton keypoints on the frame
    for person_num, person in enumerate(skeleton_frame.transpose(2, 0, 1)):
        for keypoint in person.transpose(1, 0):
            x, y = int(keypoint[0]), int(keypoint[1])
            if person_num == 0:
                cv2.circle(
                    frame, (x, y), 5, (0, 255, 0), -1
                )  # Draw a green circle for each keypoint
            else:
                cv2.circle(
                    frame, (x, y), 5, (255, 0, 0), -1
                )  # Draw a green circle for each keypoint

    # Release the video capture object
    cap.release()

    return frame


# Using video with two subjects
video_name = "S001C003P008R001A050" # A good one!
# video_name = "S001C001P001R001A026"
video_name = "S002C003P014R002A055"

# Load the names of the skeleton files
skes_name = np.loadtxt("./data/ntu/statistics/ntu_rgbd-available.txt", dtype=str)
ske_number = list(skes_name).index(video_name)

# Example usage
video_path = f"../Datasets/NTU_RGBD/nturgb+d_rgb/{video_name}_rgb.avi"
pose_path = "./data/ntu/raw_data/raw_skes_data.pkl"
pose_denoised_path = "./data/ntu/denoised_data/raw_denoised_colors.pkl"

with open(pose_path, "rb") as f:
    raw_skes_data = pickle.load(f)
with open(pose_denoised_path, "rb") as f:
    denoised_skes_data = pickle.load(f)

print(len(raw_skes_data))
print(len(denoised_skes_data))

# Get the skeleton data
for idx, bodies_data in enumerate(raw_skes_data):
    if idx == ske_number:
        ske_name = bodies_data["name"]
        num_bodies = len(bodies_data["data"])
        num_frames = bodies_data["num_frames"]

        print(f"Skeleton file name: {ske_name}")
        print(f"Number of bodies: {num_bodies}")
        print(f"Number of frames: {num_frames}")

        # # figure out steps in their denoising process
        # bodies_data, noise_info = denoising_bodies_data(bodies_data)
        bodies_data = bodies_data["data"]

        # stacked_poses = np.stack(
        #     [body_data["colors"] for body_data in bodies_data.values()], axis=-1
        # )
        for body_name, body_data in bodies_data.items():
            colors = body_data["colors"]
            print(f"Single person shape: {colors.shape}")
            print("\t(num_frames, num_keypoints, num_channels)")
            

        # print(f"Stacked poses shape: {stacked_poses.shape}")
        # print("\t(num_frames, num_keypoints, num_channels, num_people)")
        # # (channels, num_frames, num_keypoints, num_people)
        # # This is the shape my flowpose sampler takes...
        # # rearrange(stacked_poses, 't v c m -> c t v m')
        # stacked_poses = stacked_poses.transpose(2, 0, 1, 3)
        # print(f"Stacked poses shape (post transpose): {stacked_poses.shape}")
        # print("\t(num_channels, num_frames, num_keypoints, num_people)")

denoised = denoised_skes_data[ske_number]
denoised = denoised.transpose(3, 0, 2, 1)
print(denoised.shape)

# # Draw the skeleton on the frame
# frame_with_skeleton = draw_skeleton_on_frame(video_path, stacked_poses, frame_index=25)
# cv2.imwrite("skeleton_from_raw.png", frame_with_skeleton)
frame_with_skeleton = draw_skeleton_on_frame(video_path, denoised, frame_index=40)
cv2.imwrite("skeleton_from_denoised.png", frame_with_skeleton)


# flow = np.random.rand(50, 2, 240, 320)
# flowPoseSampler = FlowPoseSampler(
#     window_size=3, threshold=0.5, loop=False, norm=False, match_pose=False, ntu=True
# )

# Get the flow data
