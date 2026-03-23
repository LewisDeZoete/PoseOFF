#!/usr/bin/env python3
import numpy as np
import cv2

def write_videos(videos):
    size = (videos[0].shape[2], videos[0].shape[1])
    flow_video = cv2.VideoWriter('data/visualisations/videos/flow_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
    flow_skel_video = cv2.VideoWriter('data/visualisations/videos/flow_skels_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)

    RGB_video = cv2.VideoWriter('data/visualisations/videos/RGB_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
    RGB_skel_video = cv2.VideoWriter('data/visualisations/videos/RGB_skels_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
    
    poseoff_video = cv2.VideoWriter('data/visualisations/videos/poseoff_video.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    current_frame_index = 0
    poseoff = poseoff_sample(flow_full, poses, window_size=19) # T, MV, C
    while True:
        # Get the flow and RGB frames
        flow_frame = videos[0][current_frame_index]
        RGB_frame = videos[1][current_frame_index]

        # frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        flow_frame = flow2image(flow_frame)
        RGB_frame = cv2.cvtColor(RGB_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # Write normal videos
        flow_video.write(flow_frame.astype(np.uint8))
        RGB_video.write(RGB_frame.astype(np.uint8))

        # Draw skeletons
        flow_skel_frame = draw_skel(flow_frame, pose=poses[current_frame_index])
        RGB_skel_frame = draw_skel(RGB_frame, pose=poses[current_frame_index])

        # Write skeleton videos
        flow_skel_video.write(flow_skel_frame.astype(np.uint8))
        RGB_skel_video.write(RGB_skel_frame.astype(np.uint8))

        # Draw poseoff
        poseoff_frame = draw_poseoff(poseoff_frame=poseoff[current_frame_index],
                                       pose_frame=poses[current_frame_index],
                                       frame=flow_frame,
                                       window_size=19)
        poseoff_video.write(poseoff_frame.astype(np.uint8))
        
        current_frame_index += 1
        if current_frame_index >= get_frame_count(videos[0]):
            break
    
    flow_video.release()
    flow_skel_video.release()
    RGB_video.release()
    RGB_skel_video.release()
    poseoff_video.release()


def poseoff_sample(flows, poses, window_size=5):
    """
    Samples the optical flow in windows surrounding the pose keypoints.
    flows: (T C H W)
    poses: (T (M V) C)
    Returns array of shape:
        ((window_size**2)*2,
        frames,
        keypoints,
        num_people)
    """
    half_k=window_size//2

    # Get the shape of the input tensors
    num_frames, _, height, width = flows.shape
    num_pose_frames, keypoints, channels = poses.shape

    # The keypoints are already scaled
    pose_points = np.nan_to_num(poses, nan=0).astype(int)

    # Prepare tensor to stack flow windows
    stacker = np.zeros((num_pose_frames, keypoints, window_size**2*2,))
    print(stacker.shape)

    # Create a grid of valid indices (filter out points close to the image border)
    valid_indices = ((pose_points[:, :, 0] >= half_k) & (pose_points[:, :, 0] < width - half_k) &
                        (pose_points[:, :, 1] >= half_k) & (pose_points[:, :, 1] < height - half_k))

    # Loop through the frames and sample flow in window around each valid keypoint
    for frame_no, flow in enumerate(flows):
        for keypoint_num in range(keypoints):
            if valid_indices[frame_no, keypoint_num]:
                x, y = pose_points[frame_no, keypoint_num, 0], pose_points[frame_no, keypoint_num, 1]
                # Get the window of optical flow
                flow_window = flow[:, y - half_k : y + half_k + 1, x - half_k : x + half_k + 1]


                stacker[frame_no, keypoint_num, :] = flow_window.flatten()
    return stacker

