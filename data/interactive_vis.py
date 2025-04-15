# ----------------------------------
# NTU-RGB+D Flow visualisation
# ----------------------------------

import os.path as osp
import numpy as np
from einops import rearrange
import cv2
import argparse
from itertools import cycle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--size",
    dest="size",
    default="small",
    help="extracted size - small or full (default=small)",
)
args = parser.parse_args()

assert args.size in ['small', 'full'], '--size argument must be either `small` or `full`'

# Root path for data
root_path = "./data/visualisations/RAW"

# Load flow, rgb and poses
flow_full = np.load(osp.join(root_path, "FLOW_full-S001C003P008R001A050.npy"))
flow_small = np.load(osp.join(root_path, "FLOW_small-S001C003P008R001A050.npy"))

rgb_full = np.load(osp.join(root_path, "RGB_full-S001C003P008R001A050.npy"))[1:]
rgb_small = np.load(osp.join(root_path, "RGB_small-S001C003P008R001A050.npy"))[1:]

poses = np.load(osp.join(root_path, "POSE_small-S001C003P008R001A050.npy"))
poses = poses[:, 1:]  # C, T-1, V, M
C, T, V, M = poses.shape
poses = rearrange(poses, "C T V M -> T (M V) C", C=C, T=T, V=V, M=M)

flowpose = np.load(osp.join(root_path, 'flowpose_full_data.npy')) # C, T, V, M
flowpose = rearrange(flowpose, "C T V M -> T (M V) C", C=52, T=T, V=V, M=M)

# Depending on argument, use either small or full videos
if args.size == 'full':
    videos = [flow_full, rgb_full]
else:
    videos = [flow_small, rgb_small]

for no, vid in enumerate(videos):
    videos[no] = np.transpose(vid, (0, 2, 3, 1))  # TCHW -> THWC

def get_frame_count(video):
    """Get the total number of frames in a video."""
    return video.shape[0]


def draw_flowpose(flowpose_frame, frame):
    for keypoint in flowpose_frame:
        flow_window = rearrange(keypoint[2:], '(C H W) -> H W C', C=2, H=5, W=5)
        img = flow2image(flow_window)
        centre = [int((keypoint[1]+0.5)*frame.shape[0]),
                  int((keypoint[0]+0.5)*frame.shape[1])]
        frame[centre[0]-2:centre[0]+3, centre[1]-2:centre[1]+3] = img
    return frame


def flow2image(flow_frame):
    x, y = flow_frame[:, :, 0], flow_frame[:, :, 1]
    hsv = np.zeros((flow_frame.shape[0], flow_frame.shape[1], 3), dtype=np.uint8)
    ma, an = cv2.cartToPolar(x, y, angleInDegrees=True)
    hsv[..., 0] = (an / 2).astype(np.uint8)
    hsv[..., 1] = (
        cv2.normalize(ma, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    ).astype(np.uint8)
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def draw_skel(frame, pose):  # Poses shape: (M V) C
    pose_local = pose.copy()
    circ_params = {"radius": 5, "color": (0, 0, 255), "thickness": 4}
    if frame.shape[1] < 500:
        pose_local[:, 0] = pose_local[:, 0] * (319 / 1919)
        pose_local[:, 1] = pose_local[:, 1] * (239 / 1079)
        circ_params = {"radius": 2, "color": (0, 0, 255), "thickness": 2}
    # Draw the skeleton keypoints on the frame
    for keypoint in pose_local:
        if 0 in keypoint:
            continue
        cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), **circ_params)
        

def flow_pose_sample(flows, poses, window_size=5):
    """
    Samples the optical flow in windows surrounding the pose keypoints.
    Returns array of shape:
        (num_pose_channels+(window_size**2)*2,
        frames, 
        keypoints, 
        num_people)
    """
    half_k=window_size//2
    # Remove first frame of poses (no flow data)
    poses = poses[:, 1:, :, :]

    # Get the shape of the input tensors
    num_frames, _, height, width = flows.shape
    channels, num_pose_frames, num_keypoints, num_people = poses.shape
    total_keypoints = num_keypoints*num_people

    # Scale pose keypoints to image size
    pose_points = np.nan_to_num(poses, nan=0)
    pose_points = (pose_points.reshape(2, num_pose_frames, total_keypoints)
                * np.array([(width - 1)/1920, (height - 1)/1080]).reshape(2, 1, 1)).astype(int)
    # NTU keypoints need to be scaled [-0.5, 0.5]
    poses[0] = poses[0]/1920-0.5
    poses[1] = poses[1]/1080-0.5

    # Prepare tensor to stack flow windows
    stacker = np.zeros((window_size**2*2, num_pose_frames, total_keypoints))

    # Create a grid of valid indices (filter out points close to the image border)
    valid_indices = ((pose_points[0, :, :] >= half_k) & (pose_points[0, :, :] < width - half_k) & 
                        (pose_points[1, :, :] >= half_k) & (pose_points[1, :, :] < height - half_k))
                    
    # Loop through the frames and sample flow in window around each valid keypoint
    for frame_no, flow in enumerate(flows):
        for keypoint_num in range(total_keypoints):
            if valid_indices[frame_no, keypoint_num]:
                x, y = pose_points[0, frame_no, keypoint_num], pose_points[1, frame_no, keypoint_num]
                # Get the window of optical flow
                flow_window = flow[:, y - half_k : y + half_k + 1, x - half_k : x + half_k + 1]
                
                stacker[:, frame_no, keypoint_num] = flow_window.flatten()
    
    # NOTE: NTU does not return keypoint x,y,z coordinates
    flow_pose = stacker.reshape(stacker.shape[0], *poses.shape[1:])
    
    return flow_pose


def display_videos(videos):
    print("Viewport opening...")
    # Initialize variables
    current_video_index = 0
    current_frame_index = 0
    show_skel = False
    cc = cycle(range(3))
    flow_windows = next(cc)
    LIMB_NO = 25
    while True:
        print(f'Position: ({(flowpose[current_frame_index, LIMB_NO, 0]+0.5)*1080:.1f}, '
              f'{(flowpose[current_frame_index, LIMB_NO, 1]+0.5)*1920:.1f})')
        print(f'Flow: ({flowpose[current_frame_index, LIMB_NO, 2:27].mean():.4f}, '
              f'{flowpose[current_frame_index, LIMB_NO, 27:].mean():.4f})\n')
        # Get the current frame from the current video
        frame = videos[current_video_index][current_frame_index]

        # If the video is not RGB (e.g., flow or pose), normalize and convert to RGB
        if frame.shape[-1] != 3:  # Channels-first format
            # frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            frame = flow2image(frame)
        else:
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # If show_skel is toggled on, display skeleton
        if show_skel:
            draw_skel(frame, pose=poses[current_frame_index])
        
        # Show flow windows
        if flow_windows == 1:
            draw_flowpose(flowpose_frame=flowpose[current_frame_index], frame=frame)
        if flow_windows == 2:
            frame = draw_flowpose(flowpose_frame=flowpose[current_frame_index], frame=np.zeros((frame.shape)))

        # Display the frame
        cv2.imshow("Video Player", frame.astype(np.uint8))
        key = cv2.waitKey(0) & 0xFF  # Wait for a key press

        if key == ord("q"):  # Quit
            break
        elif key == ord("l"):  # Next frame
            if current_frame_index < get_frame_count(videos[current_video_index]) - 1:
                current_frame_index += 1
        elif key == ord("j"):  # Previous frame
            if current_frame_index > 0:
                current_frame_index -= 1
        elif key == ord("i"):  # Switch video
            current_video_index = (current_video_index + 1) % len(videos)
            print(f"Switched to video {current_video_index + 1}")
        elif key == ord("k"): # Show skeleton
            show_skel = not show_skel
        elif key == ord("s"): # Save frame
            cv2.imwrite(
                f"{'FLOW' if current_video_index == 0 else 'RGB'}-frame-{current_frame_index}.png",
                frame,
            )
        elif key == ord("f"): # Cycle flow windows on frame, or blank background
            flow_windows = next(cc)


    cv2.destroyAllWindows()


display_videos(videos)