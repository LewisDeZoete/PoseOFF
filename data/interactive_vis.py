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
parser.add_argument(
    "--live",
    dest="live",
    action="store_true",
    help="extracted size - small or full (default=small)",
)
args = parser.parse_args()

assert args.size in ['small', 'full'], '--size argument must be either `small` or `full`'
# Optionally show live display or save to file
live = True if args.live else False

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


def draw_flowpose(flowpose_frame, pose_frame, frame, window_size=5):
    k = window_size // 2
    flowpose_frame = rearrange(flowpose_frame, 'MV (C H W) -> MV H W C', C=2, H=window_size, W=window_size)
    for keypoint_num, flow_window in enumerate(flowpose_frame):
        img = flow2image(flow_window)
        centre = [int(pose_frame[keypoint_num][1]), 
                  int(pose_frame[keypoint_num][0])]
        frame[centre[0]-k:centre[0]+k+1, centre[1]-k:centre[1]+k+1] = img
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
    frame = frame.copy()
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
    return frame
        

def flow_pose_sample(flows, poses, window_size=5):
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

def generate_sample_points(center, dilation=1):
    """
    Generate a 5x5 grid of points around a center coordinate with a given dilation.
    
    Args:
        center (tuple): The center coordinate (x, y).
        dilation (int): The spacing between points in the grid.
    
    Returns:
        numpy.ndarray: An array of shape (25, 2) containing the coordinates of the points.
    """
    x, y = center
    offsets = np.arange(-2, 3) * dilation  # Generate offsets: [-2, -1, 0, 1, 2] scaled by dilation
    grid_x, grid_y = np.meshgrid(offsets, offsets)  # Create a 5x5 grid of offsets
    points = np.stack([grid_x.ravel() + x, grid_y.ravel() + y], axis=-1)  # Add offsets to center
    return points


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
    
    flowpose_video = cv2.VideoWriter('data/visualisations/videos/flowpose_video.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    current_frame_index = 0
    flowpose = flow_pose_sample(flow_full, poses, window_size=19) # T, MV, C
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

        # Draw flowpose
        flowpose_frame = draw_flowpose(flowpose_frame=flowpose[current_frame_index],
                                       pose_frame=poses[current_frame_index],
                                       frame=flow_frame,
                                       window_size=19)
        flowpose_video.write(flowpose_frame.astype(np.uint8))
        
        current_frame_index += 1
        if current_frame_index >= get_frame_count(videos[0]):
            break
    
    flow_video.release()
    flow_skel_video.release()
    RGB_video.release()
    RGB_skel_video.release()
    flowpose_video.release()



if live:
    display_videos(videos)
else:
    write_videos(videos)