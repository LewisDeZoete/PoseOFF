# ----------------------------------
# NTU-RGB+D Flow visualisation
# NOTE: This visualiser requires *ALL VIDEOS TO BE PROCESSED AHEAD OF TIME*.
# These are defined in the section just below this block.
# ----------------------------------

import os
import os.path as osp
import numpy as np
from einops import rearrange
import cv2
import argparse
from itertools import cycle
import matplotlib.pyplot as plt


trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]

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
    help="Display an ourput image live (need x11 forwarding or something)",
)
parser.add_argument(
    "--crop",
    dest="crop",
    action="store_true",
    help="If passed, crops all of the images within the data/visualisations/flow_pose_frames folder"
)
args = parser.parse_args()

assert args.size in ['small', 'full'], '--size argument must be either `small` or `full`'
# Optionally show live display or save to file
live = True if args.live else False
crop = True if args.crop else False

# Root path for data
root_path = "./data/visualisations/RAW"

# Load flow, rgb and poses
flow_full = np.load(osp.join(root_path, "FLOW_full-S001C003P008R001A050.npy"))
flow_small = np.load(osp.join(root_path, "FLOW_small-S001C003P008R001A050.npy"))

# (T, C, H, W) - (T, 3, 240/1080, 320/1920)
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
    T, C, H, W = rgb_full.shape
    videos = [
        flow_full,
        rgb_full,
        np.zeros((T, 4, H, W), dtype=np.uint8)
    ]
else:
    T, C, H, W = rgb_small.shape
    videos = [
        flow_small,
        rgb_small,
        np.zeros((T, 4, H, W), dtype=np.uint8)
    ]

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


def draw_skel(frame, pose, person_num=None, skip_points=[], debug=False):  # Poses shape: (M V) C
    frame = frame.copy()
    pose_local = pose.copy()
    pose_local = rearrange(pose_local, '(M V) C -> M V C', M=2, V=25)
    if person_num != None: # If a person_num is passed, only get that specific body!
        pose_local = (pose_local[person_num]).reshape((1, 25, 2))
    

    inner_circ_params = {
        "radius": 5,
        "color": (0, 0, 255) if frame.shape[-1] == 3 else (0, 0, 255, 255),
        "thickness": -1
    }
    outer_circ_params = {
        "radius": 6,
        "color": (255, 0, 0) if frame.shape[-1] == 3 else (255, 0, 0, 255),
        "thickness": 3
    }
    if frame.shape[1] < 500:
        pose_local[:, 0] = pose_local[:, 0] * (319 / 1919)
        pose_local[:, 1] = pose_local[:, 1] * (239 / 1079)
        circ_params = {"radius": 2, "color": (0, 0, 255), "thickness": 2}
    # Draw the skeleton keypoints on the frame
    for person in pose_local:
        for keypoint_num, keypoint in enumerate(person):
            if 0 in keypoint: 
                continue
            if keypoint_num in skip_points:
                continue

            # Draw circle fill first, then the outer circle in blue
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), **inner_circ_params)
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), **outer_circ_params)

            if debug:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,str(keypoint_num), (int(keypoint[0]), int(keypoint[1])), font, 0.5,(255,255,255),2,cv2.LINE_AA)
    return frame


def mpl_draw_skel(videos, poses, frame_num, video_num=2, show_frame=False):
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    
    poses = rearrange(poses, 'T (M V) C -> T M V C', M=2, V=25)
    pose_frame = poses[frame_num]
    frame = videos[video_num][frame_num]
    ax.imshow(frame)
    for pose in pose_frame:
        for part in body:
            x = pose[part, 0]
            y = pose[part, 1]
            ax.plot(x, y, color='b', marker='o', markerfacecolor='r')
    ax.set_xlim(250, 1150)
    ax.set_ylim(150, 1050)
    plt.gca().invert_yaxis()
    if show_frame:
        plt.show()    
    else: plt.savefig(f'./data/visualisations/flow_pose_frames/POSE-frame-{frame_num}.png', transparent=True)


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


def draw_optical_flow_arrows(flow, frame=None, step=16, scale=1, color=(0, 255, 0)):
    """
    Draws optical flow vectors as arrows on a frame.

    Args:
        flow (np.ndarray): Optical flow of shape (2, H, W) — (u,v).
        frame (np.ndarray or None): Background image to draw on (H, W, 3). If None, a blank canvas is used.
        step (int): Sampling step for arrows. Larger = fewer arrows.
        scale (float): Scale multiplier for flow vectors.
        color (tuple): BGR color for the arrows.

    Returns:
        np.ndarray: Image with arrows drawn.
    """
    u, v = flow[:, :, 0], flow[:, :, 1]
    H, W = u.shape

    if frame is None:
        vis = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        vis = frame.copy()

    # Create grid of points
    y, x = np.mgrid[step//2:H:step, step//2:W:step].astype(np.int32)
    fx, fy = u[y, x], v[y, x]

    # Draw arrows
    for (x1, y1, dx, dy) in zip(x.ravel(), y.ravel(), fx.ravel(), fy.ravel()):
        x2 = int(x1 + scale * dx)
        y2 = int(y1 + scale * dy)
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 1, tipLength=0.3)

    return vis

def display_videos(videos, save_path="./data/visualisations/flow_pose_frames"):
    print("Viewport opening...")
    # Initialize variables
    current_video_index = 0
    current_frame_index = 0
    show_skel = False
    cc = cycle(range(3))
    flow_windows = next(cc)
    while True:
        # Get the current frame from the current video
        frame = videos[current_video_index][current_frame_index]

        # If the video is not RGB (e.g., flow or pose), normalize and convert to RGB
        if frame.shape[-1] == 2:  # Channels-first format
            # frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            # frame = flow2image(frame)
            frame = draw_optical_flow_arrows(frame, step=20, scale=3)
        elif frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # If show_skel is toggled on, display skeleton
        if show_skel:
            frame = draw_skel(frame, pose=poses[current_frame_index])

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
                print(f"Current frame: {current_frame_index}")
        elif key == ord("j"):  # Previous frame
            if current_frame_index > 0:
                current_frame_index -= 1
                print(f"Current frame: {current_frame_index}")
        elif key == ord("i"):  # Switch video
            current_video_index = (current_video_index + 1) % len(videos)
            print(f"Switched to video {current_video_index + 1}")
        elif key == ord("k"): # Show skeleton
            show_skel = not show_skel
        elif key == ord("s"): # Save frame
            cv2.imwrite(
                osp.join(
                    save_path,
                    f"{ {0:'FLOW', 1:'RGB', 2:'POSE'}[current_video_index] }"\
                    f"-frame-{current_frame_index}.png",
                    ),
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


def crop_large_imgs(x_origin=250, y_origin=150, size=900):
    original_img_path = "/fred/oz141/ldezoete/MS-G3D/data/visualisations/flow_pose_frames/"
    cropped_img_path = "/fred/oz141/ldezoete/MS-G3D/data/visualisations/flow_pose_frames/cropped"
    
    img_names = os.listdir(original_img_path)
    for img_name in img_names:
        if not osp.isfile(osp.join(original_img_path, img_name)):
            continue
        if not img_name[:4] == "POSE": # Don't crop pose diagrams drawn with matplotlib...
            img_in_path = osp.join(original_img_path, img_name)
            img_out_path = osp.join(original_img_path, 'cropped', img_name)

            # Read the image
            img = cv2.imread(img_in_path, cv2.IMREAD_UNCHANGED)
            if img_name[:2] == "CV":
                print(img.shape)
            cropped = img[y_origin:y_origin+size, x_origin:x_origin+size]

            cv2.imwrite(img_out_path, cropped)


def crop_to_joint(
        videos=None,
        frame=None,
        poses=None,
        frame_num=0,
        video_num=2,
        person_num=1,
        joint_num=24,
        window_size=15
):
    poses = rearrange(poses, 'T (M V) C -> T M V C', M=2, V=25)
    pose_frame = poses[frame_num]
    if videos is not None:
        frame = videos[video_num][frame_num]
    elif frame is not None:
        frame = frame
    else:
        print('Must input either videos list or a single frame')
        quit()

    x_centre = int(pose_frame[person_num, joint_num, 0])
    y_centre = int(pose_frame[person_num, joint_num, 1])

    cropped = frame[
        y_centre-window_size//2:y_centre+window_size//2+1,
        x_centre-window_size//2:x_centre+window_size//2+1
    ]
    return cropped


def draw_bones(pose, frame, person_num=None): 
    pose = rearrange(pose, '(M V) C -> M V C', M=2, V=25)
    joint_connections = [
        [0,1], [1,2], [2,3], [1,20], [0,12], # Trunk/head
        [0,16], [12,13], [13,14], [14,15], # Right leg
        [0,16], [16,17],[17,18], [18,19],  # Left leg
        [20,4], [4,5], [5,6], [6,7], [7,21], [7,22], # Right arm
        [20,8], [8,9], [9,10], [10,11], [11,23], [11,24], # Left arm
    ]
    # Check if alpha channel exists in frame
    color = (255,0,0) if frame.shape[-1] == 3 else (255,0,0,255)
    # Get individual person's specific pose (if person_num specified)
    if person_num in [0,1]:
        pose = pose[person_num].reshape((1, 25, 2))

    for person in pose:
        for joint_connection in joint_connections:
            p1, p2 = joint_connection
            cv2.line(frame,
                    (int(person[p1,0]), int(person[p1,1])),
                    (int(person[p2,0]), int(person[p2,1])),
                    color, 3
                    )
    return frame
            

if __name__ == '__main__':
    if crop:
        crop_large_imgs()
        # frame_numbers = [i for i in range(30,40)]
        # print(frame_numbers)
        # for frame_num in frame_numbers:
        #     mpl_draw_skel(videos, poses, frame_num, video_num=2, show_frame=False)
    # if live:
    #     display_videos(videos)
    # else:
    #     write_videos(videos)


    # # Display a single image...
    # original_img_path = "/fred/oz141/ldezoete/MS-G3D/data/visualisations/flow_pose_frames/"
    # im_names = os.listdir(original_img_path)
    # rgb_imgs = [name for name in im_names if "RBD" in name]
    # img = cv2.imread(osp.join(original_img_path, rgb_imgs[0]))
    # cv2.imwrite('data/visualisations/flow_pose_frames/TMP.png', img)

    frame_num = 38

    flow_frame = videos[0][frame_num]

    rgb_frame = videos[1][frame_num]
    rgb_frame = cv2.cvtColor(rgb_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
    H,W,C = rgb_frame.shape


    pose = poses[frame_num]
    pose[11, 0] = pose[11, 0]-50
    pose[23, 0] = pose[23, 0]-50
    pose[24, 0] = pose[24, 0]-50

    pose[11, 1] = pose[11, 1]-25
    pose[23, 1] = pose[23, 1]-30
    pose[24, 1] = pose[24, 1]-20


    # First, draw the flowpose arrows on the full frame (may take time)
    frame = draw_optical_flow_arrows(flow_frame, frame=rgb_frame, step=10, scale=1, color=(0, 255, 0))
    full_frame = draw_optical_flow_arrows(flow_frame, step=20, scale=3)
    frame_transparent = np.zeros((H, W, 4), dtype=np.uint8)

    # Then, we draw the skeleton lines we want
    frame = draw_bones(pose, frame)
    full_frame = draw_bones(pose, full_frame)
    frame_transparent = draw_bones(pose, frame_transparent)

    # Finally, draw the skeleton keypoints themselves
    frame = draw_skel(frame, pose, person_num=0, skip_points=[4,5], debug=False)
    full_frame = draw_skel(full_frame, pose, debug=False)
    frame_transparent = draw_skel(frame_transparent, pose, debug=False)


    cropped_flow = crop_to_joint(
        videos=None,
        frame=frame,
        poses=poses,
        frame_num=frame_num,
        video_num=0,
        person_num=0,
        joint_num=11,
        window_size=150
    )
    cropped_flow = cv2.resize(cropped_flow, (cropped_flow.shape[1]*2,cropped_flow.shape[0]*2))

    cv2.imwrite('data/visualisations/flow_pose_frames/TMP.png', cropped_flow)
    cv2.imwrite(f'data/visualisations/flow_pose_frames/FLOWPOSE-frame-{frame_num}.png', full_frame)
    cv2.imwrite(f'data/visualisations/flow_pose_frames/CVPose-frame-{frame_num}.png', frame_transparent)

    cv2.imshow('cropped', frame_transparent)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
