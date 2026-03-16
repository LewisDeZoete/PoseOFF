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
    "--sample_name",
    dest="sample_name",
    default="S019C001P051R001A113",
    help="Name of a pre-computed sample (run data/visualisations/get_flow_samples.py...)"
)
parser.add_argument(
    "--live",
    dest="live",
    action="store_true",
    help="Display an ourput image live (need x11 forwarding or something)",
)
parser.add_argument(
    "--save_ims",
    dest="save_ims",
    action="store_true",
    help="Instead of live display, write all frames to `data/visualisations/flow_pose_frames/<sample_name>`",
)
parser.add_argument(
    "--crop",
    dest="crop",
    action="store_true",
    help="If passed, crops all of the images within the data/visualisations/flow_pose_frames/<sample_name> folder"
)
args = parser.parse_args()

# Optionally show live display or save to file
sample_name = args.sample_name
live = True if args.live else False
save_ims = True if args.save_ims else False
crop = True if args.crop else False

# Define data and save paths
data_root = "./data/visualisations/RAW"
save_root = "./data/visualisations/flow_pose_frames"


def load_data(data_root, sample_name):
    # Attempt to load the data
    try:
        data = np.load(osp.join(data_root, f"{sample_name}.npz"))
    except FileNotFound:
        print("File not found, may need to be generated...")
        print("Run ./data/visualisations/get_flow_samples.py to generate the data.")
        quit()

    # Put the data into a list (excluding pose for whatever reason)
    T, C, H, W = data['rgb'][1:].shape
    videos = [
        data['flow'],
        data['rgb'][1:],
        # np.ones((T, 4, H, W), dtype=np.uint8)*255,
        np.zeros((T, 4, H, W), dtype=np.uint8),
        ]
    # Change the dimensions to be channel last (H,W,C)
    for no, vid in enumerate(videos):
        videos[no] = np.transpose(vid, (0, 2, 3, 1))  # TCHW -> THWC

    # Get and reshape the pose array
    poses = rearrange(data['pose'][1:], 'T M V C -> T (M V) C')

    # Get the poseoff samples
    poseoff = data['flowpose']

    return videos, poses, poseoff


def get_frame_count(video):
    """Get the total number of frames in a video."""
    return video.shape[0]


def draw_poseoff(frame, poseoff, pose, frame_num=0, flow_scale=5.0, window_size=5, dilation=2, skip_points=[], average_flow=False):
    frame=frame.copy()
    k = window_size // 2
    poseoff_frame = rearrange(
        poseoff[:,frame_num],
        "(H W C) V M -> (M V) H W C",
        C=2, H=window_size, W=window_size
    )
    h,w,_ = frame.shape
    for keypoint_num, flow_window in enumerate(poseoff_frame):
        if keypoint_num in skip_points:
            continue
        x_pose=int(pose[keypoint_num][0])
        y_pose=int(pose[keypoint_num][1])
        if average_flow:
            # avg_window = flow_window.mean(axis=(0,1))
            # u,v = avg_window
            u,v = flow_window[2,2]

            x0 = int(round(x_pose))
            y0 = int(round(y_pose))
            x1 = int(round(x0 + u*flow_scale))
            y1 = int(round(y0 + v*flow_scale))
            cv2.arrowedLine(
                frame,
                (x0, y0),
                (x1, y1),
                # color=(0, 0, 0) if frame.shape[-1] == 3 else (0, 0, 0, 255),
                color=(0, 255, 0) if frame.shape[-1] == 3 else (0, 255, 0, 255),
                thickness=3,
                tipLength=0.6
            )
        else:
            for col_num, col in enumerate(flow_window):
                for row_num, vector in enumerate(col):
                    u,v = vector

                    # Window offset with dilation
                    dx = (col_num - k) * dilation
                    dy = (row_num - k) * dilation

                    x0 = int(round(x_pose + dx))
                    y0 = int(round(y_pose + dy))

                    # Scale flow for visibility
                    x1 = int(round(x0 + u * flow_scale))
                    y1 = int(round(y0 + v * flow_scale))

                    # Check if in image bounds
                    if (
                            0 <= x0 < w and 0 <= y0 < h and
                            0 <= x1 < w and 0 <= y1 < h
                    ):
                        cv2.arrowedLine(
                            frame,
                            (x0, y0),
                            (x1, y1),
                            color=(0, 255, 0) if frame.shape[-1] == 3 else (0, 255, 0, 255),
                            thickness=1,
                            tipLength=0.4
                        )
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


def draw_bones(frame, pose, person_num=None): 
    frame = frame.copy()
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


def draw_optical_flow_arrows(flow, frame=None, step=16, scale=1.0, color=(0, 0, 0), thickness=1, threshold=2.0):
    """
    Draws optical flow vectors as arrows on a frame.

    Args:
        flow (np.ndarray): Optical flow of shape (2, H, W) — (u,v).
        frame (np.ndarray or None): Background image to draw on (H, W, 3). If None, a blank canvas is used.
        step (int): Sampling step for arrows. Larger = fewer arrows.
        scale (float): Scale multiplier for flow vectors (default=1.0)
        color (tuple): BGR color for the arrows (default=(0,0,0))
        thickness (int): Width of drawn optical flow arrows (default=1.0)
        threshold (float): Magnitude threshold below which arrows will not be drawn (default=0.0)
        

    Returns:
        np.ndarray: Image with arrows drawn.
    """
    u, v = flow[:, :, 0], flow[:, :, 1]
    H, W = u.shape

    if frame is None:
        vis = np.ones((H, W, 3), dtype=np.uint8)*255
    else:
        vis = frame.copy()

    # Create grid of points
    y, x = np.mgrid[step//2:H:step, step//2:W:step].astype(np.int32)
    fx, fy = u[y, x], v[y, x]

    # Draw arrows
    for (x1, y1, dx, dy) in zip(x.ravel(), y.ravel(), fx.ravel(), fy.ravel()):
        x2 = int(x1 + scale * dx)
        y2 = int(y1 + scale * dy)
        mag = ((x2-x1)**2+(y2-y1)**2)**(0.5)
        if ((x2-x1)**2+(y2-y1)**2)**(0.5) < threshold:
            continue
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, thickness, tipLength=0.1)

    return vis


def display_videos(data_root, sample_name, save_path="./data/visualisations/flow_pose_frames"):
    print("Loading data...")
    videos, poses, poseoff = load_data(data_root, sample_name)

    print("Viewport opening...")
    # Initialize variables
    current_video_index = 0
    current_frame_index = 0
    show_skel = False
    flow_windows = False
    while True:
        # Get the current frame from the current video
        frame = videos[current_video_index][current_frame_index]

        # If the video is not RGB (e.g., flow or pose), normalize and convert to RGB
        if frame.shape[-1] == 2:  # Channels-first format
            frame = draw_optical_flow_arrows(frame, step=16, scale=3, thickness=2, threshold=0.0)
        elif frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # If show_skel is toggled on, display skeleton
        if show_skel:
            frame = draw_bones(frame, pose=poses[current_frame_index]) 
            frame = draw_skel(frame, pose=poses[current_frame_index])

        # Show flow windows
        if flow_windows:
            frame = draw_bones(frame, pose=poses[current_frame_index])
            frame = draw_poseoff(frame, pose=poses[current_frame_index], poseoff=poseoff, frame_num=current_frame_index, flow_scale=5.0, window_size=5, dilation=2, average_flow=True)

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
                    f"{ {0:'FLOW', 1:'RGB', 2:'POSE'}[current_video_index] if not flow_windows else 'PoseOFF'}"\
                    f"-frame-{current_frame_index}.png",
                    ),
                frame,
            )
        elif key == ord("f"): # Cycle flow windows on frame, or blank background
            flow_windows = not flow_windows


    cv2.destroyAllWindows()


def save_im_frames(data_root, sample_name, save_path="./data/visualisations/flow_pose_frames"):
    print("Loading data...")
    videos, poses, poseoff = load_data(data_root, sample_name)

    total_frames = get_frame_count(videos[0]) - 1
    print(f"Writing {total_frames} frames...")
    for frame_num in range(total_frames):
        # Create flow frame
        flow_frame = videos[0][frame_num]
        flow_frame = draw_optical_flow_arrows(flow_frame, step=16, scale=3, thickness=2, threshold=0.0)
        
        # Save rgb frame
        rgb_frame = videos[1][frame_num]
        rgb_frame = cv2.cvtColor(rgb_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # Draw and save poses
        pose = poses[frame_num]
        pose_frame = videos[2][frame_num]
        pose_frame = draw_bones(pose_frame, pose)
        pose_frame = draw_skel(pose_frame, pose, debug=False)

        for frame_type, frame in zip(["FLOW", "RGB", "POSE"], [flow_frame, rgb_frame, pose_frame]):
            cv2.imwrite(
                osp.join(
                    save_path,
                    f"{frame_type}-frame-{frame_num}.png"
                    ),
                frame,
            )



def crop_large_imgs(x_origin=250, y_origin=150, size=900, sample_name=""):
    original_img_path = f"/fred/oz141/ldezoete/MS-G3D/data/visualisations/flow_pose_frames/{sample_name}"
    cropped_img_path = f"/fred/oz141/ldezoete/MS-G3D/data/visualisations/flow_pose_frames/{sample_name}/cropped"

    # Make the cropped image path if it doesn't exist
    os.makedirs(cropped_img_path, exist_ok=True)
    
    img_names = os.listdir(original_img_path)
    for img_name in img_names:
        if not osp.isfile(osp.join(original_img_path, img_name)):
            continue
        print(f"Cropping: {img_name}")
        # if not img_name[:4] == "POSE": # Don't crop pose diagrams drawn with matplotlib...
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


def quick_view(img):
    frame = img.copy()
    while True:
        cv2.imshow("Video player", frame.astype(np.uint8))
        key=cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
            

if __name__ == '__main__':
    save_path = osp.join(save_root, sample_name)
    os.makedirs(save_path, exist_ok=True)
    if crop:
        crop_large_imgs(
            x_origin=440,
            y_origin=380,
            size=700,
            sample_name=sample_name
        )
        quit()
    if live:
        display_videos(data_root, sample_name, save_path=save_path)
        quit()
    if save_ims:
        save_im_frames(data_root, sample_name, save_path=save_path)
        quit()

    # Get data
    # pose; (53, 50, 2)
    # poseoff: (50, 53, 25, 2)
    videos, poses, poseoff = load_data(data_root, sample_name)

    frame_num = 17

    flow_frame = videos[0][frame_num]

    rgb_frame = videos[1][frame_num]
    rgb_frame = cv2.cvtColor(rgb_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
    H,W,C = rgb_frame.shape

    pose = poses[frame_num]
    # # Slight pose adjustments...
    # pose[11, 0] = pose[11, 0]+5

    # pose[10, 1] = pose[10, 1]+10
    # pose[11, 1] = pose[11, 1]+15 # palm
    # pose[23, 1] = pose[23, 1]+15 # Fingers
    # pose[24, 1] = pose[24, 1]+5 # Thumb
    # pose[22, 1] = pose[22, 1]-10

    # Introduce keypoints to skip drawing PoseOFF windows for...
    skip_points=[1, 2, 21, 9, 11, 12, 24, 25]
    skip_points = [22,23,24,25]
    skip_points = [point-1 for point in skip_points]
    skip_points = skip_points + [point+25 for point in skip_points]


    background_frame = np.zeros((1080, 1920, 4))
    poseoff_frame = draw_bones(background_frame, pose=poses[frame_num])
    poseoff_frame = draw_poseoff(
        poseoff_frame,
        pose=poses[frame_num],
        poseoff=poseoff,
        frame_num=frame_num,
        flow_scale=5.0,
        window_size=5,
        dilation=15,
        skip_points=skip_points,
        average_flow=True
    )
    quick_view(poseoff_frame)
    cv2.imwrite(
        f'data/visualisations/flow_pose_frames/{sample_name}/PoseOFF-frame-{frame_num}.png',
        poseoff_frame
    )

    quit()

    # # Crop into one particular joint
    # cropped_flow = crop_to_joint(
    #     videos=None,
    #     frame=poseoff_frame,
    #     poses=poses,
    #     frame_num=frame_num,
    #     video_num=0,
    #     person_num=0,
    #     joint_num=9,
    #     window_size=120
    # )
    # cropped_flow = cv2.resize(cropped_flow, (cropped_flow.shape[1]*2,cropped_flow.shape[0]*2))
    # quick_view(cropped_flow)
    # cv2.imwrite(
    #     f'data/visualisations/flow_pose_frames/{sample_name}/cropped/closeup-{frame_num}.png',
    #     cropped_flow
    # )
