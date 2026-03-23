import os.path as osp
import matplotlib.pyplot as plt
# import torch
import numpy as np
import cv2
# from config.argclass import ArgClass

# # import decord
# from decord import VideoReader, cpu
# from data_gen.utils.extractors import PoseOFFSampler
import pickle
from einops import rearrange

# # ----------------- #
# class_no = 6
# video_number = 50
# frame_number = 39
# # ----------------- #

# ----------------------------------
# UCF-101 PoseOFF Visualisation
# ----------------------------------

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


# arg = ArgClass(arg='./config/ucf101/train_base.yaml')
# rgb_path = arg.feeder_args['data_paths']['rgb_path']
# flow_path = arg.feeder_args['data_paths']['flow_path']
# pose_path = arg.feeder_args['data_paths']['pose_path']

# class_name = list(arg.classes.keys())[class_no]
# video_name = sorted(os.listdir(osp.join(rgb_path, class_name)))[video_number].split('.')[0]

# print(class_name)

# # Get the paths to the videos
# rgb_video_path = osp.join(rgb_path, class_name, f'{video_name}.avi')
# flow_video_path = osp.join(flow_path, class_name, f'{video_name}.pt')
# pose_video_path = osp.join(pose_path, class_name, f'{video_name}.npy')

# print(rgb_video_path)

# # Get the data from the videos
# rgb_data = VideoReader(rgb_video_path, ctx=cpu(0))
# flow_data = torch.load(flow_video_path, map_location='cpu')
# pose_data = np.load(pose_video_path)

# # Get the frame data
# window_size = 15
# arg.extractor['poseoff']['window_size'] = window_size
# poseOFFTransform = PoseOFFSampler(**arg.extractor['poseoff'])

# flow_pose = np.array(poseOFFTransform(flow_data, pose_data))
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
def draw_flow_windows(frame, flows, poses, frame_index=0):
    """
    Draws a group of optical flow vectors on the image.

    Parameters:
    flow_data (numpy.ndarray): The optical flow data.
    frame (numpy.ndarray): The image frame.
    start_x (int): The starting x-coordinate for the 5x5 region.
    start_y (int): The starting y-coordinate for the 5x5 region.
    size (int): The size of the region (default is 5).
    """
    skeleton_frame = poses[:, frame_index] # C V M
    flows = rearrange(flows[:, frame_index], 'C V M -> V M C', C=50, V=25, M=2)
    flows = np.mean(flows.reshape(2, 25, 2, 5, 5), axis=(3,4))*10

    frame_copy = frame.copy()
    for person_num, person in enumerate(skeleton_frame.transpose(2, 0, 1)):
        for keypoint_num, keypoint in enumerate(person.transpose(1, 0)):
            x, y = int(keypoint[0]), int(keypoint[1])
            if 0 in keypoint:
                continue
            if person_num == 0:
                cv2.arrowedLine(frame_copy, 
                                (x, y), 
                                (int(x+flows[person_num,keypoint_num,0]), int(y+flows[person_num,keypoint_num,1])), 
                                (0, 255, 0), 
                                5, 
                                tipLength=1)
            else:
                cv2.arrowedLine(frame_copy,
                                (x, y), 
                                (int(x+flows[person_num,keypoint_num,0]), int(y+flows[person_num,keypoint_num,1])), 
                                (255, 0, 0), 
                                5, 
                                tipLength=1)
    return frame_copy


def vis_video(video_path, draw_skel=False, poses=None):
    if draw_skel:
        assert poses is not None
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        quit()
    else:
        print(f"Video opened: {video_path}")

    # Set the frame position
    frame_index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't retrieve frame...")
            break
        if draw_skel:
            frame = draw_bones(frame, poses[frame_index])
            frame = draw_skel(frame, poses[frame_index])
        # Display the frame
        cv2.imshow("Video Player", frame.astype(np.uint8))
        key = cv2.waitKey(0) & 0xFF  # Wait for a key press
        if key==ord("q"):
            break
        frame_index+=1

    cap.release()
    cv2.destroyAllWindows()



# # ----------------------------------
# # NTU-RGB+D PoseOFF Visualisation
# # ----------------------------------

# poseoff_file = osp.join('./data/ntu', 'aligned_data', 'MINI_CS_poseoff.npz')
# poseoff_data = np.load(poseoff_file, allow_pickle=True)['x_train'] # (120, 300, 2650)

# # Get a single frame
# data = poseoff_data[0,0] # (2*25*C)
# data = rearrange(data, '(M V C) -> M V C', M=2, V=25, C=5)

# # Get the pose and flow data
# person = data[0] # (25, 53)
# poses = person[:, :3] # (25, 3)
# flows = rearrange(person[:, 3:], 'K (C W H) -> K C W H',K=25,C=2, W=5, H=5) # (25, 2, 5, 5)

# frame = np.zeros((240, 320))

# def draw_flow_vectors(flow_data, frame, start_x, start_y):
#     """
#     Draws a group of optical flow vectors on the image.

#     Parameters:
#     flow_data (numpy.ndarray): The optical flow data.
#     frame (numpy.ndarray): The image frame.
#     start_x (int): The starting x-coordinate for the 5x5 region.
#     start_y (int): The starting y-coordinate for the 5x5 region.
#     size (int): The size of the region (default is 5).
#     """
#     for i in range(int(-size/2), int(size/2)):
#         for j in range(int(-size/2), int(size/2)):
#             x = start_x + i
#             y = start_y + j
#             # if x < flow_data.shape[2] and y < flow_data.shape[1]:
#             flow_vector = flow_data[:, j, i]
#             end_x = int(x + flow_vector[0])
#             end_y = int(y + flow_vector[1])
#             cv2.arrowedLine(frame, (end_x, end_y), (x, y), (0, 255, 0), 1, tipLength=0.1)
#     flow = np.mean(flow_data, axis=(1,2))
#     cv2.arrowedLine(frame, (start_x, start_y), (int(start_x+flow[0]), int(start_y+flow[1])), (0, 255, 0), 5, tipLength=1)

# # Draw the skeleton keypoints on the frame
# for keypoint_number, keypoint in enumerate(poses):
#     keypoint[0] = (keypoint[0]+0.5)*319
#     keypoint[1] = (keypoint[1]+0.5)*239
#     if 0 in keypoint:
#         continue
#     draw_flow_vectors(flows[keypoint_number],
#                       frame, 
#                       int(keypoint[0]), 
#                       int(keypoint[1]))
#     cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 1, 255, -1)


# plt.imshow(frame, cmap='gray')
# plt.savefig('NTU_poseoff.png')

def get_possible_samples(class_no, skes_names):
    for S in range(30):
        for P in range(150):
            video_name = f"S{S:03d}C001P{P:03d}R001A{class_no:03d}"
            try:
                ske_number = list(skes_names).index(video_name)
                print(video_name)
                continue
            except ValueError:
                pass


if __name__ == "__main__":
    from data.visualisations.interactive_vis import draw_bones, draw_skel
    # Using video with two subjects
    # video_name = "S001C003P008R001A050" # Punch
    # video_name = "S001C001P001R001A055" # Hug
    # video_name = "S020C001P041R001A112" # High five
    # video_name = "S019C003P050R001A112" # High five
    # video_name = "S019C001P050R002A113" # Cheers and drink
    video_name = "S019C001P051R001A113" # Cheers and drink

    class_no = int(video_name.split('A')[-1])

    dataset_extn = '120' if class_no > 60 else ''


    # Load the names of the skeleton files
    skes_names = np.loadtxt(
        f"./data/ntu{dataset_extn}/statistics/ntu_rgbd{dataset_extn}-available.txt",
        dtype=str
    )

    # Ensure video exists and get skeleton number
    try:
        ske_number = list(skes_names).index(video_name)
    except ValueError:
        print(f"Video name {video_name} does not exist\nTry:")
        get_possible_samples(class_no, skes_names)
    print(f"Video name: {video_name}")
    print(f"Skeleton number: {ske_number}")

    # Example usage
    video_path = f"../Datasets/NTU_RGBD{dataset_extn}/nturgb+d_rgb{dataset_extn}/{video_name}_rgb.avi"
    pose_path = f"./data/ntu{dataset_extn}/raw_data/raw_skes_data.pkl"
    pose_denoised_path = f"./data/ntu{dataset_extn}/denoised_data/raw_denoised_colors.pkl"
    poseoff_path = f"data/ntu{dataset_extn}/flow_data/poseoff_data.npy"

    with open(pose_denoised_path, "rb") as f:
        denoised_skes_data = pickle.load(f)

    # Get the specific denoised skeleton
    denoised = denoised_skes_data[ske_number]
    T, M, V, C = denoised.shape
    denoised = rearrange(denoised, 'T M V C -> T (M V) C')

    # Quick way to visualise the video
    vis_video(video_path, draw_skel, denoised)


    # # Load the temporary poseoff path
    # # NOTE: This was found by getting the index of the skeleton video within the skes_names
    # # Then finding the specific index within the train or test set using the: 
    # # get_indices() function from seq_transformation.py, then extracting that specific array
    # # from the poseoff_aligned.npz
    # poseoff = np.load(poseoff_path)
    # poseoff = poseoff.reshape(300, 2, 25, 53)
    # flows = poseoff.transpose(3, 0, 2, 1)[3:, :denoised.shape[1],...] # C T V M

    # print('(C, T, V, M)')
    # print(f'Denoised skeletons: {denoised.shape}')
    # print(f'Flow: {flows.shape}')

    # # # Draw the skeleton on the frame
    # frame_with_skeleton = draw_skeleton_on_frame(video_path, denoised, frame_index=40)
    # frame_with_poseoff = draw_flow_windows(frame_with_skeleton, flows, denoised, frame_index=40)
    # cv2.imwrite("skeleton_from_denoised.png", frame_with_skeleton)
    # cv2.imwrite("skeleton_from_poseoff.png", frame_with_poseoff)
