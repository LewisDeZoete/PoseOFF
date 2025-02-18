import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

from utils import LoadVideo, get_class_by_index, GetPoses_YOLO
from config.argclass import ArgClass
import argparse
import time
from ultralytics import YOLO
import torch
import torchvision.transforms.v2 as v2
import numpy as np

parser = argparse.ArgumentParser(prog="pose_gendata")

parser.add_argument('-n', dest='number',
                    help='Class number for processing pose keypoints of a specific class')
parser.add_argument('--numpy', action='store_true',
                    help='Option to save data as numpy arrays')
parsed = parser.parse_args()
process_number = int(parsed.number) # Get class number command line arg
save_as_numpy = parsed.numpy

# Get the arg object and create the classes
arg = ArgClass(arg='./config/custom_pose/train_joint_infogcn.yaml')
arg.extractor['preprocessed'] = False # Override this value, since this is gendata script!
transform_args = arg.extractor['pose'] # grab transforms arg

# Get the device
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

# Create pose detector
detector = YOLO(transform_args['weights'])
detector.to(device)
transforms = v2.Compose([
    LoadVideo(max_frames=300),
    v2.Resize(size=(384,640)), # YOLO pose has a minimum input image size
    v2.ToDtype(torch.float32),
    v2.Lambda(lambda x: x/255.0), # Normalises the image to [0,1]
    GetPoses_YOLO(detector=detector, max_frames=300, num_joints=17)
    ])


def get_poses(arg, process_number: int):
    '''
    Get and save the poses for a specific class
    '''
    class_name, video_names = get_class_by_index(arg,
                                                 process_number=process_number,
                                                 modality='pose')
    print('Processing:', class_name)
    for video_name in video_names:
        # Get the video path
        video_path = f"{os.path.join(arg.feeder_args['data_paths']['rgb_path'], class_name, video_name)}.avi"
        # Transform and estimate poses
        poses = transforms(video_path)
        # Get the path to save the estimated poses to
        save_path = os.path.join(arg.feeder_args['data_paths']['pose_path'], # Data path
                                 class_name, # Class folder
                                 video_name +
                                 ('.npy' if save_as_numpy else '.pt')) # Video name (numpy/torch)
        if save_as_numpy:
            np.save(save_path, poses.numpy())
        else:
            torch.save(poses, save_path)
    
    print(f'Processed {class_name} class')


# ------------------------------
#         PROCESS
# ------------------------------
start = time.time()

get_poses(arg, process_number=process_number)

print(f'Processing time: {time.time()-start:.2f}s')