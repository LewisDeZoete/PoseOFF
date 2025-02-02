import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

# from lib.data.dataset import SingleStreamDataset
from extractors import GetPoses_YOLO
from config.argclass import ArgClass
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms.v2 as v2
import time
import argparse
from preprocess import LoadVideo

parser = argparse.ArgumentParser(prog="skel_gendata")

parser.add_argument('-n', dest='number',
                    help='Class number for processing pose keypoints of a specific class')
parser.add_argument('--numpy', action='store_true',
                    help='Option to save data as numpy arrays')
parsed = parser.parse_args()
arg_no = int(parsed.number) # Get class number command line arg
save_as_numpy = parsed.numpy

# Get the arg object and create the classes
arg = ArgClass(arg='./config/custom_pose/train_joint_infogcn.yaml')
arg.extractor['preprocessed'] = False # Override this value, since this is gendata script!
classes = arg.classes # Get the classes
transform_args = arg.extractor['pose'] # grab transforms arg

# Get the device
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

# Create pose detector
detector = YOLO(transform_args['weights'])
detector.to(device)
transforms = v2.Compose([
    v2.Resize(size=(384,640)), # YOLO pose has a minimum input image size
    v2.ToDtype(torch.float32),
    v2.Lambda(lambda x: x/255.0), # Normalises the image to [0,1]
    GetPoses_YOLO(detector=detector, max_frames=300, num_joints=17)
    ])

# Get the paths to the videos
video_paths = [os.path.join(arg.feeder_args['data_paths']['rgb_path'],
                        (list(arg.feeder_args['labels'].keys())[i]+'.avi')) \
        for i in range(len(list(arg.feeder_args['labels'].keys())))]


def get_range(class_no):
    '''
    Get the number of videos in the class (used to get indices of dataset)
    '''
    len_class = 0
    for i in arg.feeder_args['labels'].keys():
        if i.split('/')[0] == list(classes.keys())[class_no]:
            try:
                assert start_index >= 0
            except NameError:
                start_index = list(arg.feeder_args['labels'].keys()).index(i)
            len_class += 1
    return range(start_index, (start_index+len_class))


def get_poses(class_number):
    '''
    Get and save the poses for a specific class
    '''
    for idx in get_range(class_number):
        # # Load the video
        # video = LoadVideo(video_paths[idx])
        # # Transform and estimate poses
        # poses = transforms(video)
        # Get the path to save the estimated poses to
        save_path = os.path.join(arg.feeder_args['data_paths']['pose_path'], # Data path
                                 list(classes.keys())[arg_no], # Class folder
                                 list(arg.feeder_args['labels'].keys())[idx].split('/')[-1] \
                                    + ('.npy' if save_as_numpy else '.pt')) # Video name (numpy/torch)
        print(save_path)
        # if save_as_numpy:
        #     np.save(save_path, poses.numpy())
        # else:
        #     torch.save(poses, save_path)


# ------------------------------
#         PROCESS
# ------------------------------
start = time.time()

if 'unfinished' in arg.__dict__:
    # If we're processing unfinished classes (defined in arg yaml)...
    # The arg_no defines the specific class in the list within the yaml
    get_poses(classes[arg.unfinished[arg_no]])
else:
    # Otherwise, process class number according to the annotation dictionary
    folder = os.path.join(arg.feeder_args['data_paths']['pose_path'],
                          list(classes.keys())[arg_no])
    # Create the class folder if it doesn't exist
    try:
        os.mkdir(folder)
    except FileExistsError:
        print('Folder already exists')
        pass
    get_poses(arg_no)

print(f'Processing time: {time.time()-start:.2f}s')