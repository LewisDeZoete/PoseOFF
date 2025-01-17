import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

from lib.data.dataset import SingleStreamDataset
from extractors import GetPoses_YOLO
from config.argclass import ArgClass
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms.v2 as v2
import time
import argparse

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

# Get the number of videos in the class (used to get indices of dataset)
def get_range(class_no):
    len_class = 0
    for i in arg.feeder_args['labels'].keys():
        if i.split('/')[0] == list(classes.keys())[class_no]:
            try:
                assert start_index >= 0
            except NameError:
                start_index = list(arg.feeder_args['labels'].keys()).index(i)
            len_class += 1
    return range(start_index, (start_index+len_class))

# Get the device
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

# Create pose detector
detector = YOLO(transform_args['weights'])
detector.to(device)
transforms = [
    v2.Resize(size=(384,640)), # YOLO pose has a minimum input image size
    v2.ToDtype(torch.float32),
    v2.Lambda(lambda x: x/255.0)] # Normalises the image to [0,1]

# Create the dataset object
dataset = SingleStreamDataset(arg=arg, stream='rgb', ext='.avi', transforms=transforms)

start = time.time()
# Check if the indices we've been given are for the overall 
# dataset or as indices for the 'unfinished' list in config
if 'unfinished' in arg.__dict__:
    for idx in get_range(classes[arg.unfinished[arg_no]]):
        poses, label = dataset[idx]
        path = f'data/UCF-101/pose/{list(arg.feeder_args["labels"].keys())[idx]}'.split('.')[0]  + ('.npy' if save_as_numpy else '.pt')
        if save_as_numpy:
            np.save(path, poses.numpy())
        else:
            torch.save(poses, path)
        
    print(f'\nFinished processing {arg.unfinished[arg_no]} in {time.time()-start:0.5f} seconds')
else:
    for idx in get_range(arg_no):
        poses, label = dataset[idx]
        # using the class folder from the annotated file so both methods work
        folder = f'data/UCF-101/pose/{list(arg.feeder_args["labels"].keys())[idx].split("/")[0]}/'
        try:
            # If not create the folder!
            os.mkdir(folder)
        except FileExistsError:
            pass
        path = os.path.join(folder, list(arg.feeder_args['labels'].keys())[idx].split('/')[-1].split('.')[0] + ('.npy' if save_as_numpy else '.pt'))
        if save_as_numpy:
            np.save(path, poses.numpy())
        else:
            torch.save(poses, path)

    print(f'\nFinished processing {list(classes.keys())[arg_no]} in {time.time()-start:0.5f} seconds')