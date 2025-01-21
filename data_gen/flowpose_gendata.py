import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

from lib.data.dataset import MultiStreamDataset
from config.argclass import ArgClass
from extractors import FlowPoseSampler
import torch
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser(prog="flowpose_gendata")

parser.add_argument('-n', dest='number',
                    help='Class number for processing flow of a specific class')
parser.add_argument('--numpy', action='store_true',
                    help='Option to save data as numpy arrays')
parsed = parser.parse_args()
arg_no = int(parsed.number) # Get class number command line arg
save_as_numpy = parsed.numpy

# Get the arg object and create the classes
arg = ArgClass(arg='./config/custom_pose/train_joint_infogcn.yaml')
classes = arg.classes
transform_args = arg.extractor['flowpose']


def is_null(data):
    '''Check if the data is filled with zeros'''
    if isinstance(data, torch.Tensor):
        return torch.all(data == 0).item()
    elif isinstance(data, np.ndarray):
        return np.all(data == 0)
    return False

def log_zero_data(key):
    '''Function to log the name of the data to a text file'''
    class_name = key.split('/')[0] # Get the class name from the key
    with open(f'./TMP/zero_data_{class_name}.txt', 'a') as f:
        f.write(f'{key}\n') # Write the key to the file

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

# Create the FlowPoseSampler transform object
flowPoseTransform = FlowPoseSampler(device=device, **transform_args)

# Create the dataset object
dataset = MultiStreamDataset(arg=arg, transforms=flowPoseTransform)


start = time.time()
# Check if the indices we've been given are for the overall 
# dataset or as indices for the 'unfinished' list in config
if 'unfinished' in arg.__dict__:
    for idx in get_range(classes[arg.unfinished[arg_no]]):
        flowpose, label = dataset[idx]
        if is_null(flowpose): # Check if the data is all zeros
            log_zero_data(key = list(arg.feeder_args['labels'].keys())[idx])
        path = f'{arg.feeder_args["data_paths"]["flowpose_path"]}{list(arg.feeder_args["labels"].keys())[idx]}' + ('.npy' if save_as_numpy else '.pt')
        if save_as_numpy:
            np.save(path, flowpose.numpy())
        else:
            torch.save(flowpose, path)
    print(f'\nFinished processing {arg.unfinished[arg_no]} in {time.time()-start:0.5f} seconds')

else:
    for idx in get_range(arg_no):
        flowpose, label = dataset[idx] # We're using FlowPoseSampler transform so this returns the flowpose!
        if is_null(flowpose):
            log_zero_data(key = list(arg.feeder_args['labels'].keys())[idx])
        # Check if the folder that the videos belong in exists
        folder = f'{arg.feeder_args["data_paths"]["flowpose_path"]}{list(arg.feeder_args["labels"].keys())[idx].split("/")[0]}/'
        try:
            # If not create the folder!
            os.mkdir(folder)
        except FileExistsError:
            pass
        path = os.path.join(folder, list(arg.feeder_args['labels'].keys())[idx].split('/')[-1] + ('.npy' if save_as_numpy else '.pt'))
        if save_as_numpy:
            np.save(path, flowpose.numpy())
        else:
            torch.save(flowpose, path)
    print(f'\nFinished processing {list(classes.keys())[arg_no]} in {time.time()-start:0.5f} seconds')