import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

from utils import LoadVideo, get_class_by_index, GetFlow
from config.argclass import ArgClass
import argparse
import time
from torchvision.models.optical_flow import raft_large
import torch
import torchvision.transforms.v2 as v2
import numpy as np


parser = argparse.ArgumentParser(prog="flow_gendata")

parser.add_argument('-n', dest='number',
                    help='Class number for processing pose keypoints of a specific class')
parser.add_argument('--numpy', action='store_true',
                    help='Option to save data as numpy arrays')
parsed = parser.parse_args()
process_number = int(parsed.number) # Get class number command line arg
save_as_numpy = parsed.numpy

# Get the arg object and create the classes
arg = ArgClass(arg='./config/ucf101/train_joint_infogcn.yaml')
arg.extractor['preprocessed'] = False # Override this value, since this is gendata script!
transform_args = arg.extractor['flow']

# Get the device
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

# Create the model, move it to device and turn to eval mode
weights = torch.load(transform_args['weights'], weights_only=True, map_location=device)
model = raft_large(progress=False)
model.load_state_dict(weights)
model = model.eval().to(device)
transforms = v2.Compose([
    LoadVideo(max_frames=300),
    v2.Resize(size=transform_args['imsize']),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # map [0, 1] into [-1, 1]
    GetFlow(model=model, device=device, minibatch_size=transform_args['minibatch_size'])
    ])

# Ensure the data_paths['flow_path'] exists
try:
    os.mkdir(arg.feeder_args['data_paths']['flow_path'])
    print(f'Creating pose path: {arg.feeder_args["data_paths"]["flow_path"]}')
except FileExistsError:
    pass


def get_flow(arg, process_number: int):
    '''
    Get and save the flow for a specific class
    '''    
    class_name, video_names = get_class_by_index(arg,
                                                 process_number=process_number,
                                                 modality='flow_path')
    print('Processing:', class_name)
    for video_name in video_names:
        # Get the video path
        video_path = f"{os.path.join(arg.feeder_args['data_paths']['rgb_path'], class_name, video_name)}.avi"
        # Transform and estimate flow
        flow = transforms(video_path)
        # Get the path to save the estimated poses to
        save_path = os.path.join(arg.feeder_args['data_paths']['flow_path'], # Data path
                                 class_name, # Class folder
                                 video_name +
                                 ('.npy' if save_as_numpy else '.pt')) # Video name (numpy/torch)
        if save_as_numpy:
            np.save(save_path, flow.numpy())
        else:
            torch.save(flow, save_path)
    
    print(f'Processed {class_name} class')

# ------------------------------
#         PROCESS
# ------------------------------
start = time.time()

get_flow(arg, process_number=process_number)

print(f'Processing time: {time.time()-start:.2f}s')