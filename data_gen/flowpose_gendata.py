import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

# from lib.data.dataset import MultiStreamDataset
from utils import extract_data, FlowPoseSampler
from config.argclass import ArgClass
import argparse
import time
# import torch
# import numpy as np

parser = argparse.ArgumentParser(prog="flowpose_gendata")

parser.add_argument('-n', dest='number',
                    help='Class number for processing flowpose of a specific class')
parser.add_argument('--numpy', action='store_true',
                    help='Option to save data as numpy arrays')
parsed = parser.parse_args()
process_number = int(parsed.number) # Get class number command line arg
save_as_numpy = parsed.numpy

# Get the arg object and create the classes
arg = ArgClass(arg='./config/ucf101/train_joint_infogcn.yaml')
arg.extractor['preprocessed'] = False # Override this value, since this is gendata script!
transform_args = arg.extractor['flowpose'] # grab transforms arg

# Create the FlowPoseSampler transform object
flowPoseTransform = FlowPoseSampler(**transform_args)

# ------------------------------
#         PROCESS
# ------------------------------
start = time.time()

extract_data(arg, 
             process_number=process_number, 
             transforms=flowPoseTransform, 
             modality='flowpose', 
             save_as_numpy=save_as_numpy)

print(f'Processing time: {time.time()-start:.2f}s')