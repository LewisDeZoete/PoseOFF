import os
import os.path as osp
import pickle
import argparse
import re
from einops import rearrange
import yaml
import numpy as np

parser = argparse.ArgumentParser(description='NTU-RGB-D Data Preparation')
parser.add_argument(
    '--dataset', 
    dest='dataset', 
    default='ntu',
    help='Dataset, either `ntu` or `ntu120` (default=ntu)'
)
parser.add_argument(
    '--dilation',
    dest='dilation',
    default=1,
    help='Poseoff extraction window dilation amount (mostly used for debug)'
)
parser.add_argument(
    '--flow_type',
    dest='flow_type',
    default='RAFT',
    help='Type of flow used to extract the PoseOFF windows.'
)
args = parser.parse_args()

# Parsed command line arguments
dataset = args.dataset
flow_type = args.flow_type

# Printing for debug
print(f"Concatenating {args.dataset} - {args.flow_type} PoseOFF.")
print(f"\tDILATION: {args.dilation}\n\n")

# Paths
root_path = f'./data/{dataset}'
in_path = osp.join(root_path, 'flow_data', 'export_tmp')
save_path = osp.join(root_path, 'flow_data', flow_type)
# Ensure the save_path exists...
os.makedirs(save_path, exist_ok=True)
# Give the flow data a more meaningful name to reflect the dilation value
save_name = f"flow_data_{flow_type}_D{args.dilation}.pkl"
files = os.listdir(in_path)

flow_file_names = {} # Dictionary to hold file names
flow_data = [] # List to hold flow data

# Verify filenames
for file_name in files:
    # Check if the file name matches 'flow_...(number)k.pkl
    x = re.search(r'^flow_.*(\d)k\.pkl$', file_name)
    if not x:
        print(f'Skipping {file_name}, does not match expected pattern.')
        continue
    # {Lowest skeleton number : filename}
    flow_file_names[re.search(r'(\d+)', file_name).group(0)] = file_name

# Sort the dictionary by the skeleton number (keys)
flow_file_names = dict(sorted(flow_file_names.items(), key=lambda item: int(item[0])))

# Open flow files and apend data to flow_data
for _, file in flow_file_names.items():
    with open(osp.join(in_path, file), 'rb') as fr:  
        data = pickle.load(fr)
    
    for sample in data:
        flow_data.append(sample)
    print(f'Flow data samples: {len(flow_data)}')

# Save the data
with open(osp.join(save_path, save_name), 'wb') as f:
    pickle.dump(flow_data, f, pickle.HIGHEST_PROTOCOL)

print(f"Saved poseoff file to: {osp.join(save_path, save_name)}")
