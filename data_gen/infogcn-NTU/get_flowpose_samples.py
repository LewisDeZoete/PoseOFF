import sys
import os
import os.path as osp

# # add lib to path
curr_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.abspath(osp.join(curr_dir, '../..')))

from data_gen.utils import LoadVideo, GetFlow, FlowPoseSampler
from config.argclass import ArgClass
import torch
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.v2 as v2
import numpy as np
import pickle
import time

arg = ArgClass(arg='./config/custom_pose/train_base.yaml')
transform_args = arg.extractor

# Get the device
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

# Create the model, move it to device and turn to eval mode
weights = torch.load(transform_args['flow']['weights'], weights_only=True, map_location=device)
model = raft_large(progress=False)
model.load_state_dict(weights)
model = model.eval().to(device)
transforms = v2.Compose([
    LoadVideo(max_frames=300),
    v2.Resize(size=transform_args['flow']['imsize']),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # map [0, 1] into [-1, 1]
    GetFlow(model=model, device=device, minibatch_size=transform_args['flow']['minibatch_size'])
    ])

# TODO: Make sure these values are reflected in a correct ntu_config.yaml file
transform_args['flowpose']['norm'] = False
transform_args['flowpose']['match_pose'] = False
transform_args['flowpose']['ntu'] = True

# Create the FlowPoseSampler transform object
flowPoseTransform = FlowPoseSampler(**transform_args['flowpose'])

def remove_frame_drops(flow_data, frames_drop_list):
    """
    Remove the frame drops from the flow data.
    """
    # Convert the list to a set for efficient membership testing
    skip_set = set(frame-1 for frame in frames_drop_list if frame > 0)

    # Create a boolean mask indicating which frames to keep
    mask = np.array([i not in skip_set for i in range(flow_data.shape[0])])

    # Use the mask to filter the array
    return flow_data[mask]


def get_raw_flowpose_data():
    skes_names = np.loadtxt(skes_name_file, dtype=str)
    num_files = skes_names.size
    print('Found %d available skeleton files.\n' % num_files, flush=True)

    flowpose_data = []

    # Assuming we've already processed all the pose data
    with open(denoised_skes_data_file, 'rb') as fr:  # load raw skeletons data
        denoised_skes_data = pickle.load(fr)
    
    # Get the frame drop dictionary to trip the flow data
    with open(frames_drop_file, 'rb') as fr:
        frames_drop_skes = pickle.load(fr)

    start = time.time()
    for ske_number, ske_name in enumerate(skes_names[40000:]):
        ske_number += 40000
        # Get the flow data
        rgb_name = osp.join(rgb_path, ske_name + '_rgb.avi')
        flow_data = transforms(rgb_name)

        # Get the pose data
        poses = denoised_skes_data[ske_number]
        poses = poses.transpose(3, 0, 2, 1)

        # Remove the frame drops from flow_data
        if ske_name in frames_drop_skes:
        # BUG: Frame_drop_skes[ske_name] has values from 0-last frame, but flow_data has values from 1-last frame
            flow_data = remove_frame_drops(flow_data, frames_drop_skes[ske_name])
            print(f'\tFrames dropped for {ske_name}: {len(frames_drop_skes[ske_name])}', flush=True)
            if 0 in frames_drop_skes[ske_name]:
                print('\t\tFrame 0 in skip list, add blank frame to poses', flush=True)
                C, _, V, M = poses.shape
                poses = np.concatenate([np.zeros((C, 1, V, M)), poses], axis=1)

        # Get the flowpose data!
        flowpose_data.append(flowPoseTransform(flow_data, poses))

        if (ske_number+1) % 1000 == 0:
            print(f'Processed {ske_number-999}-{ske_number} in {time.time()-start:0.2f} seconds', flush=True)
            start = time.time()

    # Save the data
    flowpose_pkl = osp.join(save_path, 'flowpose_data_40k-56k.pkl')
    with open(flowpose_pkl, 'wb') as f:
        pickle.dump(flowpose_data, f, pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    root_path = './data/ntu'
    save_path = osp.join(root_path, 'flowpose_data')

    # Create the directories if they don't exist
    if not osp.exists(save_path):
        os.makedirs(save_path)    

    # Define paths
    stat_path = osp.join(root_path, 'statistics')
    rgb_path = '../Datasets/NTU_RGBD/nturgb+d_rgb/'
    skes_name_file = osp.join(stat_path, 'ntu_rgbd-available.txt')
    denoised_skes_data_file = osp.join(root_path, 'denoised_data', 'raw_denoised_colors.pkl')
    frames_drop_file = osp.join(root_path, 'raw_data', 'frames_drop_skes.pkl')
    
    # Generate the data
    get_raw_flowpose_data()

