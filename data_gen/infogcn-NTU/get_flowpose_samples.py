import sys
import os
import os.path as osp

# # add lib to path
curr_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.abspath(osp.join(curr_dir, '../..')))

from data_gen.utils import LoadVideo, get_class_by_index, GetFlow, FlowPoseSampler
from config.argclass import ArgClass
import torch
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.v2 as v2
import numpy as np
import pickle

arg = ArgClass(arg='./config/custom_pose/train_joint_infogcn.yaml')
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

transform_args['flowpose']['norm'] = False
transform_args['flowpose']['match_pose'] = False
transform_args['flowpose']['ntu'] = True

# Create the FlowPoseSampler transform object
flowPoseTransform = FlowPoseSampler(**transform_args['flowpose'])

def get_raw_flowpose_data():
    skes_names = np.loadtxt(skes_name_file, dtype=str)
    num_files = skes_names.size
    print('Found %d available skeleton files.' % num_files)

    flowpose_data = []

    # Assuming we've already processed all the pose data
    with open(denoised_skes_data_file, 'rb') as fr:  # load raw skeletons data
        denoised_skes_data = pickle.load(fr)

    for ske_number, ske_name in enumerate(skes_names):
        # Get the flow data
        rgb_name = osp.join(rgb_path, ske_name + '_rgb.avi')
        flow_data = transforms(rgb_name)

        # Get the pose data
        poses = denoised_skes_data[ske_number]
        poses = poses.transpose(3, 0, 2, 1)
        # TODO: ensure poses are in the same range as the flow data!
        # Right now, they're in the range (1920, 1080), flow is (320,240)

        # Get the flowpose data
        flowpose_data.append(flowPoseTransform(flow_data, poses))
        break # DEBUG: remove this line

    flowpose_pkl = osp.join(save_path, 'flowpose_data.pkl')
    with open(flowpose_pkl, 'wb') as f:
        pickle.dump(flowpose_data, f, pickle.HIGHEST_PROTOCOL)
    # return flowpose_data

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
    
    # Generate the data
    get_raw_flowpose_data()

