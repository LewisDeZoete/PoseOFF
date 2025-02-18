import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '../..')))

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

# Create the FlowPoseSampler transform object
flowPoseTransform = FlowPoseSampler(**transform_args)

def get_raw_flowpose_data():
    skes_name = np.loadtxt(skes_name_file, dtype=str)
    num_files = skes_name.size
    print('Found %d available skeleton files.' % num_files)
    # Assuming we've already processed all the pose data
    with open(raw_data_file, 'rb') as fr:  # load raw skeletons data
        raw_skes_data = pickle.load(fr)


    # Get the skeleton data
    for (idx, bodies_data) in enumerate(raw_skes_data):
        ske_name = bodies_data['name']
        num_bodies = len(bodies_data['data'])
        if num_bodies == 1:
            num_frames = bodies_data['num_frames']
            body_data = list(bodies_data['data'].values())[0]
            joints, colors = body_data['joints'], body_data['colors']
        else:
            # TODO: handle multiple bodies
            pass

        rgb_name = os.path.join(rgb_path, ske_name + '_rgb.avi')
        flow_data = transforms(rgb_name)
        flow_data = flowPoseTransform(flow_data, colors)

        break


if __name__ == '__main__':
    save_path = './data/ntu'

    stat_path = os.path.join(save_path, 'statistics')
    flowpose_path = os.path.join(save_path, 'flowpose_data')
    if not os.path.exists(flowpose_path):
        os.makedirs(flowpose_path)

    rgb_path = '../Datasets/NTU_RGBD/nturgb+d_rgb/'
    skes_name_file = os.path.join(stat_path, 'ntu_rgbd-available.txt')
    raw_data_file = os.path.join(save_path, 'raw_data', 'raw_skes_data.pkl')
    
    get_raw_flowpose_data()

