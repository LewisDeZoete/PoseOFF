import sys
import os.path as osp

# # add lib to path
curr_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.abspath(osp.join(curr_dir, '../..')))

from data_gen.utils import LoadVideo
import numpy as np
import pickle


broken_ske_name = 'S001C002P001R002A059'
broken_ske_name = 'S001C002P001R002A060'
rgb_path = '../Datasets/NTU_RGBD/nturgb+d_rgb/'
root_path = './data/ntu'
stat_path = osp.join(root_path, 'statistics')
skes_name_file = osp.join(stat_path, 'ntu_rgbd-available.txt')
denoised_skes_data_file = osp.join(root_path, 'denoised_data', 'raw_denoised_colors.pkl')

skes_names = np.loadtxt(skes_name_file, dtype=str)
skes_names = skes_names.tolist()
with open(denoised_skes_data_file, 'rb') as fr: # load raw skeletons data
	denoised_skes_data = pickle.load(fr)

with open('./data/ntu/raw_data/frames_drop_skes.pkl', 'rb') as fr:
    frames_drop_skes = pickle.load(fr)

for ske_name, drop_list in frames_drop_skes.items():
    print(drop_list)
    break

frames_drop_list = frames_drop_skes[broken_ske_name]

def remove_frame_drops(flow_data, frames_drop_list):
    """
    Remove the frame drops from the flow data.
    """
    # Convert the list to a set for efficient membership testing
    skip_set = set(frames_drop_list)

    # Create a boolean mask indicating which frames to keep
    mask = np.array([i not in skip_set for i in range(flow_data.shape[0])])

    # Use the mask to filter the array
    flow_data = flow_data[mask]
    
    return flow_data

rgb_name = osp.join(rgb_path, broken_ske_name+'_rgb.avi')

vid_loader = LoadVideo(max_frames=300)
video = vid_loader(rgb_name)
poses = denoised_skes_data[skes_names.index(broken_ske_name)]

if broken_ske_name in frames_drop_skes:
    trimmed = remove_frame_drops(video, frames_drop_skes[broken_ske_name])

print(trimmed.shape)
print(poses.shape)
