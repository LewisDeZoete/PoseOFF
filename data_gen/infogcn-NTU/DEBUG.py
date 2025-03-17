import numpy as np
import os
import os.path as osp
import pickle
from einops import rearrange
import yaml

root_path = './data/ntu'
save_path = osp.join(root_path, 'aligned_data')
stat_path = osp.join(root_path, 'statistics')

with open(osp.join(stat_path, 'ntu_rgbd-available.txt'), 'r') as f:
    skes_names = f.read().splitlines()

# with open(osp.join(stat_path, 'frames_cnt.txt'), 'r') as f:
#     frames_cnt = f.read().splitlines()

# # (52, 184, 25, 1)
# flow = np.load(osp.join(root_path, 'flow_data', 'flow_data.pkl'), allow_pickle=True)

# # Raw denoised joints
# # (103, 75/150)
# denoised_joints = np.load(osp.join(root_path, 'denoised_data', 'raw_denoised_joints.pkl'), allow_pickle=True)
# T, MVC = denoised_joints[0].shape
# denoised_joints[0] = rearrange(denoised_joints[0], 'T (M V C) -> T M V C', T=T, M=int(MVC/25/3), V=25, C=3)
# print(f'[denoised_joints] Shape: {denoised_joints[0].shape}')
# print(f'[denoised_joints] Frame 0, joint 0: {denoised_joints[0][0, 0, 0, :]}')
# print(f'[denoised_joints] Frame 0, joint 5: {denoised_joints[0][0, 0, 5, :]}')
# print(f'[denoised_joints] Frame 50, joint 5: {denoised_joints[0][50, 0, 5, :]}')
# print(f'[denoised_joints] Frame 50, joint -1: {denoised_joints[0][50, 0, -1, :]}')

# # --------------------------------------
# # # TESTING TWO BODY SEQ TRANSFORM
# # --------------------------------------
# f = 118
# ske_joints = denoised_joints[f]
# missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
# missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
# cnt1 = len(missing_frames_1)
# cnt2 = len(missing_frames_2)

# print(f'Missing frames person 1: {missing_frames_1}')
# print(f'Missing frames person 2: {missing_frames_2}')

# if (cnt1 > 0):
#     ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)
#     if flow is not None: # Flow data covers frames 2 through T
#         skip_set1 = [frame_no-1 for frame_no in missing_frames_1 if frame_no > 0]
#         flow[f][skip_set1, :1250] = np.zeros((len(skip_set1), 1250), dtype=np.float32)

# if (cnt2 > 0):
#     ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)
#     if flow is not None: # Flow data covers frames 2 through T
#         print(f'Flow shape: {flow[f].shape}')
#         skip_set2 = [frame_no-1 for frame_no in missing_frames_2 if frame_no > 0]
#         flow[f][skip_set2, 1250:] = np.zeros((len(skip_set2), 1250), dtype=np.float32)


def get_details(skes_name):
    details: dict = {} # Create and populate details dict
    for key in ['Setup', 'Camera', 'Performer', 'Replication', 'Label', 'Frame_cnt']:
        details[key] = np.array([], dtype=int)
    # Get the details from the file names
    for number, name in enumerate(skes_name):
        details['Setup'] = np.append(details['Setup'], int(name.split('S')[1][:3]))
        details['Camera'] = np.append(details['Camera'], int(name.split('C')[1][:3]))
        details['Performer'] = np.append(details['Performer'], int(name.split('P')[1][:3]))
        details['Replication'] = np.append(details['Replication'], int(name.split('R')[1][:3]))
        details['Label'] = np.append(details['Label'], int(name.split('A')[1][:3])-1)
        # details['Frame_cnt'] = np.append(details['Frame_cnt'], int(frames_cnt[number]))
    
    return details


details = get_details(skes_names)
annotations = {}
for idx, ske_name in enumerate(skes_names):
    annotations[ske_name] = int(details['Label'][idx])

with open('data/ntu/statistics/ntu-rgbd-annotations.yaml', 'w') as file:
    yaml.dump(annotations, file)