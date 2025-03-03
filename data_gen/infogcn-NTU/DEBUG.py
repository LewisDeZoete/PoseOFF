import numpy as np
import os.path as osp
import pickle

root_path = './data/ntu'
save_path = osp.join(root_path, 'denoised_data')
stat_path = osp.join(root_path, 'statistics')
raw_denoised_joints_path = osp.join(save_path, 'raw_denoised_joints.pkl')
raw_denoised_colors_path = osp.join(save_path, 'raw_denoised_colors.pkl')
skes_name_file = osp.join(stat_path, 'ntu_rgbd-available.txt')
frames_cnt_file = osp.join(save_path, 'frames_cnt.txt')

with open(raw_denoised_joints_path, 'rb') as fr:
    raw_denoised_joints = pickle.load(fr)
with open(raw_denoised_colors_path, 'rb') as fr:
    raw_denoised_colors = pickle.load(fr)

skes_names = np.loadtxt(skes_name_file, dtype=str)
frames_cnt = np.loadtxt(frames_cnt_file, dtype=int)

details: dict = {}
for key in ['Setup', 'Camera', 'Performer', 'Replication', 'Label', 'Frame_cnt']:
    details[key] = np.array([])

# for number, name in enumerate(skes_names):
#     details[name] = {
#         'Setup': name.split('S')[1][:3],
#         'Camera': name.split('C')[1][:3],
#         'Performer': name.split('P')[1][:3],
#         'Replication': name.split('R')[1][:3],
#         'Label': name.split('A')[1][:3],
#         'Frame_cnt': frames_cnt[number]
#     }

# Create the details dictionary containing all the setups, camera, performers, 
# replication numbers and frame numbers
# This dict only represent data within ntu_argb-available.txt
for number, name in enumerate(skes_names):
    details['Setup'] = np.append(details['Setup'], int(name.split('S')[1][:3]))
    details['Camera'] = np.append(details['Camera'], int(name.split('C')[1][:3]))
    details['Performer'] = np.append(details['Performer'], int(name.split('P')[1][:3]))
    details['Replication'] = np.append(details['Replication'], int(name.split('R')[1][:3]))
    details['Label'] = np.append(details['Label'], int(name.split('A')[1][:3]))
    details['Frame_cnt'] = np.append(details['Frame_cnt'], int(frames_cnt[number]))


test_indices = np.empty(0)
train_indices = np.empty(0)

train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                     17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
            24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

evaluation = 'CV'

if evaluation == 'CS':  # Cross Subject (Subject IDs)
    for idx in test_ids:
        temp = np.where(details['Performer'] == idx)[0]  # 0-based index
        test_indices = np.hstack((test_indices, temp)).astype(int)

    # Get indices of training data
    for train_id in train_ids:
        temp = np.where(details['Performer'] == train_id)[0]  # 0-based index
        train_indices = np.hstack((train_indices, temp)).astype(int)

else:  # Cross View (Camera IDs)
    train_ids = [2, 3]
    test_ids = 1
    # Get indices of test data
    temp = np.where(details['Camera'] == test_ids)[0]  # 0-based index
    test_indices = np.hstack((test_indices, temp)).astype(int)

    # Get indices of training data
    for train_id in train_ids:
        temp = np.where(details['Camera'] == train_id)[0]  # 0-based index
        train_indices = np.hstack((train_indices, temp)).astype(int)

print(test_indices.shape, train_indices.shape)