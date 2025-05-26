import numpy as np
import os
import os.path as osp
import pickle
from einops import rearrange
import yaml

in_path = './data/ntu120/flowpose_data'
out_path = './data/ntu120/flow_data'
files = os.listdir('./data/ntu120/flowpose_data')
files.sort()

flow_data = []

# Iterate over the files
for file in files:
    print(f'Appending {file}')
    with open(osp.join(in_path,file), 'rb') as fr:  # load raw skeletons data
        data = pickle.load(fr)
    
    for sample in data:
        flow_data.append(sample)
    print(f'Flow data samples: {len(flow_data)}')

# Save the data
with open(osp.join(out_path, 'flow_data.pkl'), 'wb') as f:
    pickle.dump(flow_data, f, pickle.HIGHEST_PROTOCOL)



# root_path = './data/ntu120'
# save_path = osp.join(root_path, 'aligned_data')
# stat_path = osp.join(root_path, 'statistics')

# with open(osp.join(stat_path, 'ntu_rgbd120-available.txt'), 'r') as f:
#     skes_names = f.read().splitlines()

# class_dict = {}
# for name in skes_names:
#     class_dict[name] = int(name[-3:])

# with open(osp.join(stat_path, 'ntu_rgbd120-annotations.yaml'), 'w') as f:
#     yaml.dump(class_dict, f)

# for file in file_paths:
#     # print(osp.join(save_path, f"MINI_{file.split('_')[2].split('-')[0]}_flowpose"))
#     org_data = np.load(file)
#     splits = ["x_train", "x_test"]
#     small_set = {}
#     for split in splits:
#         data = org_data[split]
#         small_set[split] = data[:120]
    
#     np.savez(osp.join(save_path, f"MINI_{file.split('_')[2].split('-')[0]}_flowpose"),
#              x_train=small_set["x_train"],
#              y_train=org_data["y_train"][:120],
#              x_test=small_set["x_test"],
#              y_test=org_data["y_test"][:120])


# def get_details(skes_name):
#     details: dict = {} # Create and populate details dict
#     for key in ['Setup', 'Camera', 'Performer', 'Replication', 'Label', 'Frame_cnt']:
#         details[key] = np.array([], dtype=int)
#     # Get the details from the file names
#     for number, name in enumerate(skes_name):
#         details['Setup'] = np.append(details['Setup'], int(name.split('S')[1][:3]))
#         details['Camera'] = np.append(details['Camera'], int(name.split('C')[1][:3]))
#         details['Performer'] = np.append(details['Performer'], int(name.split('P')[1][:3]))
#         details['Replication'] = np.append(details['Replication'], int(name.split('R')[1][:3]))
#         details['Label'] = np.append(details['Label'], int(name.split('A')[1][:3])-1)
#         details['Frame_cnt'] = np.append(details['Frame_cnt'], int(frames_cnt[number]))
    
#     return details

# details = get_details(skes_names)

# def get_indices(performer, camera, evaluation='CS'):
#     test_indices = np.empty(0)
#     train_indices = np.empty(0)

#     if evaluation == 'CS':  # Cross Subject (Subject IDs)
#         train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
#                      17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
#         test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
#                     24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

#         # Get indices of test data
#         for idx in test_ids:
#             temp = np.where(performer == idx)[0]  # 0-based index
#             test_indices = np.hstack((test_indices, temp)).astype(int)

#         # Get indices of training data
#         for train_id in train_ids:
#             temp = np.where(performer == train_id)[0]  # 0-based index
#             train_indices = np.hstack((train_indices, temp)).astype(int)
#     else:  # Cross View (Camera IDs)
#         train_ids = [2, 3]
#         test_ids = 1
#         # Get indices of test data
#         temp = np.where(camera == test_ids)[0]  # 0-based index
#         test_indices = np.hstack((test_indices, temp)).astype(int)

#         # Get indices of training data
#         for train_id in train_ids:
#             temp = np.where(camera == train_id)[0]  # 0-based index
#             train_indices = np.hstack((train_indices, temp)).astype(int)

#     return train_indices, test_indices

# train_indices, test_indices = get_indices(
#     details['Performer'], 
#     details['Camera'], 
#     evaluation='CV')


# print(f'Train shape: {train_indices.shape}')
# print(f'Test shape: {test_indices.shape}')
# print(f'Train array: {np.argmax(np.where(train_indices == 2801, 1, 0))}')
# print(f'Test array: {np.argmax(np.where(test_indices == 2801, 1, 0))}')

# # print(test_indices[19730])