# # Copyright (c) Microsoft Corporation. All rights reserved.
# # Licensed under the MIT License.
import sys
import os
import os.path as osp
import numpy as np
import pickle
import yaml
from data_gen.utils import create_aligned_dataset
import argparse

parser = argparse.ArgumentParser(description='NTU-RGB-D Data Preparation')
parser.add_argument('--dataset',dest='dataset', default='ntu', help='Dataset, either `ntu` or `ntu120` (default=ntu)')
parser.add_argument('--flow', action='store_true', help='If passed, add flow to the pose array') 
parser.add_argument('--realign', action='store_true', help='Reprocess the aligned data (store_true)')
args = parser.parse_args()
dataset = args.dataset

# Paths
root_path = f'./data/{dataset}'
save_path = osp.join(root_path, 'aligned_data')
stat_path = osp.join(root_path, 'statistics')
denoised_path = osp.join(root_path, 'denoised_data')
flow_path = osp.join(root_path, 'flow_data')
# Info files and folders
skes_name_file = osp.join(stat_path, f'ntu_rgbd{120 if dataset == "ntu120" else ""}-available.txt')
frames_file = osp.join(stat_path, 'frames_cnt.txt')
# Files
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
raw_flow_joints_pkl = osp.join(flow_path, 'flow_data.pkl')
raw_flowpose_pkl = osp.join(flow_path, 'raw_flowpose_data.pkl')

if not osp.exists(save_path):
    os.mkdir(save_path)


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
        details['Frame_cnt'] = np.append(details['Frame_cnt'], int(frames_cnt[number]))
    
    return details

def remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]


def seq_translation(skes_joints, flow_joints=None):
    """
    Translates the sequence of skeleton joints to a new origin based on the first non-zero frame of the first actor.

    Parameters:
    skes_joints (list: numpy.ndarray): list of length N containing numpy arrays of shape 
                                       (T, M*V*C) 
                                       T is the number of frames, 
                                       M is the number of actors (1 or 2),
                                       V is the number of joints, 
                                       C is the number of coordinates per joint.
    flow_joints (list: numpy.ndarray): list of length N containing numpy arrays of shape
                                       (T-1, M*V*C)
                                       T is the number of frames,
                                       M is the number of actors (1 or 2),
                                       V is the number of joints,
                                       C is the flow data per joint ((flow_window**2)*2).

    Returns:
    numpy.ndarray: The translated sequence of skeleton joints with the same shape as the input.
    """
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        # Set joint 2 (core) as new origin
        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        # TODO: VERY IMPORTANT HERE, Make sure missing frames corresponds to correct
        #       Indices in flow data (Should be [missing_frames] - 1 excluding 0)
        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)
            if flow_joints is not None: # Flow data covers frames 2 through T
                skip_set1 = [frame_no-1 for frame_no in missing_frames_1 if frame_no > 0]
                flow_joints[idx][skip_set1, :1250] = np.zeros((len(skip_set1), 1250), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)
            if flow_joints is not None: # Flow data covers frames 2 through T
                skip_set2 = [frame_no-1 for frame_no in missing_frames_2 if frame_no > 0]
                flow_joints[idx][skip_set2, 1250:] = np.zeros((len(skip_set2), 1250), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def align_frames(joints, frames_cnt, MVC=150):
    """
    Align all sequences with the same frame length. 
        multiplied by the number of joints and the number of channels.
    
    Parameters:
    joints (list: numpy.ndarray): List length N containing arrays of chape (T, MVC).
                            N is the number of sequences.
                            T is the number of frames.
                            MVC is the number of joints and the number of channels.
    frames_cnt (numpy.ndarray): The number of frames for each sequence.
    MVC (int, optional): The number of joints and the number of channels. Default is 150.
    """
    num_skes = len(joints)
    max_num_frames = frames_cnt.max()  # 300
    aligned_joints = np.zeros((num_skes, max_num_frames, MVC), dtype=np.float32)

    for idx, video in enumerate(joints):
        num_frames = video.shape[0]
        num_bodies = 1 if video.shape[1] == int(MVC/2) else 2
        if num_bodies == 1:
            aligned_joints[idx, :num_frames] = np.hstack((video,
                                                          np.zeros_like(video)))
        else:
            aligned_joints[idx, :num_frames] = video

    return aligned_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 60))
    for idx, label in enumerate(labels):
        labels_vector[idx, label] = 1

    return labels_vector


def split_dataset(joints, details, evaluation, save_path, data_type='pose'):
    train_indices, test_indices = get_indices(
        details['Performer'], 
        details['Camera'],
        details['Setup'],
        evaluation)

    # Save labels and num_frames for each sequence of each data set
    train_labels = details['Label'][train_indices]
    test_labels = details['Label'][test_indices]

    train_x = joints[train_indices]
    train_y = one_hot_vector(train_labels)
    test_x = joints[test_indices]
    test_y = one_hot_vector(test_labels)

    save_name = osp.join(save_path, f'NTU60_{evaluation}-{data_type}.npz')
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)



def get_indices(performer, camera, setup, evaluation='CS'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'CS':  # Cross Subject (Subject IDs)
        train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                     17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                    24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
    elif evaluation == 'CSub': # Cross Subject (NTU120)
        train_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                     31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
                     58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
                     93, 94, 95, 97, 98, 100, 103]
        test_ids = [i for i in range(1, 107) if i not in train_ids]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
    elif evaluation == 'CV':  # Cross View (Camera IDs)
        train_ids = [2, 3]
        test_ids = 1
        # Get indices of test data
        temp = np.where(camera == test_ids)[0]  # 0-based index
        test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(camera == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)

    elif evaluation == 'CSet': # Cross Setup (NTU120)
        train_ids = [i for i in range(1, 33) if i % 2 == 0]  # Even setup
        test_ids = [i for i in range(1, 33) if i % 2 == 1]  # Odd setup

        # Get indices of test data
        for test_id in test_ids:
            temp = np.where(setup == test_id)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(setup == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
    return train_indices, test_indices


def concat_flowpose(skes_joints, flow_joints):
    """
    Concatenate the flow data to the pose data to create the flowpose data.
    NOTE: First frame of skeleton data is removed and sequence is shifted by 1 frame.
    """
    N, T, MVC = skes_joints.shape
    # Remove the first frame of the skeleton data (add a zero frame at the end)
    skes_joints = np.concatenate((skes_joints[:, 1:, :], np.zeros((N, 1, MVC))), axis=1)
    skes_joints = skes_joints.reshape((N, T, 2, 25, 3))
    flow_joints = flow_joints.reshape((N, T, 2, 25, 50)) # Assuming flow window = 5

    new_flow_joints = np.empty((N, T, 2, 25, 53), dtype=np.float32)
    new_flow_joints[:,:,:,:, :3] = skes_joints
    new_flow_joints[:,:,:,:, 3:] = flow_joints
    new_flow_joints = new_flow_joints.reshape((N, T, -1))

    return new_flow_joints


if __name__ == '__main__':
    evaluations = ['CS', 'CV'] if dataset == 'ntu' else ['CSub', 'CSet']

    # Load data statistics
    frames_cnt = np.loadtxt('./data/ntu/statistics/frames_cnt.txt', dtype=int)
    skes_name = np.loadtxt('./data/ntu/statistics/ntu_rgbd-available.txt', dtype=str)
    if dataset=='ntu120':
        frames_cnt = np.hstack((frames_cnt, np.loadtxt(frames_file, dtype=int)))
        skes_name = np.hstack((skes_name, np.loadtxt(skes_name_file, dtype=str)))
    details = get_details(skes_name)

    print(f'Dataset: {dataset}')
    if args.realign:
        print('Realigning dataset')
    if args.flow:
        print('Processing optical flow data\n')

    for evaluation in evaluations:
        train_indices, test_indices = get_indices(
            details['Performer'], 
            details['Camera'], 
            details['Setup'],
            evaluation)
        print(f'\tTrain indices length: {len(train_indices)}')
        print(f'\tTest indices length: {len(test_indices)}')
        print(f'Total skes names: {len(skes_name)}\n')

    # # If this realign=True, align the flow and pose data and save it
    # if args.realign:
    #     # Load data statistics
    #     frames_cnt = np.loadtxt(frames_file, dtype=int)  # frames_cnt
    #     skes_name = np.loadtxt(skes_name_file, dtype=str) # skeleton names
    #     details = get_details(skes_name)
    #     # # Create (or recreate) the annotations file for the dataset
    #     # # TODO: Do I need to create these annotations here, or earlier?
    #     # annotations = {}
    #     # for idx, ske_name in enumerate(skes_name):
    #     #     annotations[ske_name] = int(details['Label'][idx])
    #     # with open(osp.join(stat_path, 'ntu-rgbd-annotations.yaml'), 'w') as file:
    #     #     yaml.dump(annotations, file)
        
    #     # Load the raw data
    #     with open(raw_skes_joints_pkl, 'rb') as fr:
    #         skes_joints = pickle.load(fr)  # a list
    #     # Also load the flow if we pass the argument!
    #     if args.flow:
    #         with open(raw_flow_joints_pkl, 'rb') as fr:
    #             flow_joints = pickle.load(fr)
    #         print(f'Flow joints dtype: {flow_joints[0].dtype}', flush=True)
    #     else:
    #         flow_joints = None
    #     print(f'Loaded {len(skes_joints)} skeleton sequences and {len(flow_joints)} flow sequences', flush=True)

    #     # Translates the sequence to a new origin first non-zero frame of the first actor
    #     skes_joints = seq_translation(skes_joints, flow_joints)

    #     # Aligned to the same frame length
    #     skes_joints = align_frames(skes_joints, frames_cnt)
    #     if args.flow:
    #         flow_joints = align_frames(flow_joints, frames_cnt, MVC=2500)
    #         print(f'Full flow sequence shape: {flow_joints.shape}', flush=True)
    #         print(f'Full skeleton sequence shape: {skes_joints.shape}', flush=True)
    #         flow_joints = concat_flowpose(skes_joints, flow_joints)
    #         print(f'Full flowpose sequence shape: {flow_joints.shape}', flush=True)
    #         with open(raw_flowpose_pkl, 'wb') as f:
    #             pickle.dump(flow_joints, f, pickle.HIGHEST_PROTOCOL)
        
    #     print(f'Flowpose approximate size: {sys.getsizeof(flow_joints, 5)/1e9:.2f} GB', flush=True)
    #     # Generate train-test splits and save the data
    #     file_list = []
    #     for evaluation in evaluations:
    #         split_dataset(skes_joints, details, evaluation, save_path, data_type='pose')
    #         print(f'Saved pose {evaluation}', flush=True)
    #         file_list.append(osp.join(save_path, f'NTU60_{evaluation}-pose.npz'))
    #         # If flow arg is passed, also split the flowpose dataset
    #         if args.flow:
    #             split_dataset(flow_joints, details, evaluation, save_path, data_type='flowpose')
    #             print(f'Saved flowpose {evaluation}', flush=True)
    #             file_list.append(osp.join(save_path, f'NTU60_{evaluation}-flowpose.npz'))
    
    # else:
    #     file_list = []
    #     file_list += [osp.join(save_path, f'NTU60_{evaluation}-pose.npz') for evaluation in evaluations]
    #     if args.flow:
    #         file_list += [osp.join(save_path, f'NTU60_{evaluation}-flowpose.npz') for evaluation in evaluations]
    #     print(file_list, flush=True)

    # # Create the aligned dataset
    # create_aligned_dataset(file_list=file_list)
    # print('Aligned datasets created successfully!', flush=True)