import numpy as np
import os
import os.path as osp
from config.argclass import ArgClass
from einops import rearrange
import argparse

parser = argparse.ArgumentParser(prog="flowpose_seq_transform")

parser.add_argument(
    '--dilation',
    dest='dilation',
    default=None,
    type=int,
    help='Overwrite the dilation value from the yaml config.'
)

parsed = parser.parse_args()
arg = ArgClass("./config/ucf101/base.yaml")
labels = arg.feeder_args['labels']

# Define the paths
data_labels_root = './data/ucf101/statistics/'
flowpose_path = './data/ucf101/flowpose'
save_path = './data/ucf101/aligned_data'

# Create the save directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

def combine_data(all_paths):
    """
    Load and combine all of the flowpose data from the given paths.
    
    Args:
        all_paths (list): List of paths to the flowpose data files (numpy).
            These numpy files should be of shape (C, T, V, M) where:
            C is the number of channels (53),
            T is the number of frames (300)
            V is the number of joints (17),
            M is the number of modalities (2).
    Returns:
        numpy.ndarray: Combined data of shape (N, T, M*V*C) where N is the number of sequences.
    """
    T,M,V,C = (300,2,17,53)
    num_files = len(all_paths)
    aligned_joints = np.zeros((num_files, T, M*V*C), dtype=np.float32)

    for idx, video_path in enumerate(all_paths):
        data = np.load(video_path)
        data = rearrange(data, 'C T V M -> T (M V C)', C=C, T=T, V=V, M=M)
        aligned_joints[idx] = data

    return aligned_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 101))
    for idx, label in enumerate(labels):
        labels_vector[idx, label] = 1

    return labels_vector


def get_indices(evaluation: int, data_labels_root: str):
    '''
    Get the indices of the training and testing data for the given evaluation number.
    Args:
        evaluation (int): The evaluation number (1, 2, or 3).
        data_labels_root (str): Root path to where the evaluation lists live
    '''
    train_indices = np.empty(0)
    test_indices = np.empty(0)
    
    # Open and read the train and test list files
    with open(osp.join(data_labels_root, f'trainlist0{evaluation}.txt'), 'r') as f:
        train_ids = f.readlines()
        train_ids = [line.strip() for line in train_ids]
    with open(osp.join(data_labels_root, f'testlist0{evaluation}.txt'), 'r') as f:
        test_ids = f.readlines()
        test_ids = [line.strip() for line in test_ids]
    
    label_keys = list(labels.keys())

    # Find indices for train_ids
    for train_id in train_ids:
        if train_id in label_keys:
            train_indices = np.hstack((train_indices, label_keys.index(train_id))).astype(int)

    # Find indices for test_ids
    for test_id in test_ids:
        if test_id in label_keys:
            test_indices = np.hstack((test_indices, label_keys.index(test_id))).astype(int)

    return train_indices, test_indices


def split_dataset(joints, labels, evaluation, dilation, save_path, data_labels_root):
    """
    Splits the dataset into training and testing sets based on the evaluation index,
    processes the labels into one-hot vectors, and saves the resulting data to a file.

    Args:
        joints (numpy.ndarray): A numpy array containing joint data for all sequences.
        labels (numpy.ndarray): A numpy array containing the label number corresponding to the sequences.
        evaluation (int): An integer representing the evaluation index used to split the dataset.
        save_path (str): The directory path where the resulting dataset file will be saved.

    Returns:
        None: The function saves the processed data to a `.npz` file and does not return anything.

    Notes:
        - The function assumes the existence of a `get_indices` function that provides the
          train and test indices based on the evaluation index.
        - The function also assumes the existence of a `one_hot_vector` function to convert
          labels into one-hot encoded vectors.
        - The saved file is named in the format `ucf101_0{evaluation}.npz` and contains the
          following keys:
            - `x_train`: Training set joint data.
            - `y_train`: One-hot encoded training labels.
            - `x_test`: Testing set joint data.
            - `y_test`: One-hot encoded testing labels.
    """
    train_indices, test_indices = get_indices(evaluation, data_labels_root)

    # Save labels and num_frames for each sequence of each data set
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    train_x = joints[train_indices]
    train_y = one_hot_vector(train_labels)
    test_x = joints[test_indices]
    test_y = one_hot_vector(test_labels)

    save_name = osp.join(save_path, f'ucf101_0{evaluation}_D{dilation}.npz')
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)


if __name__ == '__main__':
    from sys import getsizeof
    # Get the paths for the flowpose data
    flowpose_paths = [osp.join(flowpose_path, item)+'.npy' for item in list(arg.feeder_args['labels'].keys())]

    # Combine all the data into a single numpy array (N, T, M*V*C)
    joints = combine_data(flowpose_paths)
    print(f'Combined data shape: {joints.shape}')
    print(f'Combined data size: {getsizeof(joints)/1e6} MB')

    # For each evaluation, split the dataset and save it
    print('Processing evaluations...')
    for evaluation in range(1, 4):
        split_dataset(joints, 
                      labels=np.array(list(arg.feeder_args['labels'].values())), 
                      evaluation=evaluation,
                      dilation=parsed.dilation,
                      save_path=save_path, # I just want to save it inside the flowpose folder...
                      data_labels_root=data_labels_root)
        print(f'\tProcessed evaluation {evaluation}.')
        print(f'\t\tSaved to: {save_path}/ucf101_0{evaluation}_D{parsed.dilation}.npz')
    print('Completed processing all evaluations.')
