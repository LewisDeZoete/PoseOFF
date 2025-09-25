#!/usr/bin/env python3
# CREDIT: Not too sure

import os
import os.path as osp
import sys
import pickle

import argparse
import numpy as np
import mmnpz
from numpy.lib.format import open_memmap

from data_gen.utils import pre_normalisation

max_bodies = 1
num_joint = 20
max_frame = 192

def read_xyz(sample_folder, max_bodies=1, num_joint=20):
    print(sample_folder)
    sample_all_files = os.listdir(sample_folder)
    sample_skel_files = \
        [filename for filename in sample_all_files if filename.endswith('skeletons.txt')]
    sample_skel_files.sort()
    # NOTE: Some of the folders contain mixed samples, but the skeleton files seem
    # not to have been mixed.
    sample_data = {
        'sample_name': list(sample_folder.split('/'))[-1],
        'num_frames': len(sample_skel_files),
        'frame_names':
        [filename.strip('_skeletons.txt') for filename in sample_skel_files]
    }

    # Create an empty array to hold skeleton data
    data = np.zeros((3, len(sample_skel_files), num_joint, max_bodies))

    for frame_no, skel_frame_file_name in enumerate(sample_skel_files):
        print(skel_frame_file_name)
        with open(osp.join(sample_folder, skel_frame_file_name), 'r') as f:
            skel_frame = f.readlines()[1:]
            skel_frame = [limb.split(',')[:-1] for limb in skel_frame]
        for joint_no, joint in enumerate(skel_frame):
            data[:, frame_no, joint_no, 0] = joint
    sample_data['data'] = data

    return sample_data

def remove_mixed_frame_samples(sample_skel_files):
    '''
    Samples with mixed frames...
    'view_1/a01_s08_e00', 'view_1/a01_s08_e02', 'view_1/a01_s08_e03'
    TODO: Test whether the correct actor's samples (s08) has more or less frames...
    '''
    sample_skel_files = [list(filename.split('_'))[-1] for filename in sample_skel_files]
    bad_list = {'sample_name': [], 'index': []}
    for i in range(1, len(sample_skel_files)):
        diff = int(sample_skel_files[i])-int(sample_skel_files[i-1])
        if diff < 0 or diff > 1000:
            bad_list['sample_name'].append(sample_skel_files[i])
            bad_list['index'].append(i)

    return bad_list

def has_duplicates(sample_skel_files):
    framenum_list = [
        int(list(filename.split('_'))[1]) for filename in sample_skel_files
    ]
    return len(framenum_list) != len(set(framenum_list))


def gendata(data_path,
            label_path,
            out_path,
            ignored_sample_path=None,
            benchmark='123',
            part='val',
            ):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_names = []
    sample_labels = []
    for view in os.listdir(data_path):
        for filename in os.listdir(osp.join(data_path, view)):
            if filename in ignored_samples:
                continue
            action_id = int(
                filename[filename.find('a') + 1:filename.find('a') + 3])
            subject_id = int(
                filename[filename.find('s') + 1:filename.find('s') + 3])
            environment_id = int(
                filename[filename.find('e') + 1:filename.find('e') + 3])
            view_id = int(view[-1])

            if benchmark == '123':
                istraining = (view_id in [1, 2])
            elif benchmark == '231':
                istraining = (view_id in  [2, 3])
            elif benchmark == '132':
                istraining = (view_id in  [1, 3])
            else:
                raise ValueError()

            if part == 'train':
                issample = istraining
            elif part == 'val':
                issample = not (istraining)
            else:
                raise ValueError()

            if issample:
                sample_names.append(osp.join(view, filename))
                sample_labels.append(action_id - 1)

    with open(f"{label_path}/{benchmark}_{part}_label.pkl", 'wb') as f:
        pickle.dump((sample_names, list(sample_labels)), f)

    fp = np.zeros((len(sample_labels), 3, max_frame, num_joint, max_bodies), dtype=np.float32)
    for i, s in enumerate(sample_names):
        sample_folder = osp.join(data_path, s)
        sample_all_files = os.listdir(sample_folder)
        sample_skel_files = \
            [filename for filename in sample_all_files if filename.endswith('skeletons.txt')]

        # Some samples have strange mixed samples, these can be removed
        if has_duplicates(sample_skel_files):
            bad_samples.append(s)
            continue

        # sample_data = read_xyz(
        #     osp.join(data_path, s), max_bodies=max_bodies, num_joint=num_joint)
        # fp[i, :, 0:sample_data['num_frames'], :, :] = sample_data['data']
        # break
    return fp

    # fp = pre_normalisation(fp)
    # np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='../Datasets/N-UCLA/multiview_action'
    )
    parser.add_argument(
        '--label_path', default='./data/ucla/statistics/'
    )
    parser.add_argument(
        '--ignored_sample_path',
        default='./data/ucla/statistics/NW-UCLA_samples_with_missing_skeletons.txt'
    )
    parser.add_argument(
        '--out_path',
        default='./data/ucla/pose/'
    )

    benchmark = ['123','231','132']
    part = ['train', 'val']
    arg = parser.parse_args()

    # Make the output datafolder
    os.makedirs(arg.out_path, exist_ok=True)

    print(f"Data path: {arg.data_path}")
    print(f"Output path: {arg.out_path}")
    print(f"Ignored samples file: {arg.ignored_sample_path}")
    print(f"Label path: {arg.label_path}")
    bad_samples = []
    for b in benchmark:
        for p in part:
            fp = gendata(
                arg.data_path,
                arg.label_path,
                arg.out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p,
            )
    # print(fp[0, :, 0, :, 0])
    print(bad_samples)

    # # NOTE: There are a number of samples (I don't yet know exactly which view etc.) that have strange crossover with other actors...
    # # Not sure how I can go about parsing these out
    # # Possibly I could parse the fileList.txt since some of the lines in this list have two samples...
    # # e.g. a01_s08_e02
    # # sample_folder = osp.join(arg.data_path, ult_max['sample_name'])
    # sample_folder = osp.join(arg.data_path, 'view_1/a01_s08_e02')
    # sample_data = read_xyz(sample_folder)
    # bad_list = remove_mixed_frame_samples(sample_data['frame_names'])
    # print(bad_list)
    # print(f"Sample name: {sample_data['sample_name']}")
    # print(f"Frame names: {len(sample_data['frame_names'])}")
    # print(f"Sample data shape: {sample_data['data'].shape}")
