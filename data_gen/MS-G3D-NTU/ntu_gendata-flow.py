import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '../..')))

import pickle
import argparse

from data_gen.utils.preprocess import pre_normalization
import time
# from tqdm import tqdm
import multiprocessing
import numpy as np
import os

# https://arxiv.org/pdf/1604.02808.pdf, Section 3.2
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]

max_body_true = 2
max_body_kinect = 4

num_joint = 25
max_frame = 300

def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):
    # TODO: Implement flowpose sampling here!
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_paths, out_path, transforms, ignored_sample_path=None, benchmark='xview', part='eval'):
    print(f"### START GENERATION: benchmark {benchmark}, part {part}.", flush=True)
    if ignored_sample_path is None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in sorted(os.listdir(data_paths['rgb'])): # TODO: Use the ntu_...-available.txt
        filename=filename.split('.')[0].split('_')[0]
        if filename in ignored_samples: # Can then ignore this!
            continue

        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    # Fill in the data tensor `fp` one training example a time
    for i, s in enumerate(sample_name):
        vid = s+'_rgb.avi'
        skel = s+'.skeleton'
        flow = transforms(os.path.join(data_paths['rgb'], vid))
        pose = read_xyz(os.path.join(data_paths['pose'], skel), max_body=max_body_kinect, num_joint=num_joint)
        print(f"Flow shape: {flow.shape}, Pose shape: {pose.shape}")
        quit()

        # fp[i, :, 0:data.shape[1], :, :] = data

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

    print(f"### DONE GENERATION: benchmark {benchmark}, part {part}.", flush=True)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
#     parser.add_argument('--data_path', default='../Datasets/NTU_RGBD/nturgb+d_skeletons/')
#     parser.add_argument('--ignored_sample_path',
#                         default='./data/ntu/NTU_RGBD_samples_with_missing_skeletons.txt')
#     parser.add_argument('--out_folder', default='./data/ntu/')
#     parser.add_argument('--n_cores', default=1, type=int, help='Number of cores to run data generation by multiprocessing.')

#     benchmark = ['xsub', 'xview']
#     part = ['train', 'val']
#     arg = parser.parse_args()

#     func_args = []
#     for b in benchmark:
#         for p in part:
#             out_path = os.path.join(arg.out_folder, b)
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
            
#             f_arg = (arg.data_path, out_path, arg.ignored_sample_path, b, p)
#             func_args.append(f_arg)

#     cpu_available = multiprocessing.cpu_count()
#     if arg.n_cores > 1:
#         num_args = len(func_args)
#         arg.n_cores = min(cpu_available, num_args, arg.n_cores)
#     print(f"Cores: {cpu_available} Available, {arg.n_cores} Chosen.", flush=True)

#     start_t = time.time()
#     pool = multiprocessing.Pool(arg.n_cores)
#     pool.starmap(gendata, func_args)
    
#     pool.join()
#     pool.close()
    
#     end_t = time.time()

#     print(f"@@@ DONE all processing in {end_t - start_t:.2f}s @@@", flush=True)


# -----------------------------------------------------------------------------

import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

from utils import LoadVideo, get_class_by_index, GetFlow
from config.argclass import ArgClass
import argparse
import time
from torchvision.models.optical_flow import raft_large
import torch
import torchvision.transforms.v2 as v2
import numpy as np



# Get the arg object and create the classes
arg = ArgClass(arg='./config/ucf101/train_joint_infogcn.yaml')
arg.extractor['preprocessed'] = False # Override this value, since this is gendata script!
transform_args = arg.extractor['flow']

# Get the device
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

# Create the model, move it to device and turn to eval mode
weights = torch.load(transform_args['weights'], weights_only=True, map_location=device)
model = raft_large(progress=False)
model.load_state_dict(weights)
model = model.eval().to(device)
transforms = v2.Compose([
    LoadVideo(max_frames=300),
    v2.Resize(size=transform_args['imsize']),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # map [0, 1] into [-1, 1]
    GetFlow(model=model, device=device, minibatch_size=transform_args['minibatch_size'])
    ])

benchmark = ['xsub', 'xview']
part = ['train', 'val']

# func_args = []
# for b in benchmark:
#     for p in part:
#         out_path = os.path.join(, b)
#         if not os.path.exists(out_path):
#             os.makedirs(out_path)
    
#         f_arg = (arg.data_path, out_path, arg.ignored_sample_path, b, p)
#         func_args.append(f_arg)

if torch.cuda.is_available():
    print('Using GPU')


data_paths = {'rgb': '../Datasets/NTU_RGBD/nturgb+d_rgb/', 'pose': '../Datasets/NTU_RGBD/nturgb+d_skeletons/', 'flow': '../Datasets/NTU_RGBD/nturgb+d_flowpose/'}
out_path = 'TMP'
ignored_sample_path = './data/ntu/NTU_RGBD_samples_with_missing_skeletons.txt'
gendata(data_paths, out_path, transforms, ignored_sample_path, 'xview', 'train')