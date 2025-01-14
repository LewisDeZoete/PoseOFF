import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, labels, ext='.pt', p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=64, debug=False, use_mmap=False,
                 vel=False, sort=False, A=None):
        """
        :param data_path:
        :param labels: `dict` containing the labels of the dataset
        :param ext: extension of the files
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.labels = labels
        self.ext = ext
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        # self.normalization = normalization TODO: REMOVE all normalization references
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.vel = vel
        self.A = A
        # self.load_data()
        if sort:
            self.get_n_per_class()
            self.sort()
        # if normalization:
        #     self.get_mean_map()

    # def load_data(self):
    #     # data: N C V T M
    #     npz_data = np.load(self.data_path)
    #     if self.split == 'train':
    #         self.data = npz_data['x_train']
    #         self.labels = np.argmax(npz_data['y_train'], axis=-1)
    #     elif self.split == 'test':
    #         self.data = npz_data['x_test']
    #         self.labels = np.argmax(npz_data['y_test'], axis=-1)
    #     else:
    #         raise NotImplementedError('data split only supports train/test')
    #     nan_out = np.isnan(self.data.mean(-1).mean(-1))==False
    #     self.data = self.data[nan_out]
    #     self.labels = self.labels[nan_out]
    #     self.sample_name = [self.split + '_' + str(i) for i in range(len(self.data))]
    #     N, T, _ = self.data.shape
    #     if self.A is not None:
    #         self.data = self.data.reshape((N*T*2, 25, 3))
    #         self.data = np.array(self.A) @ self.data # x = N C T V M
    #     self.data = self.data.reshape(N, T, 2, 25, 3).transpose(0, 4, 1, 3, 2)
    #     # self.data -= self.data[:,:,:,1:2]

    def get_n_per_class(self):
        self.n_per_cls = np.zeros(len(self.labels), dtype=int)
        for label in self.labels:
            self.n_per_cls[label] += 1
        self.csum_n_per_cls = np.insert(np.cumsum(self.n_per_cls), 0, 0)

    def sort(self):
        sorted_idx = self.labels.argsort()
        self.data = self.data[sorted_idx]
        self.labels = self.labels[sorted_idx]

    # def get_mean_map(self):
    #     data = self.data
    #     N, C, T, V, M = data.shape
    #     self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
    #     self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        item_key = list(self.labels.keys())[index]
        item_path = f"{self.data_path}{item_key}{self.ext}"
        label = self.labels[item_key]

        data_numpy = torch.load(item_path, map_location=self.device)
        # valid_frame = data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)
        # valid_frame_num = np.sum(np.squeeze(valid_frame).sum(-1) != 0)
        # # reshape Tx(MVC) to CTVM
        # data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        # mask = (abs(data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)) > 0)
        # # TODO: Implement transforms AFTER moving augments in lib to feeders/tools.py
        # if self.normalization:
        #     data_numpy = (data_numpy - self.mean_map) / self.std_map
        # if self.random_shift:
        #     data_numpy = tools.random_shift(data_numpy)
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
            # data_numpy = tools.random_move(data_numpy, transform_candidate=[-0.1, -0.05, 0.0, 0.05, 0.1])

        # return data_numpy, label, mask, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__=='__main__':
    from lib.utils.objects import ArgClass     
    arg = ArgClass('./config/custom_pose/train_joint.yaml')

    # Get the annotation file and define checkpoint file
    classes = arg.classes
    
    feeder = Feeder(data_path=arg.dataloader['data_path'], 
                    labels=arg.labels, 
                    split='train')
    print(feeder[0])