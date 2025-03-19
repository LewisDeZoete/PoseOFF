import numpy as np
import pickle

from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, labels=None, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, p_interval=1, window_size=64, average_flow=False, absolute_flow=False, 
                 no_flow=False, normalization=False, debug=False, use_mmap=False, vel=False, sort=False, A=None):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set (default='train')
        :param random_choose: If true, randomly choose a portion of the input sequence (default=False)
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence (default=False)
        :param random_move: If true, perform random move (rotation, scale and translation) (default=False)
        :param random_rot: rotate skeleton around xyz axis (default=False)
        :param p_interval: proportion of valid frames to be cropped as single proportion or range (default=1)
        :param window_size: the length of the output sequence (default=64)
        :param average_flow: average value of x and y coordinates of the optical flow windows (default=False)
        :param absolute_flow: if 
        :param normalization: If true, normalize input sequence (default=False)
        :param debug: If true, only use the first 100 samples (default=False)
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory (default=False)
        :param vel: use motion modality or not (default=False)
        :param sort: 
        :param A: adjacency matrix
        """

        self.data_path = data_path
        self.label_path = label_path
        self.labels = labels
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.p_interval = p_interval
        self.window_size = window_size
        self.average_flow = average_flow
        self.absolute_flow = absolute_flow # NOTE: cannot have both average and absolute flow!
        self.no_flow = no_flow
        self.normalization = normalization
        self.debug = debug
        self.use_mmap = use_mmap
        self.vel = vel
        self.A = A
        self.load_data()
        if sort:
            self.get_n_per_class()
            self.sort()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.argmax(npz_data['y_train'], axis=-1)
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.argmax(npz_data['y_test'], axis=-1)
        else:
            raise NotImplementedError('data split only supports train/test')
        nan_out = np.isnan(self.data.mean(-1).mean(-1))==False
        self.data = self.data[nan_out]
        self.label = self.label[nan_out]
        # self.sample_name = [self.split + '_' + str(i) for i in range(len(self.data))]
        N, T, _ = self.data.shape
        C = 53 if self.data.shape[-1] > 150 else 3  # If the dataset doesn't have flow, this is false
        if self.A is not None:
            self.data = self.data.reshape((N*T*2, 25, C))
            self.data = np.array(self.A) @ self.data
        self.data = self.data.reshape(N, T, 2, 25, C).transpose(0, 4, 1, 3, 2) # N C T V M
        if self.no_flow: # If no flow argument is passed, take first three channels
            self.data = self.data[:, :3, ...]
            


    def get_n_per_class(self):
        self.n_per_cls = np.zeros(len(self.label), dtype=int)
        for label in self.label:
            self.n_per_cls[label] += 1
        self.csum_n_per_cls = np.insert(np.cumsum(self.n_per_cls), 0, 0)

    def sort(self):
        sorted_idx = self.label.argsort()
        self.data = self.data[sorted_idx]
        self.label = self.label[sorted_idx]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame = data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)
        valid_frame_num = np.sum(np.squeeze(valid_frame).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, 
                                             valid_frame_num, 
                                             self.p_interval, 
                                             self.window_size)
        mask = (abs(data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)) > 0)
        # if self.random_rot: TODO: Fix and test for flowpose data
        #     data_numpy = tools.random_rot(data_numpy)
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        if self.average_flow:
            data_numpy = tools.average_flow(data_numpy)
        if self.absolute_flow:
            data_numpy = tools.absolute_flow(
                data_numpy, window_mean=self.absolute_flow["window_mean"]
            )
        return data_numpy, label, mask, index

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
    from config.argclass import ArgClass
    from torch.utils.data import DataLoader

    arg = ArgClass('config/nturgbd-cross-subject/train_base.yaml')
    
    feeder = Feeder(**arg.feeder_args)

    dataloader = DataLoader(feeder, batch_size=60, shuffle=False, pin_memory=True)

    for epoch, (data_numpy, label, mask, index) in enumerate(dataloader):
        if epoch == 10:
            break
    
    print(f"\nTotal samples: {len(feeder)}")
    # print(f"Time taken for one epoch loading: {time.time() - start:.2f} seconds")
    print(f"Data shape: {data_numpy.shape}")
    print(f"Mask shape: {mask.shape}")