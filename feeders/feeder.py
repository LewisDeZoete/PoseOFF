import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, "..")))

import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    """
    Feeder class for loading and processing dataset.

    Attributes:
        data_paths (dict): Dictionary containing paths to data files.
        data_path (str): Path to the specific modality data.
        label_path (str): Path to the label file.
        labels (dict): Dictionary containing the labels of the dataset.
        split (str): Indicates whether the dataset is for training or testing.
        random_choose (bool): If true, randomly choose a portion of the input sequence.
        random_shift (bool): If true, randomly pad zeros at the beginning or end of sequence.
        random_move (bool): If true, apply random movement transformations.
        window_size (int): The length of the output sequence.
        use_mmap (bool): If true, use mmap mode to load data, which can save the running memory.
        p_interval (list): List of intervals for cropping in valid_crop_resize.
        random_rot (bool): If true, rotate skeleton around xyz axis.
        vel (bool): If true, use motion modality.
        A (Any): Adjacency matrix for graph-based models.
        n_per_cls (np.ndarray): Number of samples per class.
        csum_n_per_cls (np.ndarray): Cumulative sum of samples per class.

    Methods:
        get_n_per_class(): Calculate the number of samples per class.
        sort(): Sort the data and labels based on the labels.
        __len__(): Return the number of samples in the dataset.
        __iter__(): Return the iterator object.
        __getitem__(index): Get the data and label for a given index.
        top_k(score, top_k): Calculate the top-k accuracy.
    """

    def __init__(
        self,
        data_paths,
        label_path,
        labels,
        modality,
        split="train",
        p_interval=[0.95],
        random_shift=False,
        random_choose=False,
        window_size=64,
        random_move=False,
        random_rot=False,
        average_flow=False,
        absolute_flow=False,
        no_flow=False,
        use_mmap=False,
        vel=False,
        sort=False,
        A=None,
    ):
        """
        Initialize the feeder.
        Args:
            data_paths (dict): Dictionary containing paths to data files.
            label_path (str): Path to the label file.
            labels (list): List of labels.
            modality (str): Modality of the data (e.g., 'rgb', 'depth').
            p_interval (list, optional): Probability interval for sampling valid_crop_resize. Defaults to [0.95].
            split (str, optional): Dataset split ('train', 'val', 'test'). Defaults to 'train'.
            random_choose (bool, optional): Whether to randomly choose frames. Defaults to False.
            random_shift (bool, optional): Whether to randomly shift frames. Defaults to False.
            random_move (bool, optional): Whether to randomly move frames. Defaults to False.
            random_rot (bool, optional): Whether to randomly rotate frames. Defaults to False.
            window_size (int, optional): Size of the window for sampling frames. Defaults to 64.
            use_mmap (bool, optional): Whether to use memory-mapped files. Defaults to False.
            vel (bool, optional): Whether to use velocity information. Defaults to False.
            sort (bool, optional): Whether to sort the data. Defaults to False.
            A (optional): Adjacency matrix. Defaults to None.
        """
        self.data_paths = data_paths
        self.data_path = self.data_paths[f"{modality}_path"]
        self.label_path = label_path
        self.labels = labels
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if split == "train":
            self.p_interval = p_interval
            self.random_shift = random_shift
            self.random_choose = random_choose
            self.window_size = window_size
            self.random_move = random_move
            self.random_rot = random_rot
        else:
            self.p_interval = [0.95]
            self.random_shift = False
            self.random_choose = False
            self.window_size = 64
            self.random_move = False
            self.random_rot = False
        self.average_flow = average_flow # Average and absolute flow must be the same for train and test
        self.absolute_flow = absolute_flow # NOTE: cannot have both average and absolute flow!
        self.no_flow = no_flow

        # self.normalization = normalization TODO: REMOVE all normalization references

        self.use_mmap = use_mmap
        self.vel = vel
        self.A = A
        if sort:
            self.get_n_per_class()
            self.sort()
        # if normalization:
        #     self.get_mean_map()

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
        # NOTE: here we assume that we're working with preprocessed (flowpose) data
        item_key = list(self.labels.keys())[index]
        item_path = f"{self.data_path}{item_key}.npy"
        label = self.labels[item_key]

        data_numpy = np.load(item_path)
        valid_frame = data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)
        valid_frame_num = np.sum(np.squeeze(valid_frame).sum(-1) != 0)
        # # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size
        )
        mask = abs(data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)) > 0
        # Apply optional transforms
        # if self.normalization:
        #     data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.no_flow:
            data_numpy = data_numpy[:3]
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(
                data_numpy, transform_candidate=[-0.1, -0.05, 0.0, 0.05, 0.1]
            )
        # if self.random_rot:
        #     data_numpy = tools.random_rot(data_numpy)
        # TODO: Test random_rot function
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


if __name__ == "__main__":
    from config.argclass import ArgClass
    import time
    from torch.utils.data import DataLoader

    arg = ArgClass("./config/custom_pose/train_joint_infogcn.yaml")

    feeder = Feeder(**arg.feeder_args)
    # feeder = Feeder(**arg.feeder_args, split="test")
    dataloader = DataLoader(feeder, batch_size=arg.batch_size, shuffle=True)

    start = time.time()
    for epoch, (data_numpy, label, mask, index) in enumerate(dataloader):
        break

    print(f"\nTotal samples: {len(feeder)}")
    print(f"Time taken for one epoch loading: {time.time() - start:.2f} seconds")
    print(f"Data shape: {data_numpy.shape}")
    print(f"Mask shape: {mask.shape}")
