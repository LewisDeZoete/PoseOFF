import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    """
    TODO: Based on `split` input, split the data based on the 3 train/test splits outlined in the .txt files in the dataset folder
    Feeder class for loading and processing dataset.
    NOTE: If you're finding issues with this feeder, it might be caused by 
    the renaming of folders and videos, and to keep consistencies 
    in naming (eg. HandstandPushups -> HandStandPushups, HandstandWalking etc.)

    Attributes:
        data_path (str): Path to flowpose data.
        eval (int): evaluation benchmark split (ucf101 comes with 3, labeeled 1,2 & 3)
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
        data_paths: str,
        eval: int = 1,
        label_path=None,
        labels=None,
        split: str = "train",
        random_choose=False,
        random_shift: bool = False,
        random_move: bool = False,
        random_rot: bool = False,
        p_interval: list[float] = [1.0],
        window_size: int = 64,
        average_flow: bool = False,
        absolute_flow: bool = False,
        no_flow: bool = False,
        # normalisation=False,
        use_mmap: bool = False,
        vel: bool = False,
        sort: bool = False,
        A=None,
    ):
        self.eval = int(eval)
        self.data_path = data_paths[self.eval]
        # self.label_path = label_path
        # self.labels = labels
        self.split = split
        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")
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
        self.absolute_flow = (
            absolute_flow # NOTE: cannot have both average and absolute flow!
        )
        self.no_flow = no_flow
        self.use_mmap = use_mmap
        self.vel = vel
        self.A = A
        self.load_data()
        if sort:
            self.get_n_per_class()
            self.sort()

    def load_data(self):
        # data: N T (MVC)
        npz_data = np.load(self.data_path)
        if self.split == "train":
            self.data = npz_data["x_train"]
            self.labels = np.argmax(npz_data["y_train"], axis=-1)
        elif self.split == "test":
            self.data = npz_data["x_test"]
            self.labels = np.argmax(npz_data["y_test"], axis=-1)
        else:
            raise NotImplementedError("data split only supports train/test")
        nan_out = np.isnan(self.data.mean(-1).mean(-1)) == False
        self.data = self.data[nan_out]
        self.labels = self.labels[nan_out]
        # self.sample_name = [self.split + '_' + str(i) for i in range(len(self.data))]
        N, T, _ = self.data.shape
        C = (
            53 if self.data.shape[-1] > 150 else 3
        )  # If the dataset doesn't have flow, this is false
        if self.A is not None:
            self.data = self.data.reshape((N * T * 2, 17, C))
            self.data = np.array(self.A) @ self.data
        self.data = self.data.reshape(N, T, 2, 17, C).transpose(
            0, 4, 1, 3, 2
        )  # N C T V M
        if self.no_flow:  # If no flow argument is passed, take first three channels
            self.data = self.data[:, :3, ...]

    def get_n_per_class(self):
        self.n_per_cls = np.zeros(len(self.labels), dtype=int)
        for label in self.labels:
            self.n_per_cls[label] += 1
        self.csum_n_per_cls = np.insert(np.cumsum(self.n_per_cls), 0, 0)

    def sort(self):
        sorted_idx = self.labels.argsort()
        self.data = self.data[sorted_idx]
        self.labels = self.labels[sorted_idx]

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.labels[index]
        data_numpy = np.array(data_numpy)

        valid_frame = data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)
        valid_frame_num = np.sum(np.squeeze(valid_frame).sum(-1) != 0)
        # # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size
        )
        mask = abs(data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)) > 0
        # Apply optional transforms
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_padding(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(
                data_numpy, transform_candidate=[-0.1, -0.05, 0.0, 0.05, 0.1]
            )
        # if self.random_rot: TODO: Fix and test for flowpose data
        #     data_numpy = tools.random_rot(data_numpy)
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

    # srun --mem-per-cpu=40G python feeders/ucf101.py

    embed =  'base'
    arg = ArgClass(f"./config/ucf101/train_{embed}.yaml")
    arg.evaluation = 1

    train_feeder = Feeder(**arg.feeder_args, eval=arg.evaluation, split="train")
    test_feeder = Feeder(**arg.feeder_args, eval=arg.evaluation, split="test")
    
    dataloader = DataLoader(train_feeder, 
                            batch_size=arg.batch_size,
                            shuffle=False,
                            pin_memory=True)

    start = time.time()
    for epoch, (data_numpy, label, mask, index) in enumerate(dataloader):
        if epoch == 10:
            break

    # Print the shapes of the data and mask
    # and the first two frames of the first two persons
    print(f"\nTotal samples: {len(train_feeder)}")
    print(f"Time taken for one epoch loading: {time.time() - start:.2f} seconds")
    print(f"Data shape: {data_numpy.shape}") # (B (256), C (3/53), T (64), V (17), M (2))
    print(f"Label shape: {label.shape}") # (B)
    print(f"Mask shape: {mask.shape}") # (B, 1, 64, 1, 2)

    print(f"Frame 0, person 0: {data_numpy[0, :, 0, 0, 0]}")
    print(f"Frame 0, person 1: {data_numpy[0, :, 0, 0, 1]}\n")
    print(f"Frame 1, person 0: {data_numpy[0, :, 1, 0, 0]}")
    print(f"Frame 1, person 1: {data_numpy[0, :, 1, 0, 1]}")