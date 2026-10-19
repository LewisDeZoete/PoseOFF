#!/usr/bin/env python3
import numpy as np
import mmnpz

from torch.utils.data import Dataset

from feeders import tools

# 0-based indices (NTU joint order) of the 17 joints that can be derived from COCO-17
# keypoints: SpineBase, SpineMid, Neck, Head, L/R shoulder, elbow, wrist, L/R hip,
# knee, ankle, SpineShoulder. Excludes hands, feet, hand tips and thumbs.
NTU17_JOINTS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]


class Feeder(Dataset):
    def __init__(
            self,
            data_paths: dict,
            eval=None,
            split: str ="train",
            random_choose: bool = False,
            random_shift: bool = False,
            random_move: bool = False,
            random_rot: bool = False,
            p_interval: list[float] = [1.0],
            window_size: int = 64,
            pad_method: str = "last_frame",
            average_flow: bool = False,
            absolute_flow: bool = False,
            no_flow: bool = False,
            skel_displacement: bool = False,
            no_Z: bool = False,
            obs_ratio: float = 1.0,
            normalisation: bool = False,
            debug: bool = False,
            use_mmap: bool = True,
            vel: bool = False,
            A=None,
            num_joints: int = 25,
            num_persons: int = 2,
            joint_subset: bool = False,
            stats_path: str = None,
            *args,
            **kwargs
            ):
        """Feeder class for the NTU60 and NTU120 datasets.

        Args:
            data_path (str):
            eval (str): Evaluation set (CS/CV, CSub/CSet for ntu120)
            label_path (str):
            split (str): Training set or test set. Default is 'train'.
            random_choose (bool): If true, randomly choose a portion of the input sequence. Default is False.
            random_shift (bool): If true, randomly pad zeros at the begining or end of sequence. Default is False.
            random_move (bool): If true, perform random move (rotation, scale and translation). Default is False.
            random_rot (bool): Rotate skeleton around xyz axis. Default is False.
            p_interval list[float]: Proportion of valid frames to be cropped as single proportion or range. Default is 1.
            window_size (int): The length of the output sequence. Default is 64.
            pad_method (str): Method used for auto-padding data to correct window_size,
                either "last_frame" or "replay". Default is "last_frame".
            average_flow (bool): Average value of x and y coordinates of the optical flow windows. Default is False.
            absolute_flow (bool): If true, calculates the absolute value of all flow vectors in flow window. Default is False.
            no_flow (bool): If true, remove flow windows from data. Default is False.
            skel_displacement (bool): Use skel joint displacement in place of flow features. Default is False.
            no_Z (bool): If true, remove the z-component of skeleton keypoints. Default is False.
            obs_ratio (float): Observation ratio, 1.0 is full obs, 0.0 is no frames. Default is 1.0.
            normalisation (bool): If true, normalise input sequence. Default is False.
            debug (bool): If true, only use the first 100 samples. Default is False.
            use_mmap (bool): If true, use mmap mode to load data, which can save the running memory. Default is False.
            vel (bool): Use motion modality or not. Default is False.
            A: Adjacency matrix
            num_joints (int): Joints per body in the stored data (25 for NTU, 17 for COCO). Default is 25.
            num_persons (int): Bodies per frame in the stored data. Default is 2.
            joint_subset (bool): If given, keep only the COCO joints (indices into the stored
                joint order) when loading, e.g. NTU17_JOINTS. mean/std maps are subset to match.
            stats_path (str): Optional .npz holding mean_map/std_map from the *source* dataset, so
                new data is normalised with the pretrained model's statistics. Default is the data file.
        """

        self.eval = eval
        self.data_path = data_paths[eval]
        self.split = split
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val' or 'test'")
        if split == "train":
            self.p_interval = p_interval
            self.random_shift = random_shift
            self.random_choose = random_choose
            self.window_size = window_size
            self.pad_method = pad_method
            self.random_move = random_move
            self.random_rot = random_rot
        else:
            self.p_interval = [0.95]
            self.random_shift = False
            self.random_choose = False
            self.window_size = window_size
            self.pad_method = pad_method
            self.random_move = False
            self.random_rot = False
        if average_flow and absolute_flow:
            print("Cannot simultaneously calculate absolute and average optical flow...")
            quit()
        self.average_flow = average_flow
        self.absolute_flow = (
            absolute_flow
        )
        self.no_flow = no_flow
        self.skel_displacement = skel_displacement
        self.no_Z = no_Z
        self.obs_ratio = float(obs_ratio)
        self.normalisation = normalisation
        self.debug = debug
        self.use_mmap = use_mmap
        self.vel = vel
        self.A = A
        self.V_total = num_joints          # joints stored in the file
        self.M = num_persons
        self.joint_subset = None if not joint_subset else np.asarray(NTU17_JOINTS, dtype=int)
        self.V = self.V_total if not joint_subset else len(NTU17_JOINTS)
        self.stats_path = stats_path
        self.data = None  # defer loading (lazy loading)

    def load_data(self):
        if self.data is None:
            # data: N T (MVC)
            if self.use_mmap:
                npz_data = mmnpz.load(self.data_path, mmap_mode='r')
                print("\tLoaded data using mmap mode")
            else:
                npz_data = np.load(self.data_path)
                print("\tLoaded data into memory")

            self.data = npz_data[f"x_{self.split}"]
            print("\tSelf.data assigned")
            y = npz_data[f"y_{self.split}"]
            # Accept one-hot (N, num_classes) or integer (N,) labels
            self.labels = np.argmax(y, axis=-1) if y.ndim > 1 else np.asarray(y).astype(int)
            print("\tSelf.labels assigned")
            print(f"\tAssigned self.data to unique split ({self.split})")
            if self.debug:
                self.data = self.data[:100]
                self.labels = self.labels[:100]

            # Handle NaN filtering more memory efficiently when using mmap
            if self.use_mmap:
                # For memory-mapped arrays, avoid operations that load entire array into memory
                print(
                    "\tUsing memory-mapped data - skipping NaN filtering to preserve memory efficiency")
                # You may want to handle NaN values during __getitem__ instead
            else:
                nan_out = ~np.isnan(self.data.mean(-1).mean(-1))
                self.data = self.data[nan_out]
                self.labels = self.labels[nan_out]

            N, T, _ = self.data.shape
            # Channels per joint (3 = xyz / xy+conf, 53 = plus 50 flow features), inferred from
            # the stored width rather than a magic threshold.
            width = self.data.shape[-1]
            if width % (self.M * self.V_total) != 0:
                raise ValueError(
                    f"Feature width {width} is not divisible by num_persons*num_joints "
                    f"({self.M}*{self.V_total}); check num_joints/num_persons.")
            self.C = width // (self.M * self.V_total)
            if self.A is not None:
                self.data = self.data.reshape((N * T * self.M, self.V_total, self.C))
                self.data = np.array(self.A) @ self.data
            print("\tFinished loading data!")

            # Here is where we need to get the mean map (npz data loaded here)
            if self.normalisation:
                self.get_mean_map(npz_data)

    def get_n_per_class(self):
        self.n_per_cls = np.zeros(len(self.labels), dtype=int)
        for labels in self.labels:
            self.n_per_cls[labels] += 1
        self.csum_n_per_cls = np.insert(np.cumsum(self.n_per_cls), 0, 0)

    def _subset_joints_axis(self, arr):
        """Select self.joint_subset along the (unique) axis of size V_total, if there is one."""
        if self.joint_subset is None:
            return arr
        axes = [a for a, n in enumerate(arr.shape) if n == self.V_total]
        if len(axes) == 0:
            return arr                      # per-channel statistics only
        if len(axes) > 1:
            raise ValueError(f"Ambiguous joint axis in map of shape {arr.shape}")
        return np.take(arr, self.joint_subset, axis=axes[0])

    def get_mean_map(self, npz_data):
        if self.stats_path is not None:     # use the source dataset's statistics
            npz_data = np.load(self.stats_path)
        try:
            self.mean_map = self._subset_joints_axis(npz_data['mean_map'])
            self.std_map = self._subset_joints_axis(npz_data['std_map'])
        except KeyError as e:
            print("Feeder failed to obtain mean map")
            print(f"Data does not contain key: {e}")

    def __len__(self):
        self.load_data()
        return len(self.labels)

    def _reshape(self, data_numpy):
        T, _ = data_numpy.shape
        # data_numpy = np.array(data_numpy)
        data_numpy = data_numpy.reshape(T, self.M, self.V_total, self.C).transpose(
            3, 0, 2, 1
        )
        if self.joint_subset is not None:   # keep only the requested joints
            data_numpy = data_numpy[:, :, self.joint_subset, :]
        if self.no_flow:  # If no_flow argument is passed, only take x,y,z positions
            data_numpy = data_numpy[:3, ...]
        if self.no_Z:
            data_numpy = np.delete(data_numpy, 2, axis=0)
        return data_numpy  # C, T, V, M

    def __getitem__(self, index):
        self.load_data()
        data_numpy = self.data[index]
        data_numpy = self._reshape(data_numpy)
        label = self.labels[index]
        data_numpy = np.array(data_numpy)

        # Find the first non-zero frame and crop to it
        valid_frame = data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)
        valid_frame_num = np.sum(np.squeeze(valid_frame).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size
        )
        mask = abs(data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)) > 0


        # Apply optional transforms
        if self.normalisation:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_padding(data_numpy, self.window_size, self.pad_method)
        if self.random_move:
            data_numpy = tools.random_move(
                data_numpy, transform_candidate=[-0.1, -0.05, 0.0, 0.05, 0.1]
            )
        # if self.random_rot: TODO: Fix and test for poseoff data
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
        if self.skel_displacement:
            data_numpy = tools.skel_displacement(data_numpy)

        # Mask input data if observation ratio != 1.0
        if self.obs_ratio < 1.0:
            data_numpy = data_numpy[:, :int(self.window_size*self.obs_ratio), ...]
            data_numpy = tools.auto_padding(data_numpy, self.window_size, self.pad_method)

        return data_numpy, label, mask, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.labels)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == "__main__":
    import math
    import logging
    import argparse
    import os.path as osp
    from einops import rearrange
    from config.argclass import ArgClass
    from torch.utils.data import DataLoader

    # Create logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename='./logs/debug/feeders/ntu_feeder.log',
        encoding="utf-8",
        filemode="w",
        level=logging.DEBUG
    )

    # Argparser to test data moved to the slurm jobfs directory
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        dest="model_type",
        default="infogcn2",
        help="Base model type ['infogcn2', 'msg3d', 'stgcn2'] (default=infogcn2)"
    )
    parser.add_argument(
        "-d",
        dest="dataset",
        default="ntu",
        help="Config dataset, for this file it must be 'ntu' or 'ntu120' (default=ntu)",
    )
    parser.add_argument(
        "-f",
        dest="flow_embedding",
        default="base",
        help="Optical flow embedding method [base, cnn, avg, abs] (default=base)"
    )
    parser.add_argument(
        "-e",
        dest="evaluation",
        help="Evaluation benchmark used for specific dataset \
            (CV/CS for ntu60, CSub/CSert for ntu120)"
    )
    parser.add_argument(
        "-o",
        dest="obs_ratio",
        default="1.0",
        help="Observation ratio, used for training or evaluation."
    )
    parser.add_argument(
        "-j", "--joint_subset",
        dest="joint_subset", action="store_true",
        help="If passed, take a 17-point coco-style joint subset for the dataset"
    )
    parser.add_argument(
        "--debug", help="Use to debug, only take first 100 samples",
        action="store_true"
    )
    parser.add_argument(
        "--data_path_overwrite", help="Overwrite dataset path.\
        This overwrites `config['data_paths'][arg.evaluation]` completely (must be path to a file)."
    )
    parsed = parser.parse_args()

    assert parsed.model_type in ['infogcn2', 'msg3d', 'stgcn2']
    assert parsed.dataset in ['ntu', 'ntu120']
    assert parsed.evaluation in ['CV', 'CS', 'CSub', 'CSet']

    # Pass the argparse.Namespace object (parsed) to ArgClass to create an arg obj
    # `with open(f'./config/{arg.model_type}/{arg.config}/{arg.flow_embedding}.yaml', 'r')...`
    arg = ArgClass(arg=parsed)

    # Pass root path for the dataset objects
    if parsed.data_path_overwrite is not None:
        arg.feeder_args['use_mmap'] = True
        # Ensure the evaluation keys are strings... (mostly for ucf101...)
        arg.feeder_args['data_paths'] = dict(
            zip([str(key) for key in arg.feeder_args['data_paths'].keys()],
                list(arg.feeder_args['data_paths'].values()))
        )
        arg.feeder_args['data_paths'][parsed.evaluation] = parsed.data_path_overwrite

    if arg.joint_subset and arg.dataset in ["ntu", "ntu120"]:
        arg.feeder_args["joint_subset"] = NTU17_JOINTS if parsed.joint_subset else None

    # Parse the observation ratio
    if arg.obs_ratio != "1.0":
        arg.feeder_args['obs_ratio'] = float(arg.obs_ratio)

    logger.debug(f"Feeder testing for dataset: {arg.dataset}")
    logger.debug(f"\tFlow embedding: {arg.flow_embedding}")
    logger.debug(f"\tEvaluation: {arg.evaluation}")
    logger.debug(f"\tData path: {arg.feeder_args['data_paths'][parsed.evaluation]}")

    # Create the datasets and dataloaders objects
    train_dataset = Feeder(
        **arg.feeder_args,
        eval=arg.evaluation,
        split="train",
        debug=parsed.debug,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True
    )
    logger.debug(f"\tTrain feeder length: {len(train_dataset)}")
    test_dataset = Feeder(
        **arg.feeder_args,
        eval=arg.evaluation,
        split="test",
        debug=parsed.debug,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=arg.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True
    )
    logger.debug(f"\tTest feeder length: {len(test_dataset)}")

    # Calculate the number of iterations per epoch (used for cosine annealing)
    total_scheduler_iters = math.ceil(arg.num_epoch * ((len(train_dataset)) / arg.batch_size))
    logger.debug(f"Calculated number of iteractions per epoch (used for consine annealing): \
    {total_scheduler_iters}")

    # Log the overall shape of data in one of the feeders...
    logger.debug(f"Full data shape: {train_dataset.data.shape}")

    # This is to log the number of samples of each class (checking for class imbalance)
    debug_labels = {'train': [0 for i in range(120)], 'test': [0 for i in range(120)]}
    debug_means = {
        'train': {'total_count': 0},
        'test': {'total_count': 0}
    }

    # iterate over the train dataloader
    for iter_number, (data_numpy, label, mask, index) in enumerate(train_dataloader):
        # data_numpy shape: (B, C, T, V, M)
        for sample_label in label:
            debug_labels['train'][sample_label] += 1

    logger.debug("TRAIN DATA:")
    logger.debug(f"\tData shape: {data_numpy.shape}")  # (60, 3/53, 64, 25, 2)
    logger.debug(f"\tMask shape: {mask.shape}")
    logger.debug(f"\tLabel shape: {label.shape}")

    logger.debug(f"train labels: {debug_labels['train']}")


    # iterate over the test dataloader
    for iter_number, (data_numpy, label, mask, index) in enumerate(test_dataloader):
        # data_numpy shape: (B, C, T, V, M)
        for sample_label in label:
            debug_labels['test'][sample_label] += 1

    logger.debug("TEST DATA:")
    logger.debug(f"\tData shape: {data_numpy.shape}")  # (60, 3/53, 64, 25, 2)
    logger.debug(f"\tMask shape: {mask.shape}")
    logger.debug(f"\tLabel shape: {label.shape}")

    logger.debug(f"test labels: {debug_labels['test']}")
