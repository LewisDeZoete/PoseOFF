#!/usr/bin/env python3

from config.argclass import ArgClass
import time
from torch.utils.data import DataLoader
import logging
import argparse
import os.path as osp
from einops import rearrange

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
    "-d",
    dest="dataset",
    default="ntu",
    help="Config dictionary location (default=ucf101)",
)
parser.add_argument(
    "-f",
    dest="flow_embedding",
    default="base",
    help="Flow embedding, either base or cnn (default=base)"
)
parser.add_argument(
    "-e",
    dest="evaluation",
    help="Evaluation"
)
parser.add_argument(
    "--debug", help="Use to debug, only take first 100 samples",
    action="store_true"
)
parser.add_argument(
    "--data_path_overwrite", help="Overwrite dataset path.\
    This overwrites `config['data_paths'][arg.evaluation]` completely (must be a file)."
)
parsed = parser.parse_args()

assert parsed.evaluation in ['CV', 'CS', 'CSub', 'CSet']
assert parsed.flow_embedding in ['base', 'cnn']

# CHANGE THIS TO TEST DIFFERENT EMBEDDING CONFIGS
model_type = "msg3d"
dataset = parsed.dataset
evaluation = parsed.evaluation
flow_embedding = parsed.flow_embedding


arg = ArgClass(f"config/{model_type}/{dataset}/{flow_embedding}.yaml", verbose=True)
arg.feeder_args['eval'] = evaluation
arg.feeder_args['normalisation'] = False
arg.feeder_args['data_paths'][parsed.evaluation] = \
    arg.feeder_args['data_paths'][parsed.evaluation].replace("_mean", "")

# Pass root path for the dataset objects
if parsed.data_path_overwrite is not None:
    arg.feeder_args['use_mmap'] = True
    arg.feeder_args['data_paths'][parsed.evaluation] = parsed.data_path_overwrite

logger.debug(f"Feeder testing for dataset: {dataset}")
logger.debug(f"\tFlow embedding: {flow_embedding}")
logger.debug(f"\tEvaluation: {evaluation}")
logger.debug(f"\tData path: {arg.feeder_args['data_paths'][evaluation]}")

# Create the dataset objects
train_feeder = Feeder(
    **arg.feeder_args,
    split="train",
    debug=parsed.debug
)
logger.debug(f"\tTrain feeder length: {len(train_feeder)}")
test_feeder = Feeder(
    **arg.feeder_args,
    split="test",
    debug=parsed.debug)
logger.debug(f"\tTest feeder length: {len(test_feeder)}")

# Calculate the number of iterations per epoch (used for cosine annealing)
cal_epoch_iters = math.ceil(len(train_feeder) / arg.batch_size)
logger.debug(f"Calculated epoch iters: {cal_epoch_iters}")

# Log the overall shape of data in one of the feeders...
logger.debug(f"Full data shape: {train_feeder.data.shape}")

# Create a dataloaders
train_dataloader = DataLoader(train_feeder,
                        batch_size=arg.batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True)
test_dataloader = DataLoader(test_feeder,
                        batch_size=arg.batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True)

# This is to log the number of samples of each class (checking for class imbalance)
debug_labels = {'train': [0 for i in range(120)], 'test': [0 for i in range(120)]}
debug_means = {
    'train': {'total_count': 0},
    'test': {'total_count': 0}
}

C = 3 if flow_embedding=='base' else 53
V = 25

sum_map = np.zeros((C, V), dtype=np.float64)
total = 0

# iterate over the train dataloader
for iter_number, (data_numpy, label, mask, index) in enumerate(train_dataloader):
    # data_numpy shape: (B, C, T, V, M)
    for sample_label in label:
        debug_labels['train'][sample_label] += 1
    batch_sum = data_numpy.sum(axis=2, keepdims=False).sum(axis=3, keepdims=False)
    sum_map += batch_sum.sum(axis=0).numpy()
    total += data_numpy.shape[0] * data_numpy.shape[1] * data_numpy.shape[2]

# iterate over the test dataloader
for iter_number, (data_numpy, label, mask, index) in enumerate(test_dataloader):
    # data_numpy shape: (B, C, T, V, M)
    for sample_label in label:
        debug_labels['test'][sample_label] += 1
    batch_sum = data_numpy.sum(axis=2, keepdims=False).sum(axis=3, keepdims=False)
    sum_map += batch_sum.sum(axis=0).numpy()
    total += data_numpy.shape[0] * data_numpy.shape[1] * data_numpy.shape[2]

mean_map = (sum_map / total)


# After mean map calculations, we can calculate standard deviation map
sq_diff_sum = np.zeros((C * V), dtype=np.float64)

# iterate over the train dataloader again for standard deviation map
for iter_number, (data_numpy, label, mask, index) in enumerate(train_dataloader):
    # data_numpy shape: (B, C, T, V, M)
    B, C, T, V, M = data_numpy.shape
    data_numpy_t = rearrange(data_numpy, 'B C T V M -> (B T M) (C V)')
    mean_flat = rearrange(mean_map, 'C V -> 1 (C V)')
    sq_diff = (data_numpy_t - mean_flat) ** 2
    sq_diff_sum += sq_diff.sum(axis=0).numpy()

# iterate over the test dataloader again for standard deviation map
for iter_number, (data_numpy, label, mask, index) in enumerate(test_dataloader):
    # data_numpy shape: (B, C, T, V, M)
    B, C, T, V, M = data_numpy.shape
    data_numpy_t = rearrange(data_numpy, 'B C T V M -> (B T M) (C V)')
    mean_flat = rearrange(mean_map, 'C V -> 1 (C V)')
    sq_diff = (data_numpy_t - mean_flat) ** 2
    sq_diff_sum += sq_diff.sum(axis=0).numpy()

std_flat = np.sqrt(sq_diff_sum / total)
std_map = std_flat.reshape(C, 1, V, 1)

logger.debug("TRAIN DATA:")
logger.debug(f"\tData shape: {data_numpy.shape}")  # (60, 3/53, 64, 25, 2)
logger.debug(f"\tMask shape: {mask.shape}")
logger.debug(f"\tLabel shape: {label.shape}")

logger.debug("TEST DATA:")
logger.debug(f"\tData shape: {data_numpy.shape}")  # (60, 3/53, 64, 25, 2)
logger.debug(f"\tMask shape: {mask.shape}")
logger.debug(f"\tLabel shape: {label.shape}")

logger.debug(f"mean map: {mean_map}")
logger.debug(f"standard deviation map: {std_map}")
logger.debug(f"train labels: {debug_labels['train']}")
logger.debug(f"test labels: {debug_labels['test']}")


if not parsed.debug:
    npz_data = mmnpz.load(arg.feeder_args['data_paths'][parsed.evaluation], mmap_mode='r')
    filename = arg.feeder_args['data_paths'][parsed.evaluation].split('/')[-1]
    with mmnpz.NpzWriter(
            f"./data/{dataset}/aligned_data/{filename.replace('aligned', 'aligned_mean')}") as f:
        f.write("x_train", npz_data['x_train'])
        f.write("x_test", npz_data['x_test'])
        f.write("y_train", npz_data['y_train'])
        f.write("y_test", npz_data['y_test'])
        f.write("mean_map", mean_map)
        f.write("std_map", std_map)

    print(f"Saved data to: ./data/{dataset}/aligned_data/{filename.replace('aligned', 'aligned_mean')}")
