#!/usr/bin/env python3

# from config.argclass import ArgClass
# import time
# import math
# from torch.utils.data import DataLoader
# import logging
# import argparse
# import os.path as osp
# from einops import rearrange
# import numpy as np
# import mmnpz


# # Argparser to test data moved to the slurm jobfs directory
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-d",
#     dest="dataset",
#     default="ntu",
#     help="Config dictionary location (default=ntu)",
# )
# parser.add_argument(
#     "-f",
#     dest="flow_embedding",
#     default="base",
#     help="Flow embedding, either base or cnn (default=base)"
# )
# parser.add_argument(
#     "--data_path_overwrite", help="Overwrite dataset path.\
#     This overwrites `config['data_paths'][arg.evaluation]` completely (must be a file)."
# )
# parser.add_argument(
#     "--save_path_overwrite", help="Overwrite the save file path.\
#     This is just the path to the file that was copied, so we can save a '_mean.npz' version!"
# )
# parser.add_argument(
#     "--debug", help="Use to debug, only take first 100 samples",
#     action="store_true"
# )
# parsed = parser.parse_args()


# assert parsed.flow_embedding in ['base', 'cnn']

# # Set the variables
# model_type = "msg3d"
# dataset = parsed.dataset
# flow_embedding = parsed.flow_embedding

# # Create logger
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename=f"./logs/debug/data_gen/get_mean_map_{dataset}.log",
#     encoding="utf-8",
#     filemode="w",
#     level=logging.DEBUG
# )

# arg = ArgClass(f"config/{model_type}/{dataset}/{flow_embedding}.yaml", verbose=True)
# arg.feeder_args['eval'] = evaluation
# arg.feeder_args['normalisation'] = False
# arg.feeder_args['data_paths'][parsed.evaluation] = \
#     arg.feeder_args['data_paths'][parsed.evaluation].replace("_mean", "")

# # Pass root path for the dataset objects
# if parsed.data_path_overwrite is not None:
#     arg.feeder_args['use_mmap'] = True
#     arg.feeder_args['data_paths'][parsed.evaluation] = parsed.data_path_overwrite

# logger.debug(f"Datagen testing for dataset: {dataset}")
# logger.debug(f"\tEvaluation: {evaluation}")
# logger.debug(f"\tFlow embedding: {flow_embedding}")
# logger.debug(f"\tData path: {arg.feeder_args['data_paths'][evaluation]}")
# logger.debug(f"\tSave path: {parsed.save_path_overwrite}")

# # Create the dataset objects
# FeederClass = arg.import_class(arg.feeder)
# train_feeder = FeederClass(
#     **arg.feeder_args,
#     split="train",
#     debug=parsed.debug
# )
# logger.debug(f"\tTrain feeder length: {len(train_feeder)}")
# test_feeder = FeederClass(
#     **arg.feeder_args,
#     split="test",
#     debug=parsed.debug)
# logger.debug(f"\tTest feeder length: {len(test_feeder)}")

# # Calculate the number of iterations per epoch (used for cosine annealing)
# cal_epoch_iters = math.ceil(len(train_feeder) / arg.batch_size)
# logger.debug(f"Calculated epoch iters: {cal_epoch_iters}")

# # Log the overall shape of data in one of the feeders...
# logger.debug(f"Full data shape: {train_feeder.data.shape}")

# # Create a dataloaders
# train_dataloader = DataLoader(train_feeder,
#                         batch_size=arg.batch_size,
#                         num_workers=4,
#                         shuffle=False,
#                         pin_memory=True)
# test_dataloader = DataLoader(test_feeder,
#                         batch_size=arg.batch_size,
#                         num_workers=4,
#                         shuffle=False,
#                         pin_memory=True)


# C = 3 if flow_embedding=='base' else 53
# V = 25

# sum_map = np.zeros((C, V), dtype=np.float64)
# total = 0




# # iterate over the train dataloader
# for iter_number, (data_numpy, label, mask, index) in enumerate(train_dataloader):
#     # data_numpy shape: (B, C, T, V, M)
#     batch_sum = data_numpy.sum(axis=2, keepdims=False).sum(axis=3, keepdims=False)
#     sum_map += batch_sum.sum(axis=0).numpy()
#     total += data_numpy.shape[0] * data_numpy.shape[1] * data_numpy.shape[2]

# # iterate over the test dataloader
# for iter_number, (data_numpy, label, mask, index) in enumerate(test_dataloader):
#     # data_numpy shape: (B, C, T, V, M)
#     batch_sum = data_numpy.sum(axis=2, keepdims=False).sum(axis=3, keepdims=False)
#     sum_map += batch_sum.sum(axis=0).numpy()
#     total += data_numpy.shape[0] * data_numpy.shape[1] * data_numpy.shape[2]

# mean_map = (sum_map / total)
# mean_map = rearrange(mean_map, 'C V -> C 1 V 1')


# # After mean map calculations, we can calculate standard deviation map
# sq_diff_sum = np.zeros((C * V), dtype=np.float64)

# # iterate over the train dataloader again for standard deviation map
# for iter_number, (data_numpy, label, mask, index) in enumerate(train_dataloader):
#     # data_numpy shape: (B, C, T, V, M)
#     B, C, T, V, M = data_numpy.shape
#     data_numpy_t = rearrange(data_numpy, 'B C T V M -> (B T M) (C V)')
#     mean_flat = rearrange(mean_map, 'C 1 V 1 -> 1 (C V)')
#     sq_diff = (data_numpy_t - mean_flat) ** 2
#     sq_diff_sum += sq_diff.sum(axis=0).numpy()

# # iterate over the test dataloader again for standard deviation map
# for iter_number, (data_numpy, label, mask, index) in enumerate(test_dataloader):
#     # data_numpy shape: (B, C, T, V, M)
#     B, C, T, V, M = data_numpy.shape
#     data_numpy_t = rearrange(data_numpy, 'B C T V M -> (B T M) (C V)')
#     mean_flat = rearrange(mean_map, 'C 1 V 1 -> 1 (C V)')
#     sq_diff = (data_numpy_t - mean_flat) ** 2
#     sq_diff_sum += sq_diff.sum(axis=0).numpy()

# std_flat = np.sqrt(sq_diff_sum / total)
# std_map = std_flat.reshape(C, 1, V, 1)

# logger.debug("TRAIN DATA:")
# logger.debug(f"\tData shape: {data_numpy.shape}")  # (60, 3/53, 64, 25, 2)
# logger.debug(f"\tMask shape: {mask.shape}")
# logger.debug(f"\tLabel shape: {label.shape}")

# logger.debug("TEST DATA:")
# logger.debug(f"\tData shape: {data_numpy.shape}")  # (60, 3/53, 64, 25, 2)
# logger.debug(f"\tMask shape: {mask.shape}")
# logger.debug(f"\tLabel shape: {label.shape}")

# logger.debug(f"mean map shape: {mean_map.shape}") # (3/53, 1, 25, 1)
# logger.debug(f"mean map: {mean_map}")
# logger.debug(f"standard deviation map shape: {std_map.shape}") # (3/53, 1, 25, 1)
# logger.debug(f"standard deviation map: {std_map}")


# if not parsed.debug:
#     # Load the npz data from the ssd (this is overwritten by --data_path_overwrite)
#     npz_data = mmnpz.load(arg.feeder_args['data_paths'][parsed.evaluation], mmap_mode='r')
#     filepath = osp.join(*parsed.save_path_overwrite.split('/')[:-1])
#     filename = parsed.save_path_overwrite.split('/')[-1]
#     with mmnpz.NpzWriter(
#             osp.join(filepath, filename.replace("aligned", "aligned_mean"))
#     ) as f:
#         f.write("x_train", npz_data['x_train'])
#         f.write("x_test", npz_data['x_test'])
#         f.write("y_train", npz_data['y_train'])
#         f.write("y_test", npz_data['y_test'])
#         f.write("mean_map", mean_map)
#         f.write("std_map", std_map)

#     print(f"Saved data to: {osp.join(filepath, filename.replace('aligned', 'aligned_mean'))}")

import time
import os.path as osp
import numpy as np
import mmnpz

def get_mean_map(
        file_list=["data/nt/aligned_data/pose/ntu_CS-pose_aligned.npz", "data/ntu/aligned_data/pose/ntu_CV-pose_aligned.npz"],
        batch_size=64
):
    for file in file_list:
        if not osp.exists(file):
            print(f"File: \t{file}\nDoes not exist. Regenerate the aligned dataset and retry.")
        start_time = time.time()
        org_data = mmnpz.load(file)
        splits = ["x_train", "x_test"]

        count = None
        mean = None # shape (V, C)
        M2 = None # shape (V, C)

        # Original splits are arrays of shape (N, T, M*V*C)
        M, V = 2, 25
        C = 53 if "poseoff" in file else 3

        # Process the splits :)
        for split in splits:
            data = org_data[split]
            N, T, _ = data.shape
            # Batch the data
            for start in range(0, N, batch_size):
                # Load and reshape batch
                batch = data[start:start+batch_size].astype(np.float64)
                batch = batch.reshape(batch.shape[0], T, M, V, C)
                # batch shape: (B, T, M, V, C)

                # A frame is "real" if any of (M, V, C) is non-zero
                valid_mask = (batch != 0).any(axis=(3,4)) # (B, T, M) - True if a frame is real!
                batch_count = valid_mask.sum()

                if batch_count == 0:
                    continue

                # Extract mean and var only for valid frames
                valid_frames = batch[valid_mask]
                batch_mean = valid_frames.mean(axis=0) # V, C
                batch_var = valid_frames.var(axis=0) # V, C

                if count is None:
                    count = batch_count
                    mean = batch_mean
                    M2 = batch_var * batch_count
                else:
                    new_count = count + batch_count
                    delta = batch_mean - mean
                    mean = mean + delta * (batch_count / new_count)
                    M2 = M2 + batch_var * batch_count + (delta ** 2) * (count*batch_count/new_count)
                    count = new_count

        # Calculate standard deviation after mean calculation
        variance = M2 / count
        std = np.sqrt(variance)
        # Reshape (V, C) -> (C, 1, V, 1)
        mean_map = mean.T[:, np.newaxis, :, np.newaxis]
        std_map = std.T[:, np.newaxis, :, np.newaxis]
        print(f"\t\tMean and standard deviation maps took {time.time()-start_time: 0.2f} seconds", flush=True)

        # Save the array again with new keys "mean_map" and "std_map"
        start_time = time.time()
        with mmnpz.NpzWriter(file.replace("aligned", "aligned_mean")) as f:
            f.write("x_train", org_data['x_train'])
            f.write("x_test", org_data['x_test'])
            f.write("y_train", org_data['y_train'])
            f.write("y_test", org_data['y_test'])
            f.write("mean_map", mean_map)
            f.write("std_map", std_map)
        print(f"\t\tSaved data to: {file.replace('aligned', 'aligned_mean')}", flush=True)
        print(f"\t\tSaving took {time.time()-start_time:0.2f} seconds", flush=True)

if __name__=="__main__":
    get_mean_map()
