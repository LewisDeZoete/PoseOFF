#!/usr/bin/env python3

import pickle
import numpy as np
import h5py
import time

root_path = "data/ntu/raw_data/raw_skes_data.pkl" # This contains dictionaries with more info...

loaded = pickle.load(open(root_path, 'rb'))

MAX_T = 300
N_SOURCES = 2
N_JOINTS = 25
N_DIMS = 2

def auto_pad(a,b):
    a = np.pad(a, [(0,max(a.shape[0], b.shape[0])-a.shape[0]), (0,0), (0,0)], 'constant')
    b = np.pad(b, [(0,max(a.shape[0], b.shape[0])-b.shape[0]), (0,0), (0,0)], 'constant')
    return a,b

def process_and_save(loaded, output_path):
    start = time.time()
    n = len(loaded)

    with h5py.File(output_path, "w") as f:
        # Pre-allocate full dataset upfront

        data_ds = f.create_dataset(
            "data",
            shape=(n, MAX_T, N_SOURCES, N_JOINTS, N_DIMS),
            dtype=np.float32,
            chunks=(1, MAX_T, N_SOURCES,N_JOINTS, N_DIMS),
            compression="lzf",
            fillvalue=0.0, # zero-pad by default
        )


        # Store names as variable-length strings
        name_ds = f.create_dataset(
            "names",
            shape=(n,),
            dtype=h5py.string_dtype(),
        )

        # Store the actual T for each sample so you can recover unpadded data later
        t_ds = f.create_dataset(
            "t_lengths",
            shape=(n,),
            dtype=np.int32,
        )

        for num, sample in enumerate(loaded):
            name = sample['name']

            # Concat the arrays if there
            performer_names = list(sample['data'])
            if len(performer_names) > 1:
                # ensure arrays to stack are the same length!
                multi_sample_list = list(auto_pad(
                    sample['data'][performer_names[0]]['colors'],
                    sample['data'][performer_names[1]]['colors']
                ))
                data = np.stack(multi_sample_list, axis=1)
            else:
                data = np.array(sample['data'][performer_names[0]]['colors'])[:, np.newaxis, ...]

            T = data.shape[0]

            padded = np.zeros((MAX_T, N_SOURCES, N_JOINTS, N_DIMS), dtype=np.float32)
            padded[:T, :data.shape[1], :, :] = data

            data_ds[num] = padded
            name_ds[num] = name
            t_ds[num] = T

            # Write the HDF5 file
            f.create_dataset(
                name=sample['name'],
                data=data,
            )

            # if num > 50:
            #     break

    print(f"Finished processing {output_path} in {time.time() - start:.2f} seconds")

process_and_save(loaded, output_path="./data/ntu/raw_data/raw_skes_data.h5")
