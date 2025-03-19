import numpy as np
import os
import os.path as osp
import pickle
from einops import rearrange
import yaml

root_path = './data/ntu'
save_path = osp.join(root_path, 'aligned_data')
stat_path = osp.join(root_path, 'statistics')

with open(osp.join(stat_path, 'ntu_rgbd-available.txt'), 'r') as f:
    skes_names = f.read().splitlines()

with open(osp.join(stat_path, 'frames_cnt.txt'), 'r') as f:
    frames_cnt = f.read().splitlines()

file_paths = [osp.join(save_path, f'NTU60_{evaluation}-flowpose_aligned.npz') for evaluation in ['CS', 'CV']]

for file in file_paths:
    print(osp.join(save_path, f"MINI_{file.split('_')[2].split('-')[0]}_flowpose"))
    org_data = np.load(file)
    splits = ["x_train", "x_test"]
    small_set = {}
    for split in splits:
        data = org_data[split]
        small_set[split] = data[:120]
    
    np.savez(osp.join(save_path, f"MINI_{file.split('_')[2].split('-')[0]}_flowpose"),
             x_train=small_set["x_train"],
             y_train=org_data["y_train"][:120],
             x_test=small_set["x_test"],
             y_test=org_data["y_test"][:120])
        