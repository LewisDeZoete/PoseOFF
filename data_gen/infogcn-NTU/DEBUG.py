import numpy as np
import os
import os.path as osp
import pickle

root_path = './data/ntu'
save_path = osp.join(root_path, 'aligned_data')
stat_path = osp.join(root_path, 'statistics')

with open(osp.join(stat_path, 'ntu_rgbd-available.txt'), 'r') as f:
    skes_names = f.read().splitlines()

with open(osp.join(stat_path, 'frames_cnt.txt'), 'r') as f:
    frames_cnt = f.read().splitlines()

with open(osp.join(save_path, 'NTU60_CD_aligned.npz'), 'rb') as f:
    aligned_data = np.load(f)

print(len(aligned_data['x_train']), len(aligned_data['x_test']))
print(aligned_data['x_train'].shape, aligned_data['x_test'].shape)