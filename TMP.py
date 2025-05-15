from training.train_infogcn import load_checkpoint
# import torch
import os.path as osp
import os

embed = 'cnn'

ucf_root = './results/ucf101/1'
ntu_root = './results/nturgbd/CV'

ucf_path = osp.join(ucf_root, f'ucf101_1_{embed}.pt')
# ntu_path = osp.join(ntu_root, f'nturgbd_CV_{embed}_TMP.pt')

ucf_checkpoint = load_checkpoint(ucf_path, device='cpu')
# ntu_checkpoint = load_checkpoint(ntu_path, device='cpu')

def get_max_acc(dict):
    return max(dict, key=dict.get)
        

for epoch in range(ucf_checkpoint['epoch']):
    print(f'Epoch {epoch+1}')
    print('\t'+get_max_acc(ucf_checkpoint['results']['test_ACC'][epoch]))
    # print('\t'+get_max_acc(ntu_checkpoint['results']['test_ACC'][epoch]))
    print('\n')