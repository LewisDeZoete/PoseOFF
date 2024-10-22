import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

import yaml
import torch
from torch.utils.data import Dataset
from lib.utils.transforms import LoadVideo

 
class CustomVideoDataset(Dataset):
    '''
    Custom video dataset
    TODO: add transforms and max_frames to the yaml file
    TODO: A more robust way to apply transforms
    '''
    def __init__(self, arg, transforms=None, max_frames=300):
        self.transforms = transforms
        self.max_frames = max_frames

        # Get the annotation file
        with open(arg.dataloader['label_path'], 'r') as file:
            self.ann_file = yaml.safe_load(file)
        # I don't actually think we use this...
        # self.phase = arg.phase

        # If the data has been preprocessed, use the skel path not data path
        if 'preprocessed' in arg.dataloader.keys() and arg.dataloader['preprocessed'] == True:
            self.preprocessed = True
            self.data_path = arg.dataloader['skel_path']
            self.ext = '.pt'
        else:
            self.preprocessed = False
            self.data_path = arg.dataloader['data_path']
            self.ext = '.avi'

    def __len__(self):
        return len(self.ann_file.keys())

    def __getitem__(self, idx):
        data_path = f'{self.data_path}{list(self.ann_file.keys())[idx]}{self.ext}'

        # Create the labels based on the videos we're grabbing
        label = list(self.ann_file.values())[idx]

        # Apply transforms
        if self.transforms:
            # get the video
            video = LoadVideo(video_path=data_path, 
                          max_frames=self.max_frames)
            output = self.transforms(video)
        else:
            # if no transforms are passed, assume we're just loading a tensor
            output = torch.load(data_path)

        return output, label




if __name__ == '__main__':
    from lib.utils import Dict2Class
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', metavar='config', default='custom_pose',
                        help='config dictionary location')
    parser.add_argument('-p', metavar='phase', default='test',
                        help='network phase [train, test]')
    parser.add_argument('-l', metavar='limb', default='joint',
                        help='limb (joint or bone)')
    parsed = parser.parse_args()


    arg = {'dataloader': {'label_path': '../Datasets/UCF-101/ucf101_annotations.yaml',
                                'data_path': '../Datasets/UCF-101/',
                                'skel_path': './data/UCF-101/',
                                'preprocessed': True},
                                
            'phase': 'test'}
    arg = Dict2Class(arg)
    dataset = CustomVideoDataset(arg)

    # Test loading preprocessed data
    start = time.time()
    
    print(f'Loaded in {time.time()-start} seconds')