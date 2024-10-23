import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '../..')))

import yaml
import torch
from torch.utils.data import Dataset
from lib.utils.transforms import LoadVideo


class CustomDataset(Dataset):
    '''
    CustomDataset base class. Useful for inheritence of for unprocessed dataloading.
    TODO: A more robust way to apply transforms
    '''
    def __init__(self, arg):
        self.classes = arg.classes
        self.labels = arg.labels

        # If the data has been preprocessed, use the skel path not data path
        if 'preprocessed' in arg.dataloader.keys() and arg.dataloader['preprocessed'] == True:
            self.preprocessed = True
            # self.datapath must be set manually
            self.ext = '.pt'
        else:
            self.preprocessed = False
            self.data_path = arg.dataloader['data_path']
            self.ext = '.avi'
        
        # Get the device in arg (otherwise default to 'cpu')
        self.device = arg.device if torch.cuda.is_available() else 'cpu'
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Assumes unprocessed!
        item_path = f'{self.data_path}{list(self.labels.keys())[idx]}{self.ext}'

        # Create the labels based on the videos we're grabbing
        label = list(self.labels.values())[idx]

        # If there are transforms
        if self.transforms:
            # get the video
            video = LoadVideo(video_path=item_path, 
                          max_frames=self.max_frames)
            output = self.transforms(video)
        return output, label
    

class SingleStreamDataset(CustomDataset):
    '''
    Load a single stream dataset (either flow or skeleton) by processing videos or by loading preprocessed tensors.
    '''
    def __init__(self, arg, stream:str, transforms=None, max_frames=300):
        super().__init__(arg)
        self.transforms = transforms
        self.max_frames = max_frames

        # Ensure stream is either skel or flow
        if stream not in ['skel', 'flow']:
            raise ValueError("For SingleStreamDataset, stream input must be 'skel' or 'flow'")
        
        self.data_path = arg.dataloader[f'{stream}_path']

    def __getitem__(self, idx):
        item_path = f'{self.data_path}{list(self.labels.keys())[idx]}{self.ext}'

        # Create the labels based on the videos we're grabbing
        label = list(self.labels.values())[idx]

        # SingleStreamDataset assumes preprocessed data, simply load it!
        output = torch.load(item_path, map_location=self.device)

        return output, label


class MultiStreamDataset(CustomDataset):
    def __init__(self, arg):
        super().__init__(arg)
        self.data_path = './data/UCF-101/'
    
    def __getitem__(self, idx):
        flow = torch.load(f'{self.data_path}flow/{list(self.labels.keys())[idx]}{self.ext}',
                          map_location=self.device)
        pose = torch.load(f'{self.data_path}skeleton/{list(self.labels.keys())[idx]}{self.ext}',
                          map_location=self.device)
        return flow, pose

        

if __name__ == '__main__':
    from lib.utils import ArgClass
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='stream', default='skel',
                        help="stream for SingleStreamDataset, either 'skel' or 'flow' (default skel)")
    parsed = parser.parse_args()

    arg = ArgClass(arg='./config/custom_pose/train_joint.yaml')
    single = SingleStreamDataset(arg, stream=parsed.stream)
    multi = MultiStreamDataset(arg)

    # Test loading preprocessed data
    start = time.time()

    output, label = single[0]
    print(f'Single stream {parsed.stream} output shape: {output.shape}')
    flow, pose = multi[0]
    print(f'Multi-stream output shape: {flow.shape}, {pose.shape}')
        
    print(f'Loaded in {time.time()-start} seconds')