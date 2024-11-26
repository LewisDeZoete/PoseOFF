import os
import sys

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, "../..")))

import torch
from torch.utils.data import Dataset

from lib.utils.transforms import LoadVideo


class CustomDataset(Dataset):
    """
    CustomDataset base class for unprocessed data loading.
    TODO: Implement a more robust way to apply transforms.
    """

    def __init__(self, arg):
        self.classes = arg.classes
        self.labels = arg.labels

        # Determine if the dataset is preprocessed
        self.preprocessed = arg.dataloader.get("preprocessed", False)
        self.ext = ".pt" if self.preprocessed else ".avi"
        self.data_path = arg.dataloader.get("data_path", "")

        # Get the device, defaulting to 'cpu'
        self.device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item_key = list(self.labels.keys())[idx]
        item_path = f"{self.data_path}{item_key}{self.ext}"
        label = self.labels[item_key]

        # If it's preprocessed, load the tensor
        if self.preprocessed: 
            output = torch.load(item_path, map_location=self.device)
        else: # Else, we'll need to load the video
            output = LoadVideo(video_path=item_path, max_frames=self.max_frames)
        
        # If transforms are defined, apply them to the array
        if hasattr(self, "transforms") and self.transforms:
            for transform in self.transforms: # Apply the transforms passed
                output = transform(output)

        return output, label


class SingleStreamDataset(CustomDataset):
    """
    Load a single stream dataset (either flow, pose or flowpose) by loading preprocessed tensors.
    """

    def __init__(self, arg, stream: str, transforms=None, max_frames=300):
        super().__init__(arg)
        self.transforms = transforms
        self.max_frames = max_frames

        # Ensure stream is either 'skel' or 'flow'
        if stream not in ["skel", "flow", "flowpose"]:
            raise ValueError("stream must be 'skel', 'flow' or 'flowpose'")

        self.data_path = arg.dataloader[f"{stream}_path"] if self.preprocessed else arg.dataloader['data_path']


class MultiStreamDataset(CustomDataset):
    def __init__(self, arg, transforms):
        super().__init__(arg)
        self.skel_path = arg.dataloader['skel_path']
        self.flow_path = arg.dataloader['flow_path']
        self.transforms = transforms
        # We have to pass the FlowPoseSampler arg!
        assert transforms is not None

    def __getitem__(self, idx):
        item_key = list(self.labels.keys())[idx]
        label = self.labels[item_key]
        # Preload the pre-processed poses and flows
        poses = torch.load(
            f"{self.skel_path}{item_key}{self.ext}", map_location=self.device
        )
        flows = torch.load(
            f"{self.flow_path}{item_key}{self.ext}", map_location=self.device
        )

        output = self.transforms(flows,poses)

        return output, label


if __name__ == "__main__":
    import time

    from lib.utils import ArgClass, FlowPoseSampler
    from lib.utils.augments import swap_numpy, flow_mag_norm, random_shift, random_move
    from torch.utils.data import DataLoader

    arg = ArgClass(arg="./config/custom_pose/train_joint.yaml")

    batch_size=1
    stream='flowpose'
    device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

    # Create the FlowPoseSampler transform object
    flowPoseTransform = FlowPoseSampler(device=device)
    # Create some transforms for debugging SingleStreamDataset
    input_transform = [swap_numpy(),
                       flow_mag_norm(),
                       random_shift(),
                       random_move(),
                       swap_numpy()]

    # Create the single- and multi-stream datasets
    singledataset = SingleStreamDataset(arg, stream=stream,transforms=input_transform)
    multidataset = MultiStreamDataset(arg, flowPoseTransform)
    
    # Create the dataloaders!
    singledataloader = DataLoader(singledataset, batch_size=batch_size, shuffle=True)
    multidataloader = DataLoader(multidataset, batch_size=batch_size, shuffle=False)

    # Test loading preprocessed data
    start = time.time()
    
    # Get a few batches
    for inputs, labels in singledataloader:
        print(f'Singlestream dataloader ({stream})')
        print(f'\tInput shape: {inputs.shape}')
        print(f'\tLabel shape: {labels.shape}')
        break
    # Get a few batches
    for inputs, labels in multidataloader:
        print(f'Multistream dataloader ({stream})')
        print(f'\tInput shape: {inputs.shape}')
        print(f'\tLabel shape: {labels.shape}')
        break

    print(f'Completed a batch of {batch_size} in {time.time()-start}seconds')