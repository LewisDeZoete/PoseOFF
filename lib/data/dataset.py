import os
import sys

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, "../..")))

import torch
from torch.utils.data import Dataset
from data_gen.preprocess import LoadVideo


class CustomDataset(Dataset):
    """
    CustomDataset base class for unprocessed data loading.
    TODO: Implement a more robust way to apply transforms. SEE FEEDER_YOLO.PY
    """

    def __init__(self, arg):
        self.classes = arg.classes
        self.labels = arg.feeder_args['labels']

        # Determine if the dataset is preprocessed
        self.preprocessed = arg.extractor.get("preprocessed", False)
        self.ext = ".pt" if self.preprocessed else ".avi"
        self.data_paths = arg.feeder_args['data_paths']

        # Get the device, defaulting to 'cpu'
        self.device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item_key = list(self.labels.keys())[idx]
        item_path = f"{self.data_path}{item_key}{self.ext}"
        print(item_path)
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
    A dataset class for handling single stream data.

    Args:
        arg: Arguments required by the CustomDataset.
        stream (str): The type of stream data. Must be one of 'rgb', 'pose', 'flow', or 'flowpose'.
        ext (str, optional): The file extension for the data files. Defaults to '.pt'.
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version. Defaults to None.
        max_frames (int, optional): The maximum number of frames to consider. Defaults to 300.

    Raises:
        ValueError: If the stream is not one of 'rgb', 'pose', 'flow', or 'flowpose'.
    """
    def __init__(self, arg, stream: str, ext: str = '.pt', transforms=None, max_frames=300):
        super().__init__(arg)
        self.transforms = transforms
        self.max_frames = max_frames

        # Ensure stream is one of the data types
        if stream not in ["rgb", "pose", "flow", "flowpose"]:
            raise ValueError("stream must be 'rgb', 'pose', 'flow' or 'flowpose'")

        self.data_path = arg.feeder_args['data_paths'][f"{stream}_path"]
        self.ext = ext


class MultiStreamDataset(CustomDataset):
    def __init__(self, arg, transforms):
        super().__init__(arg)
        self.pose_path = arg.feeder_args['data_paths']['pose_path']
        self.flow_path = arg.feeder_args['data_paths']['flow_path']
        self.transforms = transforms
        # We have to pass the FlowPoseSampler arg!
        assert transforms is not None

    def __getitem__(self, idx):
        item_key = list(self.labels.keys())[idx]
        label = self.labels[item_key]
        # Preload the pre-processed poses and flows
        poses = torch.load(
            f"{self.pose_path}{item_key}{self.ext}", map_location=self.device
        )
        flows = torch.load(
            f"{self.flow_path}{item_key}{self.ext}", map_location=self.device
        )

        output = self.transforms(flows,poses)

        return output, label


if __name__ == "__main__":
    import time

    from config.argclass import ArgClass
    from data_gen.extractors import FlowPoseSampler
    from torch.utils.data import DataLoader
    import torchvision.transforms.v2 as v2

    arg = ArgClass(arg="./config/custom_pose/train_joint_infogcn.yaml")

    batch_size=1
    stream='pose'
    device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

    # Create some transforms for debugging SingleStreamDataset
    single_transforms = [
        v2.Resize(size=(384,640)), # YOLO pose has a minimum input image size
        v2.ToDtype(torch.float32),
        v2.Lambda(lambda x: x/255.0),]
    # Create the FlowPoseSampler transform object
    flowPoseTransform = FlowPoseSampler(device=device)

    # Create the single- and multi-stream datasets
    singledataset = SingleStreamDataset(arg, stream=stream,transforms=single_transforms)
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