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

        # If transforms are defined, apply them to the video
        if hasattr(self, "transforms") and self.transforms:
            video = LoadVideo(video_path=item_path, max_frames=self.max_frames)
            output = self.transforms(video)
        else:
            output = torch.load(item_path, map_location=self.device)

        return output, label


class SingleStreamDataset(CustomDataset):
    """
    Load a single stream dataset (either flow or skeleton) by loading preprocessed tensors.
    PREPROCESSED DATA ONLY
    """

    def __init__(self, arg, stream: str, transforms=None, max_frames=300):
        super().__init__(arg)
        self.transforms = transforms
        self.max_frames = max_frames

        # Ensure stream is either 'skel' or 'flow'
        if stream not in ["skel", "flow"]:
            raise ValueError("stream must be 'skel' or 'flow'")

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
    import argparse
    import time

    from lib.utils import ArgClass, FlowPoseSampler, FlowPoseSampler_FAST
    # from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        dest="stream",
        default="skel",
        help="stream for SingleStreamDataset, either 'skel' or 'flow' (default skel)",
    )
    parsed = parser.parse_args()
    arg = ArgClass(arg="./config/custom_pose/train_joint.yaml")

    device = torch.device('cuda',arg.device) if torch.cuda.is_available() else "cpu"

    # Create the FlowPoseSampler transform object
    flowPoseTransform = FlowPoseSampler(device=device)
    flowPoseTransform_fast = FlowPoseSampler_FAST(device=device)

    # Create the datasets
    # single = SingleStreamDataset(arg, stream=parsed.stream)
    multi = MultiStreamDataset(arg,flowPoseTransform)


    # Test loading preprocessed data
    start = time.time()
    
    # Get the first 10 outputs from each dataset
    for i in range(20):
        # output_S, label = single[i]
        output_M, label = multi[i]

        # # Interrogate the information
        # print(f"Single stream {parsed.stream} output shape: {output_S.shape}")
        # print(f"Multi-stream output shape {output_M.shape}")

        # print(f"Single stream first row frame 10: {output_S[:,10,0,0]}")
        # print(f"Multi-stream first row frame 10: {output_M[:,10,0,0]}\n")

    print(f"Multi-stream first row frame 10: {output_M[:,10,0,0]}")
    print(f"Original in {time.time()-start} seconds")

     
    
    #### DELETE

    # Create dataset again using fast flowpose sampler
    multi = MultiStreamDataset(arg,flowPoseTransform_fast)

    # Test loading preprocessed data
    start = time.time()
    
    # Get the first 10 outputs from each dataset
    for i in range(20):
        # output_S, label = single[i]
        output_M, label = multi[i]

    print(f"Multi-stream first row frame 10: {output_M[:,10,0,0]}")
    print(f"Fast in {time.time()-start} seconds")