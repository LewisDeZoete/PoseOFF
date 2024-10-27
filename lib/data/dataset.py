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
        self.device = arg.device if torch.cuda.is_available() else "cpu"

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
    Load a single stream dataset (either flow or skeleton) by processing videos or loading preprocessed tensors.
    """

    def __init__(self, arg, stream: str, transforms=None, max_frames=300):
        super().__init__(arg)
        self.transforms = transforms
        self.max_frames = max_frames

        # Ensure stream is either 'skel' or 'flow'
        if stream not in ["skel", "flow"]:
            raise ValueError("stream must be 'skel' or 'flow'")

        self.data_path = arg.dataloader[f"{stream}_path"]

    def __getitem__(self, idx):
        item_key = list(self.labels.keys())[idx]
        item_path = f"{self.data_path}{item_key}{self.ext}"
        label = self.labels[item_key]

        # Load the preprocessed data directly
        output = torch.load(item_path, map_location=self.device)

        return output, label


class MultiStreamDataset(CustomDataset):
    def __init__(self, arg):
        super().__init__(arg)
        self.data_path = "./data/UCF-101/"

    def __getitem__(self, idx):
        item_key = list(self.labels.keys())[idx]
        flow = torch.load(
            f"{self.data_path}flow/{item_key}{self.ext}", map_location=self.device
        )
        pose = torch.load(
            f"{self.data_path}skeleton/{item_key}{self.ext}", map_location=self.device
        )

        return flow, pose


if __name__ == "__main__":
    import argparse
    import time

    from lib.utils import ArgClass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        dest="stream",
        default="skel",
        help="stream for SingleStreamDataset, either 'skel' or 'flow' (default skel)",
    )
    parsed = parser.parse_args()

    arg = ArgClass(arg="./config/custom_pose/train_joint.yaml")
    single = SingleStreamDataset(arg, stream=parsed.stream)
    multi = MultiStreamDataset(arg)

    # Test loading preprocessed data
    start = time.time()

    # output, label = single[0]
    # print(f"Single stream {parsed.stream} output shape: {output.shape}")
    flows, poses = multi[0]
    print(f"Multi-stream output shape - flow: {flows.shape}, pose: {poses.shape}")
    
    for i, flow in enumerate(flows):
        pass
    # for person_no in range(poses.shape[-1]):
    #     for pose_no in range(poses.shape[2]):
    #         if poses[-1,frame_no,pose_no,person_no] > 0.5:
    #             x,y,_ = poses[:,frame_no,pose_no,person_no]
    #             print(f'({int((x+1/2)*flow.shape[2])}, {int((y+1)/2*flow.shape[3])})')

            

    print(f"Loaded in {time.time()-start} seconds")
