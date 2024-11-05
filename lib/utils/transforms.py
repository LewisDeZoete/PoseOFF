import numpy as np
import torch
import decord
from decord import VideoReader, cpu
import re
from flow import stack_frames


def LoadVideo(video_path, max_frames) -> torch.tensor:
    """
    Args:
        `max_frames` - maximum number of output frames

    Returns:
        `result` - on `__call__`, when passing in a `result` dict with the
            key ['video_path'], it will get the frames of that video.
            `output.shape == (nframes (max_frames), height, width, color)`
    """
    decord.bridge.set_bridge("torch")
    vr = VideoReader(video_path, ctx=cpu(0))
    # if there's too many frames, get `max_frames` linearly spaced frames
    if len(vr) > max_frames:
        # output = torch.tensor(vr.get_batch(np.linspace(0, len(vr)-1, self.max_frames)).asnumpy())
        output = vr.get_batch(np.linspace(0, len(vr) - 1, max_frames))
    else:
        # output = torch.tensor(vr.get_batch(np.linspace(0, len(vr)-1, len(vr))).asnumpy())
        output = vr.get_batch(np.linspace(0, len(vr) - 1, len(vr)))

    # RGB Colour format
    output = torch.permute(output, (0, 3, 1, 2))

    return output


class GetPoses_YOLO:
    """
    Creates a numpy array of shape:
        (channels, max_frames, num_joints, max_number_people)
        (3,        300,        17,         2)
    Only parses video frames up to max_frames, the rest are skipped.
    """

    def __init__(
        self,
        detector,
        max_frames: int = 300,
        num_joints: int = 17,
        num_people_out: int = 2,
        resdict=False,
    ):
        self.detector = detector
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.num_people_out = num_people_out
        self.resdict = resdict

    def __call__(self, video) -> torch.tensor:
        # If input is a dicitonary, we want the video to work with
        if isinstance(video, dict):
            video = video["rgb"]
        # If not, it could be we just want the poses
        else:
            # OR we may want to output a dict but we need to create it
            if self.resdict:
                results = {"rgb": video}

        data_torch = torch.zeros(
            (
                3,  # channels (x,y,confidence)
                self.max_frames,  # total_frames
                self.num_joints,  # number of joints
                self.num_people_out,
            )
        )  # max number of people output

        # Get pose results
        pose_results = self.detector(video, verbose=False)

        # Get data from yolo
        for frame in pose_results:
            # Get the frame number that yolo outputs in the frame.path variable
            frame_index = int(re.findall(r"\d+", frame.path)[0])
            for m, person in enumerate(frame.keypoints):
                # if there are more than num_people_out people, skip the rest
                if m >= self.num_people_out:
                    break
                # if no person is detected, it still returns a results dict with shape
                # [1, 0, 2]
                # we check if there are 17 joints for the person
                try:
                    assert person.xyn.shape[1] == self.num_joints
                except AssertionError:
                    continue
                # each landmark has .x, .y and .visibility
                data_torch[0, frame_index, :, m] = person.xyn[0, :, 0]
                data_torch[1, frame_index, :, m] = person.xyn[0, :, 1]
                data_torch[2, frame_index, :, m] = person.conf[0]

        # Centralisation (about zero [-0.5 : 0.5])
        data_torch[0:2] = data_torch[0:2] - 0.5
        data_torch[1:2] = -data_torch[1:2]
        data_torch[0][data_torch[2] == 0] = 0
        data_torch[1][data_torch[2] == 0] = 0

        # sort by score
        sort_index = (-data_torch[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            # Took out a `.transpose(1,2,0)` on the second tensor... did it break?
            data_torch[:, t, :, :] = data_torch[:, t, :, s]
        data_torch = data_torch[:, :, :, 0:2].type(torch.float32)

        # If we're expecting to output a dictionary, add the new pose key
        if self.resdict:
            results["pose"] = data_torch
            return results
        # Else simply return the poses
        else:
            return data_torch


class GetFlow:
    """Calculates flow for each image pair in a video using RAFT.

    Returns a tensor with (N, 2, H, W) where each entry corresponds to the
    horizontal and vertical displacement of each pixel from the first image
    to the second image.
    Note that the predicted flows are in “pixel” unit, they are not normalized
    w.r.t. the dimensions of the images.

    Attributes:
        model (torch.tensor): RAFT flow model.
        device (torch.device): Compute device.
        resdict (bool): Whether or not input for call method is a results `dict`
        minibatch_size (int): Number of batches to break up video processing into.
    """

    def __init__(self, model, device, resdict: bool = False, minibatch_size: int = 8):
        self.model = model
        self.device = device
        self.resdict = resdict
        self.minibatch_size = minibatch_size

        # Move the model to the corresponding device
        self.model.to(self.device)

    def __call__(self, video) -> torch.tensor:
        # If input is a dicitonary, we want the video to work with
        if isinstance(video, dict):
            video = video["rgb"]
        # If not, it could be we just want the poses
        else:
            # OR we may want to output a dict but we need to create it
            if self.resdict:
                results = {"rgb": video}

        # Stack the frames in a tensor that looks like: [[0, 1],
        stacked = stack_frames(video)  #                  [1, 2]] etc.
        # NOTE: this returns the video video stacked, we're batching it

        # Create a flow list, then calculate flow with raft (no_grad)
        flow = []
        with torch.no_grad():
            # process each video in batches (faced OOM issues)
            for i in range(0, stacked.shape[0], self.minibatch_size):
                minibatch = stacked[i : i + self.minibatch_size].to(self.device)

                # Calcualte the flow (returns a list of length 12, last element
                # is the the last pass of the model and most accurate flow
                flow_list = self.model(minibatch[:, 0, ...], minibatch[:, 1, ...])
                flow.append(flow_list[-1])

        # Concatenate the list elements back into one array!
        flow = torch.cat(flow, axis=0)
        # If we're expecting a dictionary output, add the new key
        if self.resdict:
            results["flow"] = flow
            return results
        # Otherwise simply return the flow
        else:
            return flow


class FlowPoseSampler:
    """Sample optical flow using input poses.
    NOTE: Assumes pre-processed optical flow and poses (for now).
    NOTE: Utilises the MultiStreamDataset, it passes in the loaded arrays.
    """

    def __init__(
        self,
        device: torch.device,
        kernel_size: int = 3,
        threshold: float = 0.5,
    ):
        self.device = device # Making sure all tensors are on the same device
        self.kernel_size = kernel_size  # Kernel size about pose keypoint
        self.half_k = self.kernel_size // 2  # Half the kernel size
        self.threshold = threshold

    def __call__(self, flows, poses):
        """Samples the optical flow in windows surrounding the pose keypoints.
        Returns array of shape (5,frames,keypoints,num_people) eg. (5,300,17,2)"""
        # Used to stack the sampled flows to the output array
        stacker = torch.zeros((2, poses.shape[1], poses.shape[-1] * poses.shape[-2])).to(self.device)
        poses = poses.to(self.device) # these are the only two being concatenated, so only need to move this one

        # Get details of the video before processing
        num_frames, _, height, width = flows.shape

        # Iterate over the flow frames (since pose array has more empty frames)
        for i, flow in enumerate(flows):
            # Reshape the pose points to be a continuous array
            pose_points = poses[:2,i].clone().reshape(2,poses.shape[-1]*poses.shape[-2])
            # Get visibility tensor to use as a mask
            vis = poses[2,i].clone().reshape(34)
            vis = vis > self.threshold
            # Rescale values from (-0.5:0.5) to (0,height-1) since we're using it to index
            pose_points[0] = (pose_points[0]+0.5)*(width-1)
            pose_points[1] = (pose_points[1]+0.5)*(height-1)
            pose_points = pose_points.type(torch.int)

            # Get the keypoints and check they're within the bounds of the flow image
            for keypoint_num in range(pose_points.shape[1]):
                # x,y = poses[0,keypoint_num], poses[1,keypoint_num]
                x,y = pose_points[0,keypoint_num], pose_points[1,keypoint_num]
                
                # Since we're using the vis mask, this captures low visibility points
                if y < self.half_k or y >= height - self.half_k or x < self.half_k or x >= width - self.half_k:
                    continue

                # Get the window of optical flow about the input point 
                # (of shape (2,kernel,kernel))
                flow_window = flow[:,y-self.half_k : y+self.half_k+1, x-self.half_k : x+self.half_k+1]
                av_flow_x = flow_window[0].mean().item() # Get the average x and y flow
                av_flow_y = flow_window[1].mean().item() # return as a single number
                
                # Add them to our temporary results array
                # 'i' is the frame number (+1 since we need two frames for flow)
                stacker[0,i+1,keypoint_num] = av_flow_x
                stacker[1,i+1,keypoint_num] = av_flow_y
        
        # Finally, concatenate the original pose array with the new stacked flows
        flow_pose = torch.cat((poses,stacker.reshape(2,poses.shape[1],poses.shape[2],poses.shape[3])),0)
        return flow_pose


class FlowPoseSampler_FAST:
    """Sample optical flow using input poses.
    NOTE: Assumes pre-processed optical flow and poses (for now).
    NOTE: Utilises the MultiStreamDataset, it passes in the loaded arrays.
    """

    def __init__(self, device: torch.device, kernel_size: int = 3, threshold: float = 0.5):
        self.device = device  # Making sure all tensors are on the same device
        self.kernel_size = kernel_size  # Kernel size about pose keypoint
        self.half_k = self.kernel_size // 2  # Half the kernel size
        self.threshold = threshold

    def __call__(self, flows, poses):
        """Samples the optical flow in windows surrounding the pose keypoints.
        Returns array of shape (5, frames, keypoints, num_people) e.g., (5, 300, 17, 2)"""
        stacker = torch.zeros((2, poses.shape[1], poses.shape[-1] * poses.shape[-2]), device=self.device)

        # Convert poses once to the correct device and scale for indexing
        poses = poses.to(self.device)
        num_frames, _, height, width = flows.shape

        # Iterate over the flow frames
        for i, flow in enumerate(flows):
            # Reshape and rescale pose points in a single operation
            pose_points = ((poses[:2, i] + 0.5) * torch.tensor([width - 1, height - 1], device=self.device).view(-1, 1, 1)).type(torch.int)
            vis = poses[2, i].flatten() > self.threshold  # Visibility mask
            
            # Loop through keypoints, filtering out low-visibility points
            for keypoint_num in range(pose_points.shape[1]):
                x, y = pose_points[0, keypoint_num], pose_points[1, keypoint_num]
                # TODO: ensure that pose points array is of shape (2,34) !!
                
                # Ignore out-of-bounds keypoints
                if vis[keypoint_num] and self.half_k <= y < height - self.half_k and self.half_k <= x < width - self.half_k:
                    # Get the window of optical flow and calculate mean directly
                    flow_window = flow[:, y - self.half_k : y + self.half_k + 1, x - self.half_k : x + self.half_k + 1]
                    stacker[0, i + 1, keypoint_num] = flow_window[0].mean()
                    stacker[1, i + 1, keypoint_num] = flow_window[1].mean()

        # Concatenate poses with computed flow
        flow_pose = torch.cat((poses, stacker.reshape(2, poses.shape[1], poses.shape[2], poses.shape[3])), 0)
        return flow_pose

if __name__=='__main__':
    flow = torch.load('../../data/UCF-101/flow/Archery/v_Archery_g01_c01.pt', map_location='cpu')
    pose = torch.load('../../data/UCF-101/pose/Archery/v_Archery_g01_c01.pt', map_location='cpu')
    
    flowPoseSampler_FAST = FlowPoseSampler_FAST('cpu')
    flow_pose = flowPoseSampler_FAST(flow, pose)
    