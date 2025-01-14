import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '../..')))

import numpy as np
import torch
import decord
from decord import VideoReader, cpu
import re
from lib.utils.flow import stack_frames
import lib.utils.augments as augments


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



class pose_match(object):
    '''
    Pytorch version of the pose_match augmentation.
    Matches skeletons across video using square of distance between frames.
    Only takes pose as the input data.'''
    def __call__(self, data):
        C, T, V, M = data.shape
        assert (C == 3)
        score = data[2, :, :, :].sum(axis=1)
        # the rank of body confidence in each frame (shape: T-1, M)
        rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

        # data of frame 1
        xy1 = data[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
        # data of frame 2
        xy2 = data[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
        # square of distance between frame 1&2 (shape: T-1, M, M)
        distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

        # match pose
        forward_map = torch.zeros((T, M), dtype=int) - 1
        forward_map[0] = torch.arange(2)
        for m in range(M):
            choose = (rank == m)
            forward = distance[choose].argmin(axis=1)
            for t in range(T - 1):
                distance[t, :, forward[t]] = np.inf
            forward_map[1:][choose] = forward
        assert (torch.all(forward_map >= 0))

        # string data
        for t in range(T - 1):
            forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

        # generate data
        new_data = torch.zeros(data.shape)
        for t in range(T):
            new_data[:, t, :, :] = data[:, t, :, forward_map[t]]#.permute(1, 2, 0)
        data = new_data

        # score sort
        trace_score = data[2, :, :, :].sum(axis=1).sum(axis=0)
        rank = (-trace_score).argsort()
        data_numpy = data[:, :, :, rank]

        return data_numpy


class FlowPoseSampler:
    """Sample optical flow using input poses.
    NOTE: Assumes pre-processed optical flow and poses (for now).
    NOTE: Utilises the MultiStreamDataset, it passes in the loaded arrays.
    """

    def __init__(self, 
                 device: torch.device, 
                 window_size: int = 3, 
                 threshold: float = 0.5, 
                 loop_graph: bool = True,
                 to_cpu: bool = True,
                 norm: bool = True):
        self.device = device  # Making sure all tensors are on the same device
        self.window_size = window_size  # Window size about pose keypoint
        self.half_k = self.window_size // 2  # Half the window size
        self.threshold = threshold
        self.to_cpu = to_cpu
        if loop_graph:
            self.loop_graph = augments.loop_graph()
        if norm:
            self.norm = augments.flow_mag_norm(flow_window=window_size)
        self.pose_match = pose_match()

    def __call__(self, flows, poses):
        """Samples the optical flow in windows surrounding the pose keypoints.
        Returns array of shape (5, frames, keypoints, num_people) e.g., (5, 300, 17, 2)"""
        num_frames, _, height, width = flows.shape
        num_keypoints = poses.shape[2]*poses.shape[3]
        poses = self.pose_match(poses)
        
        # Convert poses once to the correct device and scale for indexing
        poses = poses.to(self.device)
        # Calculate pose points and visibility mask outside of the loop
        pose_points = ((poses[:2, :, :] + 0.5).contiguous().view(2,poses.size()[1],num_keypoints)
                       * torch.tensor([width - 1, height - 1], device=self.device).view(2, 1, 1)).type(torch.int)
        vis = poses[2, :, :].flatten() > self.threshold  # Visibility mask (frames, keypoints)

        # Prepare stacker tensor for the flow data (pre-allocate the size)
        stacker = torch.zeros((self.window_size**2*2, poses.shape[1], num_keypoints), device=self.device)

        # Create a grid of valid indices (filter out bad points upfront)
        valid_indices = (vis.view(poses.shape[1], num_keypoints) & 
                         (pose_points[0, :, :] >= self.half_k) & (pose_points[0, :, :] < width - self.half_k) & 
                         (pose_points[1, :, :] >= self.half_k) & (pose_points[1, :, :] < height - self.half_k))
                        
        # Loop through the frames and sample flow in window around each valid keypoint
        for i in range(num_frames):
            flow = flows[i]
            for keypoint_num in range(num_keypoints):
                if valid_indices[i+1, keypoint_num]:
                    x, y = pose_points[0, i+1, keypoint_num], pose_points[1, i+1, keypoint_num]
                    # Get the window of optical flow and calculate mean directly
                    flow_window = flow[:, y - self.half_k : y + self.half_k + 1, x - self.half_k : x + self.half_k + 1]
                    
                    stacker[:, i+1, keypoint_num] = flow_window.flatten()
        
        # Concatenate poses with computed flow
        flow_pose = torch.cat((poses, stacker.view(stacker.size()[0], *poses.size()[1:])), dim=0)
        
        # If we pass loop_graph = True, then loop the graph using this function!
        if hasattr(self, 'loop_graph'):
            flow_pose = self.loop_graph(flow_pose)
        
        if hasattr(self, 'norm'):
            flow_pose = self.norm(flow_pose)

        # By default, move the flow_pose to the cpu (since we're manipulating with numpy)
        if self.to_cpu:
            flow_pose.to('cpu')
        
        return flow_pose



if __name__=='__main__':
    from objects import ArgClass

    arg = ArgClass('./data_gen/UCF-101_config.yaml')

    flow = torch.load('./data/UCF-101/flow/Archery/v_Archery_g01_c01.pt',
                      map_location='cpu')
    pose = torch.load('./data/UCF-101/pose/Archery/v_Archery_g01_c01.pt',
                      map_location='cpu')
    
    # Create the flow pose sampler (testing on cpu)
    FPTransform = FlowPoseSampler('cpu', **arg.flowpose)
    
    # Sample optical flow based on pose points
    print('Input shapes:')
    print(f'\tFlow: {flow.shape}\n\tPose: {pose.shape}')
    flow_pose = FPTransform(flow,pose)
    print(f'Flow pose sampler output shape: {flow_pose.shape}')
    print(f'Frame 0: {flow_pose[:5,0,0,0]}')
    print(f'Frame len(flow): {flow_pose[:5,len(flow),0,0]}')
    print(f'Frame len(flow)+1: {flow_pose[:5,len(flow)+1,0,0]}')
    print(f'Frame -1: {flow_pose[:5,-1,0,0]}')