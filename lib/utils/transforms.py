import numpy as np
import torch
import decord
from decord import VideoReader, cpu
import re
from .flow import stack_frames


def LoadVideo(video_path, max_frames) -> torch.tensor:
    '''
    ARGS:
        `max_frames` - maximum number of output frames
    
    OUTPUT:
        `result` - on `__call__`, when passing in a `result` dict with the
            key ['video_path'], it will get the frames of that video.
            `output.shape == (nframes (max_frames), height, width, color)`
    '''
    decord.bridge.set_bridge('torch')
    vr = VideoReader(video_path, ctx=cpu(0))
    # if there's too many frames, get `max_frames` linearly spaced frames
    if len(vr) > max_frames:
        # output = torch.tensor(vr.get_batch(np.linspace(0, len(vr)-1, self.max_frames)).asnumpy()) 
        output = vr.get_batch(np.linspace(0, len(vr)-1, max_frames))
    else:
        # output = torch.tensor(vr.get_batch(np.linspace(0, len(vr)-1, len(vr))).asnumpy())
        output = vr.get_batch(np.linspace(0, len(vr)-1, len(vr)))

    # RGB Colour format
    output = torch.permute(output, (0, 3, 1, 2))

    return output


class GetPoses_YOLO:
    '''
    Creates a numpy array of shape:
        (channels, max_frames, num_joints, max_number_people)
        (3,        300,        17,         2)
    Only parses video frames up to max_frames, the rest are skipped.
    '''
    def __init__(self, detector, max_frames: int = 300, num_joints: int = 17, num_people_out: int = 2, resdict=False):
        self.detector = detector
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.num_people_out = num_people_out
        self.resdict = resdict

    def __call__(self, video) -> torch.tensor:
        # If input is a dicitonary, we want the video to work with
        if isinstance(video, dict):
            video = video['rgb']
        # If not, it could be we just want the poses
        else:
            # OR we may want to output a dict but we need to create it
            if self.resdict:
                results = {'rgb': video}

        data_torch = torch.zeros((3,                # channels (x,y,confidence)
                            self.max_frames,        # total_frames
                            self.num_joints,        # number of joints
                            self.num_people_out))   # max number of people output
        
        # Get pose results
        pose_results = self.detector(video, verbose=False)
        
        # Get data from yolo
        for frame in pose_results:
            # Get the frame number that yolo outputs in the frame.path variable
            frame_index = int(re.findall(r'\d+',frame.path)[0])
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
                data_torch[0, frame_index, :, m] = person.xyn[0,:,0]
                data_torch[1, frame_index, :, m] = person.xyn[0,:,1]
                data_torch[2, frame_index,:, m] = person.conf[0]
        
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
            results['pose'] = data_torch
            return results
        # Else simply return the poses
        else:
            return data_torch


class GetFlow:
    def __init__(self, model, device, resdict=False, minibatch_size:int=8):
        self.model = model
        self.device = device
        self.resdict = resdict
        self.minibatch_size = minibatch_size

        # Move the model to the corresponding device
        self.model.to(self.device)

    def __call__(self, video) -> torch.tensor:
        # If input is a dicitonary, we want the video to work with
        if isinstance(video, dict):
            video = video['rgb']
        # If not, it could be we just want the poses
        else:
            # OR we may want to output a dict but we need to create it
            if self.resdict:
                results = {'rgb': video}

        # Stack the frames in a tensor that looks like: [[0, 1],
        stacked = stack_frames(video) #                  [1, 2]] etc.
        # NOTE: this returns the video video stacked, we're batching it
        
        # Create a flow list, then calculate flow with raft (no_grad)
        flow = []
        with torch.no_grad():
            # process each video in batches (faced OOM issues)
            for i in range(0, stacked.shape[0], self.minibatch_size):
                minibatch = stacked[i:i+self.minibatch_size].to(self.device)

                # Calcualte the flow (returns a list of length 12, last element
                # is the the last pass of the model and most accurate flow
                flow_list = self.model(minibatch[:,0,...], minibatch[:,1,...])
                flow.append(flow_list[-1])
            
        # Concatenate the list elements back into one array!
        flow = torch.cat(flow, axis=0)
        # If we're expecting a dictionary output, add the new key
        if self.resdict:
            results['flow'] = flow
            return results
        # Otherwise simply return the flow
        else:
            return flow


def compute_divergence_and_curl(flows, poses, kernel_size=3, threshold=0.5):
    """
    Calculate divergence and curl of an optical flow field at specified poses.

    Args:
        flow (torch.Tensor): A tensor of shape (num_frames, 2, height, width),
                                     where the second dimension represents (u, v) optical flow.
        poses (torch.Tensor): A tensor of pose keyposes of shape 
                                (channels (x,y,v), max frames (300), keyposes (17), max persons (2)).
        kernel_size (int): The size of the neighborhood to consider for calculations.
        threshold (float): Confidence threshold for rejecting a pose keypoint.

    Returns:
        divergences (list): List of divergence values at the specified poses.
        curls (list): List of curl values at the specified poses.
    TODO: move to transforms!
    """
    num_frames, _, height, width = flows.shape

    # Define the kernel size for neighborhood calculations
    half_k = kernel_size // 2

    for i, flow in enumerate(flows):
        # Reshape the pose points to be a continuous array
        pose_points = poses[:2,i].reshape(2,34)
        # Get visibility tensor to use as a mask
        vis = poses[2,i].reshape(34)
        vis = vis > threshold
        # Rescale values from (-0.5:0.5) to (0,height-1) since we're using it to index
        pose_points[0] = (pose_points[0]+0.5)*(width-1)
        pose_points[1] = (pose_points[1]+0.5)*(height-1)
        pose_points = pose_points.type(torch.int)

        # Get the divergence and curl for each keypoint
        for keypoint_num in range(pose_points.shape[1]):
            x,y = poses[0,keypoint_num], poses[1,keypoint_num]
            # disregard the div and curl if the keypoint is too close to the edge
            if y < half_k or y >= height - half_k or x < half_k or x >= width - half_k:
                pass
                
        # For now just simply getting the flow points
        flow_points = flows[i,:, pose_points[1], pose_points[0]]*vis


if __name__ == '__main__':
    from ..utils.objects import ArgClass