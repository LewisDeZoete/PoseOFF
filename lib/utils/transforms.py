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


if __name__ == '__main__':
    from ..utils.objects import ArgClass