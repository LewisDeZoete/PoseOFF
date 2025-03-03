import torch
import numpy as np
import re
from .preprocess import stack_frames
from .postprocess import loop_graph, flow_mag_norm, pose_match


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
    ):
        self.detector = detector
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.num_people_out = num_people_out

    def __call__(self, video) -> torch.tensor:
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
            # Get the frame number that yolo outputs in the frame.path attribute
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

        # Output from yolo is (x,y,conf), normalised between 0 and 1
        # Centralisation (about zero [-0.5 : 0.5])
        data_torch[0:2] = data_torch[0:2] - 0.5
        data_torch[1:2] = -data_torch[1:2]
        
        # Set x and y to zero if confidence is zero
        data_torch[0][data_torch[2] == 0] = 0
        data_torch[1][data_torch[2] == 0] = 0

        # sort by score
        sort_index = (-data_torch[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            # Took out a `.transpose(1,2,0)` on the second tensor... did it break?
            data_torch[:, t, :, :] = data_torch[:, t, :, s]
        data_torch = data_torch[:, :, :, 0:2].type(torch.float32)

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
    """
    A class to sample optical flow in windows surrounding pose keypoints.
    Attributes:
        device (torch.device): The device to run the computations on.
        window_size (int): The size of the window around each pose keypoint. Default is 3.
        threshold (float): The threshold for visibility of keypoints. Default is 0.5.
        loop (bool): Whether to loop the graph using loop_graph function. Default is True.
        to_cpu (bool): Whether to move the resulting tensor to CPU. Default is True.
        norm (bool): Whether to normalize the flow magnitude using flow_mag_norm function. Default is False.
    Methods:
        __call__(flows, poses):
            Samples the optical flow in windows surrounding the pose keypoints.
            Args:
                flows (torch.Tensor/np.array): The optical flow tensor of shape (num_frames-1, 2, height, width).
                poses (torch.Tensor): The pose keypoints tensor of shape (channels, num_frames, num_keypoints, num_people).
            Returns:
                torch.Tensor: The concatenated tensor of poses and sampled flow data.
                    shape: (num_pose_channels+(window_size**2)*2, frames-1, keypoints, num_people)
                    NOTE: The first frame of poses is discarded given it does not yet contain optical flow.
    """
    def __init__(self, 
                 window_size: int = 3, 
                 threshold: float = 0.05, 
                 loop: bool = True,
                 norm: bool = False,
                 match_pose: bool = True,
                 ntu: bool = False):
        self.window_size = window_size  # Window size about pose keypoint
        self.half_k = self.window_size // 2  # Half the window size
        self.threshold = threshold
        if loop:
            self.loop_graph = loop_graph
        if norm:
            self.norm = flow_mag_norm
        if match_pose:
            self.pose_match = pose_match
        self.ntu = ntu

    def __call__(self, flows, poses):
        """
        Samples the optical flow in windows surrounding the pose keypoints.
        Returns array of shape:
            (num_pose_channels+(window_size**2)*2,
            frames, 
            keypoints, 
            num_people)
        """
        if isinstance(flows, torch.Tensor):
            flows = flows.cpu().numpy()
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()
        
        # Remove first frame of poses (no flow data)
        poses = poses[:, 1:, :, :]

        # Get the shape of the input tensors
        num_frames, _, height, width = flows.shape
        channels, num_pose_frames, num_keypoints, num_people = poses.shape
        total_keypoints = num_keypoints*num_people
        
        if hasattr(self, 'pose_match'):
            poses = self.pose_match(poses)

        if self.ntu: # NTU does not return confidence values for keypoints
            # Scale pose keypoints to image size
            pose_points = np.nan_to_num(poses, nan=0)
            pose_points = (pose_points.reshape(2, num_pose_frames, total_keypoints)
                        * np.array([(width - 1)/1920, (height - 1)/1080]).reshape(2, 1, 1)).astype(int)
            # NTU keypoints need to be scaled [-0.5, 0.5]
            poses[0] = poses[0]/1920-0.5
            poses[1] = poses[1]/1080-0.5
        else:
            # Get and scale YOLO pose keypoints
            pose_points = ((poses[:2, ...] + 0.5).reshape(2, num_pose_frames, total_keypoints)
                        * np.array([width - 1, height - 1]).reshape(2, 1, 1)).astype(int)
            vis = poses[2, :, :].flatten() > self.threshold  # Visibility mask (frames, keypoints)

        # Prepare tensor to stack flow windows
        stacker = np.zeros((self.window_size**2*2, num_pose_frames, total_keypoints))

        # Create a grid of valid indices (filter out points close to the image border)
        if self.ntu: # No visibility mask, NTU doesn't return conf values for keypoints
            valid_indices = ((pose_points[0, :, :] >= self.half_k) & (pose_points[0, :, :] < width - self.half_k) & 
                             (pose_points[1, :, :] >= self.half_k) & (pose_points[1, :, :] < height - self.half_k))
        else: 
            valid_indices = (vis.reshape(num_pose_frames, total_keypoints) & 
                             (pose_points[0, :, :] >= self.half_k) & (pose_points[0, :, :] < width - self.half_k) & 
                             (pose_points[1, :, :] >= self.half_k) & (pose_points[1, :, :] < height - self.half_k))
                        
        # Loop through the frames and sample flow in window around each valid keypoint
        for frame_no, flow in enumerate(flows):
            for keypoint_num in range(total_keypoints):
                if valid_indices[frame_no, keypoint_num]:
                    x, y = pose_points[0, frame_no, keypoint_num], pose_points[1, frame_no, keypoint_num]
                    # Get the window of optical flow and calculate mean directly
                    flow_window = flow[:, y - self.half_k : y + self.half_k + 1, x - self.half_k : x + self.half_k + 1]
                    
                    stacker[:, frame_no, keypoint_num] = flow_window.flatten()
        
        # Concatenate poses with computed flow
        flow_pose = np.concatenate((poses, stacker.reshape(stacker.shape[0], *poses.shape[1:])), axis=0)
        
        # If we pass loop_graph = True, then loop the graph using this function!
        if hasattr(self, 'loop_graph'):
            flow_pose = self.loop_graph(flow_pose)
        
        if hasattr(self, 'norm'):
            flow_pose = self.norm(flow_pose, flow_window=self.window_size)
        
        return flow_pose


class FlowPoseSampler_backup:
    """
    A class to sample optical flow in windows surrounding pose keypoints.
    Attributes:
        device (torch.device): The device to run the computations on.
        window_size (int): The size of the window around each pose keypoint. Default is 3.
        threshold (float): The threshold for visibility of keypoints. Default is 0.5.
        loop (bool): Whether to loop the graph using loop_graph function. Default is True.
        to_cpu (bool): Whether to move the resulting tensor to CPU. Default is True.
        norm (bool): Whether to normalize the flow magnitude using flow_mag_norm function. Default is False.
    Methods:
        __call__(flows, poses):
            Samples the optical flow in windows surrounding the pose keypoints.
            Args:
                flows (torch.Tensor/np.array): The optical flow tensor of shape (num_frames, 2, height, width).
                poses (torch.Tensor): The pose keypoints tensor of shape (3, num_frames, num_keypoints, num_people).
            Returns:
                torch.Tensor: The concatenated tensor of poses and sampled flow data.
    """
    def __init__(self, 
                 window_size: int = 3, 
                 threshold: float = 0.5, 
                 loop: bool = True,
                 norm: bool = False,
                 match_pose: bool = True):
        self.window_size = window_size  # Window size about pose keypoint
        self.half_k = self.window_size // 2  # Half the window size
        self.threshold = threshold
        if loop:
            self.loop_graph = loop_graph
        if norm:
            self.norm = flow_mag_norm
        if match_pose:
            self.pose_match = pose_match

    def __call__(self, flows, poses):
        """
        Samples the optical flow in windows surrounding the pose keypoints.
        Returns array of shape:
            (num_pose_channels+(window_size**2)*2,
            frames, 
            keypoints, 
            num_people)
        """
        if isinstance(flows, torch.Tensor):
            flows = flows.cpu().numpy()
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()
        
        num_frames, _, height, width = flows.shape
        num_keypoints = poses.shape[2]*poses.shape[3]
        
        if self.pose_match:
            poses = self.pose_match(poses)

        # Calculate pose points and visibility mask outside of the loop
        pose_points = ((poses[:2, :, :] + 0.5).reshape(2, poses.shape[1], num_keypoints)
                       * np.array([width - 1, height - 1]).reshape(2, 1, 1)).astype(int)
        vis = poses[2, :, :].flatten() > self.threshold  # Visibility mask (frames, keypoints)

        # Prepare tensor to stack flow windows
        stacker = np.zeros((self.window_size**2*2, poses.shape[1], num_keypoints))

        # Create a grid of valid indices (filter out points close to the image border)
        valid_indices = (vis.reshape(poses.shape[1], num_keypoints) & 
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
        flow_pose = np.concatenate((poses, stacker.reshape(stacker.shape[0], *poses.shape[1:])), axis=0)
        
        # If we pass loop_graph = True, then loop the graph using this function!
        if hasattr(self, 'loop_graph'):
            flow_pose = self.loop_graph(flow_pose)
        
        if hasattr(self, 'norm'):
            flow_pose = self.norm(flow_pose, flow_window=self.window_size)
        
        return flow_pose


if __name__ == '__main__':
    import sys
    import os

    # # add lib to path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '../...')))

    from config.argclass import ArgClass
    from data_gen.utils.extract_utils import extract_data

    arg = ArgClass(arg='./config/custom_pose/train_base.yaml')

    flowposeSampler = FlowPoseSampler(device=torch.device('cpu'), norm=True)
    extract_data(arg, 0, flowposeSampler, 'flowpose', save_as_numpy=True, debug=True)
    
    # from ultralytics import YOLO
    # from preprocess import LoadVideo
    # from torchvision.transforms import v2

    # # Get the config dict (including labels)
    # arg = ArgClass(arg='./config/custom_pose/train_joint_infogcn.yaml')
    # transform_args = arg.extractor['pose']

    # # Get the device
    # device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

    # # Create the detector (YOLO pose)
    # detector = YOLO(transform_args['weights'])
    # detector.to(device)
    # # Create the pose extractor object
    # transforms = v2.Compose([
    #     v2.Resize(size=(384,640)), # YOLO pose has a minimum input image size
    #     v2.ToDtype(torch.float32),
    #     v2.Lambda(lambda x: x/255.0), # Normalises the image to [0,1]
    #     GetPoses_YOLO(detector=detector, max_frames=300, num_joints=17)
    # ])

    # # Load a video and extract poses
    # video_paths = [os.path.join(arg.feeder_args['data_paths']['rgb_path'],
    #                       (list(arg.feeder_args['labels'].keys())[i]+'.avi')) \
    #         for i in range(len(list(arg.feeder_args['labels'].keys())))]

    # print(video_paths[0])
    
    # # Test pose extraction method
    # print("Testing pose extraction method...")
    # for video_path in video_paths:
    #     video = LoadVideo(video_path, max_frames=300)
    #     poses = transforms(video)
    #     print(poses.shape)
    #     print(poses[:, 0, :, :])
    #     break