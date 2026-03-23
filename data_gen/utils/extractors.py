import torch
import numpy as np
import re
from einops import rearrange
from data_gen.utils.preprocess import stack_frames
from data_gen.utils.postprocess import loop_graph, flow_mag_norm, pose_match
import cv2 as cv


class ToNumpy:
    def __call__(self, tensor):
        """
        Converts a tensor to a numpy array.
        Args:
            tensor (torch.Tensor): The input tensor to convert.
        Returns:
            np.ndarray: The converted numpy array.
        """
        return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor


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
        num_joints: int = 17,
        num_people_out: int = 2,
    ):
        self.detector = detector
        self.num_joints = num_joints
        self.num_people_out = num_people_out

    def __call__(self, video) -> torch.tensor:
        # Get pose results
        pose_results = self.detector(video, verbose=False)

        num_frames = len(pose_results)
        
        data_torch = torch.zeros(
            (
                3,  # channels (x,y,confidence)
                num_frames,  # total_frames
                self.num_joints,  # number of joints
                self.num_people_out,
            )
        )  # max number of people output

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

    def _time_inference(self, video):
        total_ms = 0.0

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Get pose results
        with torch.no_grad():
            start.record()
            pose_results = self.detector(video, verbose=False)
            end.record()

            total_ms += start.elapsed_time(end)

        # output_string = "{" + f"'frames': {video.shape[0]},'inference_time': {total_ms/1000:.4f}" + "},"
        # print(output_string)
        return total_ms


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
        minibatch_size (int): Number of batches to break up video processing into.

    Methods:
        __init__: pass
        __call__(self, video): video of shape: (n_frames, height, width, channels).
    """

    def __init__(self, model, device, minibatch_size: int = 8):
        self.model = model
        self.device = device
        self.minibatch_size = minibatch_size

        # Move the model to the corresponding device
        self.model.to(self.device)

    def __call__(self, video) -> torch.tensor:
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

        return flow

    def _time_inference(self, video):
        stacked = stack_frames(video)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        total_ms = 0.0

        with torch.no_grad():
            for i in range(0, stacked.shape[0], self.minibatch_size):
                minibatch = stacked[i : i + self.minibatch_size].to(self.device)

                start.record()
                flow_list = self.model(minibatch[:, 0, ...], minibatch[:, 1, ...])
                end.record()

                torch.cuda.synchronize()
                total_ms += start.elapsed_time(end)

        # output_string = "{" + f"'frames': {stacked.shape[0]},'inference_time': {total_ms/1000:.4f}" + "},"
        # print(output_string)
        return total_ms


class PoseOFFSampler:
    """
    A class to sample optical flow in windows surrounding pose keypoints.
    Attributes:
        device (torch.device): The device to run the computations on.
        window_size (int): The size of the window around each pose keypoint. Default is 3.
        threshold (float): The threshold for visibility of keypoints. Default is 0.5.
        loop (bool): Whether to loop the graph using loop_graph function. Default is True.
        norm (bool): Whether to normalize the flow magnitude using flow_mag_norm function. Default is False.
        match_pose (bool): Whether to match the pose keypoints using pose_match function. Default is True.
        ntu (bool): Whether the pose keypoints are from NTU dataset. Default is False.
        dilation (int): The dilation factor for sampling points around keypoints. Default is 1.
        debug_vis (bool): Default is False.
    TODO: scalable dilation with z-coordinate or mean joint distance.
    Methods:
        __call__(flows, poses):
            Samples the optical flow in windows surrounding the pose keypoints.
            Args:
                flows (torch.Tensor/np.array): The optical flow tensor of shape (num_frames-1, 2, height, width).
                poses (torch.Tensor): The pose keypoints tensor of shape (channels, num_frames, num_keypoints, num_people).
            Returns:
                torch.Tensor: Tensor of flow windows (optionally concat poses+flow).
    """
    def __init__(self, 
                 window_size: int = 3, 
                 threshold: float = 0.05, 
                 loop: bool = True,
                 norm: bool = False,
                 match_pose: bool = True,
                 ntu: bool = False,
                 dilation: int = 1,
                 debug_vis: bool = False):
        self.window_size = window_size  # Window size about pose keypoint
        self.half_k = self.window_size // 2  # Half the window size
        self.threshold = threshold
        self.dilation = dilation  # Dilation factor for sampling
        if loop:
            self.loop_graph = loop_graph
        if norm:
            self.norm = flow_mag_norm
        if match_pose:
            self.pose_match = pose_match
        self.ntu = ntu
        self.debug_vis = debug_vis

    def __call__(self, flows, poses):
        '''Call function to process optical flow and poses to produce PoseOFF
        Args:
            flows: (torch.Tensor/np.array) Optical flow array of shape (T, C, H, W)
            poses: (torch.Tensor/np.array) Pose array of shape (C, T, V, M)
        Samples the optical flow in windows surrounding the pose keypoints.
        Returns array of shape:
            (num_pose_channels*(window_size**2)*2,
            frames, 
            keypoints, 
            num_people)'''
        if isinstance(flows, torch.Tensor):
            flows = flows.cpu().numpy()
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()
        
        # Remove first frame of poses (no flow data)
        poses = poses[:, 1:, :, :]

        # Get the shape of the input tensors
        num_flow_frames, _, height, width = flows.shape
        channels, num_pose_frames, num_keypoints, num_people = poses.shape
        total_keypoints = num_keypoints * num_people
        
        if hasattr(self, 'pose_match'):
            poses = self.pose_match(poses)

        if self.ntu:  # NTU does not return confidence values for keypoints
            # TODO: check a) poses array is needed anymore
            # Remove nan values (replace with zeros)
            pose_points = np.nan_to_num(poses, nan=0)
            # Scale between x:[0-1919] y:[0-1079]
            pose_points = (rearrange(pose_points, 'C T V M -> C T (V M)')
                        * np.array([(width - 1)/1920, (height - 1)/1080]).reshape(2, 1, 1)).astype(int)
            poses[0] = poses[0]/1920-0.5
            poses[1] = poses[1]/1080-0.5
        else: # Else we're assuming it has estimation confidence
            pose_points = ((poses[:2, ...] + 0.5).reshape(2, num_pose_frames, total_keypoints)
                        * np.array([width - 1, height - 1]).reshape(2, 1, 1)).astype(int)
            vis = poses[2, :, :].flatten() > self.threshold  # Visibility mask (frames, keypoints)


        # Exclude keypoints that are too close to the edge where the flow window is cut off
        if self.ntu:
            valid_indices = ((pose_points[0, :, :] >= self.half_k * self.dilation) & 
                             (pose_points[0, :, :] < width - self.half_k * self.dilation) & 
                             (pose_points[1, :, :] >= self.half_k * self.dilation) & 
                             (pose_points[1, :, :] < height - self.half_k * self.dilation))
        else:
            valid_indices = (vis.reshape(num_pose_frames, total_keypoints) & 
                             (pose_points[0, :, :] >= self.half_k * self.dilation) & 
                             (pose_points[0, :, :] < width - self.half_k * self.dilation) & 
                             (pose_points[1, :, :] >= self.half_k * self.dilation) & 
                             (pose_points[1, :, :] < height - self.half_k * self.dilation))

        # For stacking flow windows with poses
        stacker = np.zeros((self.window_size**2*2, num_pose_frames, total_keypoints))

        for frame_no, flow in enumerate(flows):
            for keypoint_num in range(total_keypoints):
                if valid_indices[frame_no, keypoint_num]:
                    x, y = pose_points[0, frame_no, keypoint_num], pose_points[1, frame_no, keypoint_num]
                    flow_window = flow[:, 
                                       y - self.half_k * self.dilation : y + self.half_k * self.dilation + 1 : self.dilation, 
                                       x - self.half_k * self.dilation : x + self.half_k * self.dilation + 1 : self.dilation]
                    stacker[:, frame_no, keypoint_num] = flow_window.flatten()
        
        if self.ntu and not self.debug_vis:
            flow_pose = stacker.reshape(stacker.shape[0], *poses.shape[1:])
        else:
            flow_pose = np.concatenate((poses, stacker.reshape(stacker.shape[0], *poses.shape[1:])), axis=0)
        
        if hasattr(self, 'loop_graph'):
            flow_pose = self.loop_graph(flow_pose)
        
        if hasattr(self, 'norm'):
            flow_pose = self.norm(flow_pose, flow_window=self.window_size)

        # Pad sequence to ensure it is of the same shape as the poses!
        # TODO: TEST THIS ON NTU GENDATA!!!
        if not self.ntu:
            flow_pose = np.pad(flow_pose, ((0, 0), (0, 1), (0, 0), (0, 0)), mode="constant")

        return flow_pose


class PoseOFFSampler_LK(PoseOFFSampler):
    def __init__(self, *args, lk_winSize=(15, 15), lk_maxLevel=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.lk_params = {
            "winSize": lk_winSize,
            "maxLevel": lk_maxLevel,
            "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        }


    def __call__(self, video, poses):
        '''Using the LK method of optical flow calculation to generate PoseOFF.
        CV implementation: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
        goodFeaturesToTrack returns list of length `max_corners`, of shape: [max_corners, 1, 2].
        For each corner, you can simply ravel to flatten the array and get (x,y) positions.
        NOTE: The raw poses (from denoised_skes_data) are of shape: (T, M, V, C)
            In the get_poseoff_samples.py loop, we reshape (poses = poses.transpose(3, 0, 2, 1)) -> (C, T, V, M)

        Args:
            video (torch.Tensor): Tensor of shape (n_frames, channels, height, width)
            poses (torch.Tensor): Pose keypoint tensor of shape (C, T, V, M)

        Returns:
            poseoff_aray: Array containing only the flow windows?? of shape:
                (C*window_size**2, num_pose_frames, total_keypoints)
        '''
        # OpenCV expects numpy arrays...
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()

        # Remove first frame of poses (no flow data)
        poses = poses[1:, ...]

        # Get some shapes of input tensors
        n_frames, height, width, _ = video.shape
        channels, num_pose_frames, num_keypoints, num_people = poses.shape
        total_keypoints = num_keypoints * num_people

        if self.ntu:  # NTU does not return confidence values for keypoints
            # Remove nan values (replace with zeros)
            pose_points = np.nan_to_num(poses, nan=0)
            # Scale between x:[0-1919] y:[0-1079]
            pose_points = (rearrange(pose_points, 'C T V M -> C T (V M)')
                        * np.array([(width - 1)/1920, (height - 1)/1080]).reshape(2, 1, 1)).astype(int)
        else: # Else we're assuming it's 2D poses (x,y, conf.)
            pose_points = ((poses[:2, ...] + 0.5).reshape(2, num_pose_frames, total_keypoints)
                        * np.array([width - 1, height - 1]).reshape(2, 1, 1)).astype(int)
            vis = poses[2, :, :].flatten() > threshold  # Visibility mask (frames, keypoints)

        # Exclude keypoints that are too close to the edge where the flow window is cut off
        if self.ntu:
            # (T, total_keypoints)
            valid_indices = ((pose_points[0, :, :] >= self.half_k * self.dilation) &
                            (pose_points[0, :, :] < width - self.half_k * self.dilation) &
                            (pose_points[1, :, :] >= self.half_k * self.dilation) &
                            (pose_points[1, :, :] < height - self.half_k * self.dilation))
        else:
            # (T, total_keypoints)
            valid_indices = (vis.reshape(num_pose_frames, total_keypoints) &
                                (pose_points[0, :, :] >= self.half_k * self.dilation) &
                                (pose_points[0, :, :] < width - self.half_k * self.dilation) &
                                (pose_points[1, :, :] >= self.half_k * self.dilation) &
                                (pose_points[1, :, :] < height - self.half_k * self.dilation))

        # Get the first frame in order to calculate from from frame 0->1
        old_grey = cv.cvtColor(video[0], cv.COLOR_BGR2GRAY)

        # Create the array of just the optical flow windows ((C*H*W), T, V*M)
        flow_windows = np.zeros((self.window_size**2*2, num_pose_frames, total_keypoints))

        # Iterate over the frame numbers and keypoints
        for frame_num, frame in enumerate(video[1:]):
            frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Initialise points to track
            p0 = []
            skip_points = []
            for keypoint_num in range(total_keypoints):
                if valid_indices[frame_num, keypoint_num]:
                    x,y = pose_points[0, frame_num, keypoint_num], pose_points[1, frame_num, keypoint_num]
                    # Create grid of positions about each keypoint ((x,y), 5, 5)
                    grid = np.array(
                        np.meshgrid(
                            np.linspace(x-self.half_k*self.dilation, x+self.half_k*self.dilation, self.window_size).astype(int),
                            np.linspace(y-self.half_k*self.dilation, y+self.half_k*self.dilation, self.window_size).astype(int)
                        )
                    )
                    p0.append(grid)
                else:
                    # If keypoint is too close to screen edge...
                    p0.append(np.zeros((2, 5, 5)))
                    skip_points.append(keypoint_num)
                    pass
            # Reshape points to track...
            p0 = rearrange(np.array(p0), 'N C H W -> (N H W) 1 C').astype('float32')

            # Estimate the optical flow (LK method)
            p1, st, err = cv.calcOpticalFlowPyrLK(old_grey, frame_grey, p0, None, **self.lk_params)

            # Get vectors only for all keypoints on the frame (N=total_keypoints)
            # ((N H W) C) -> ((C H W) N) equivalent to flow_window.flatten
            flow_vectors = rearrange(
                (p1-p0).squeeze(),
                '(N H W) C -> (C H W) N',
                N=total_keypoints, H=self.window_size, W=self.window_size, C=2
            )

            flow_vectors[:, skip_points] = np.zeros(((self.window_size**2)*2, len(skip_points)))

            # Set frame to the calculated flow vectors for the keypoints within the frame
            flow_windows[:, frame_num] = flow_vectors
            old_grey = frame_grey.copy()

        # Reshape ((C H W) T (V M) -> (C H W) T V M)
        # Here, C is the x and y channels of flow, H and W are height and width respectively
        flow_windows = rearrange(flow_windows, 'W T (V M) -> W T V M', V=num_keypoints, M=num_people)
        return flow_windows


def flow_normal(video, alpha=1):
    '''Calculate normal flow from a given video.

    Args:
        video: Tensor of shape (n_frames, height, width, channels)

    Returns:
        norm_flows: Tensor of shape (n_frames-1, 2, H, W).
    '''
    video = np.array(video)
    norm_flows = torch.empty(video.shape[0]-1, video.shape[1], video.shape[2], 2)
    img1 = cv.cvtColor(video[0], cv.COLOR_BGR2GRAY)
    for frame_no, frame in enumerate(video[1:]):
        img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculate spatial gradients
        Ix = cv.Sobel(img1, cv.CV_64F, 1, 0, ksize=5)
        Iy = cv.Sobel(img1, cv.CV_64F, 0, 1, ksize=5)

        # Normal flow vectors
        norm_flows[frame_no, ...] = torch.stack((torch.tensor(Ix), torch.tensor(Iy)), dim=-1)

        # # Temporal gradient
        # It = img2.astype(float) - img1.astype(float)
        # (must add small factor in demoninator to avoid div by zero error)
        # norm_flows.append(-It / (alpha * (np.sqrt(Ix**2 + Iy**2) + 1e-6)))

        img1 = img2.copy()

    return norm_flows




def poseoff_lk(video, poses, window_size=3, ntu=False, dilation=1, debug_frame=None):
    '''Using the LK method of optical flow calculation...
    CV implementation: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    goodFeaturesToTrack returns list of length `max_corners`, of shape: [max_corners, 1, 2].
    For each corner, you can simply ravel to flatten the array and get (x,y) positions.
    NOTE: The raw poses (from denoised_skes_data) are of shape: (T, M, V, C)
        In the get_poseoff_samples.py loop, we reshape (poses = poses.transpose(3, 0, 2, 1)) -> (C, T, V, M)

    Args:
        video (torch.Tensor): Tensor of shape (n_frames, channels, height, width)
        poses (torch.Tensor): Pose keypoint tensor of shape (C, T, V, M)
        window_size (int): The size of the window around each pose keypoint. Default is 3.
        ntu (bool): Whether the pose keypoints are from NTU dataset. Default is False.
        dilation (int): The dilation factor for sampling points around keypoints. Default is 1.
        debug_frame (None/int): Optionally return the frame_number, the frame itself and
            the current state of the poseoff array. Default is None.

    Returns:
        poseoff_aray: Array containing only the flow windows?? of shape:
            (C*window_size**2, num_pose_frames, total_keypoints)
    '''
    lk_params = {
        "winSize": (15, 15),
        "maxLevel": 2,
        "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    }

    half_k = window_size // 2

    # # Remove first frame of poses (no flow data)
    # poses = poses[1:, ...]

    # Get some shapes of input tensors
    n_frames, height, width, _ = video.shape
    channels, num_pose_frames, num_keypoints, num_people = poses.shape
    total_keypoints = num_keypoints * num_people

    if ntu:  # NTU does not return confidence values for keypoints
        # Remove nan values (replace with zeros)
        pose_points = np.nan_to_num(poses, nan=0)
        # Scale between x:[0-1919] y:[0-1079]
        pose_points = (rearrange(pose_points, 'C T V M -> C T (V M)')
                    * np.array([(width - 1)/1920, (height - 1)/1080]).reshape(2, 1, 1)).astype(int)
    else: # Else we're assuming it's 2D poses (x,y, conf.)
        pose_points = ((poses[:2, ...] + 0.5).reshape(2, num_pose_frames, total_keypoints)
                    * np.array([width - 1, height - 1]).reshape(2, 1, 1)).astype(int)
        vis = poses[2, :, :].flatten() > threshold  # Visibility mask (frames, keypoints)

    # Exclude keypoints that are too close to the edge where the flow window is cut off
    if ntu:
        valid_indices = ((pose_points[0, :, :] >= half_k * dilation) &
                        (pose_points[0, :, :] < width - half_k * dilation) &
                        (pose_points[1, :, :] >= half_k * dilation) &
                        (pose_points[1, :, :] < height - half_k * dilation))
    else:
        valid_indices = (vis.reshape(num_pose_frames, total_keypoints) &
                            (pose_points[0, :, :] >= half_k * dilation) &
                            (pose_points[0, :, :] < width - half_k * dilation) &
                            (pose_points[1, :, :] >= half_k * dilation) &
                            (pose_points[1, :, :] < height - half_k * dilation))

    # Get the first frame in order to calculate from from frame 0->1
    old_grey = cv.cvtColor(video[0], cv.COLOR_BGR2GRAY)

    # Create the array of just the optical flow windows ((C*H*W), T, V*M)
    flow_windows = np.zeros((window_size**2*2, num_pose_frames, total_keypoints))

    # Iterate over the frame numbers and keypoints
    for frame_num, frame in enumerate(video[1:]):
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Initialise points to track
        p0 = []
        skip_points = []
        for keypoint_num in range(total_keypoints):
            if valid_indices[frame_num, keypoint_num]:
                x,y = pose_points[0, frame_num, keypoint_num], pose_points[1, frame_num, keypoint_num]
                # Create grid of positions about each keypoint ((x,y), 5, 5)
                grid = np.array(
                    np.meshgrid(
                        np.linspace(x-half_k*dilation, x+half_k*dilation, window_size).astype(int),
                        np.linspace(y-half_k*dilation, y+half_k*dilation, window_size).astype(int)
                    )
                )
                p0.append(grid)
            else:
                # If keypoint is too close to screen edge...
                p0.append(np.zeros((2, 5, 5)))
                skip_points.append(keypoint_num)
                pass
        # Reshape points to track...
        p0 = rearrange(np.array(p0), 'N C H W -> (N H W) 1 C').astype('float32')

        # Estimate the optical flow (LK method)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_grey, frame_grey, p0, None, **lk_params)

        # Get vectors only for all keypoints on the frame (N=total_keypoints idk why)
        # ((N H W) C) -> ((C H W) N) equivalent to flow_window.flatten
        flow_vectors = rearrange(
            (p1-p0).squeeze(),
            '(N H W) C -> (C H W) N',
            N=total_keypoints, H=window_size, W=window_size, C=2
        )
        flow_vectors[:, skip_points] = np.zeros((50, len(skip_points)))

        # Set frame to the calculated flow vectors for the keypoints within the frame
        flow_windows[:, frame_num] = flow_vectors
        old_grey = frame_grey.copy()
        # Get the flow vectors!
        if frame_num == debug_frame:
            print(f"Returning debug frame number {frame_num}")
            return frame_num, old_grey, stacker

    # Reshape ((C H W) T (V M) -> (C H W) T V M)
    # Here, C is the x and y channels of flow, H and W are height and width respectively
    flow_windows = rearrange(flow_windows, 'W T (V M) -> W T V M', V=num_keypoints, M=num_people)
    return flow_windows



# if __name__ == '__main__':
#     import os.path as osp
#     from config.argclass import ArgClass

#     from torchvision.models.optical_flow import raft_large
#     import torchvision.transforms.v2 as v2
#     from ultralytics import YOLO
#     from data_gen.utils import LoadVideo
#     import math

#     # Get the argparse object
#     arg = ArgClass(arg=f"./config/infogcn2/ntu/cnn.yaml")
#     transform_arg = arg.extractor

#     # Get the device
#     device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

#     # Create the model, move it to device and turn to eval mode
#     weights = torch.load(transform_arg['flow']['weights'], weights_only=True, map_location=device)
#     model = raft_large(progress=False)
#     model.load_state_dict(weights)
#     model = model.eval().to(device)
#     transform_flow = v2.Compose([
#         LoadVideo(max_frames=300),
#         v2.ToImage(),
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # map [0, 1] into [-1, 1]
#     ])

#     # Create the pose model
#     detector = YOLO(transform_arg['pose']['weights'])
#     detector.to(device)
#     transform_pose = v2.Compose([
#         LoadVideo(max_frames=300),
#         v2.Resize(size=(384,640)), # YOLO pose has a minimum input image size
#         v2.ToDtype(torch.float32),
#         v2.Lambda(lambda x: x/255.0), # Normalises the image to [0,1]
#     ])

#     # Create the flow and pose transforms separately
#     getFlow = GetFlow(model=model, device=device, minibatch_size=transform_arg['flow']['minibatch_size'])
#     getPose = GetPoses_YOLO(detector=detector, num_joints=17)

#     # Get one representative sample (100 frames, 2 bodies)
#     rgb_path = '../Datasets/NTU_RGBD/nturgb+d_rgb/'
#     ske_name = "S001C001P001R001A058"
#     rgb_name = osp.join(rgb_path, ske_name+"_rgb.avi")
#     rgb_flow = transform_flow(rgb_name)
#     rgb_pose = transform_pose(rgb_name)

#     # Pad to correct length...
#     T, *_ = rgb_flow.shape
#     rgb_flow_padded = torch.cat(
#         [rgb_flow for i in range(math.ceil(100/T))],
#         axis=0
#         )[:100]
#     T, *_ = rgb_pose.shape
#     rgb_pose_padded = torch.cat(
#         [rgb_pose for i in range(math.ceil(100/T))],
#         axis=0
#         )[:100]

#     print(f"RGB flow padded shape: {rgb_flow_padded.shape}")
#     print(f"RGB pose padded shape: {rgb_pose_padded.shape}")

#     # Dictionary for storing timing!
#     timing_dict = {'flow': [], 'pose': []}


#     for i in range(100):
#         timing_dict['flow'].append(getFlow._time_inference(rgb_flow))
#         timing_dict['pose'].append(getPose._time_inference(rgb_pose))
#     print(timing_dict)
#     # timing_dict['flow'].append(getFlow._time_inference(rgb_flow_padded))
#     # timing_dict['pose'].append(getPose._time_inference(rgb_pose_padded))
#     # print(timing_dict)


if __name__ == '__main__':
    from config.argclass import ArgClass
    from data.visualisations.interactive_vis import load_data, quick_view, draw_poseoff

    # Get the argparse object
    arg = ArgClass(arg=f"./config/infogcn2/ntu/cnn.yaml")
    transform_arg = arg.extractor
    # Change the dilation factor for testing
    dilation = 1
    transform_arg['poseoff']['dilation'] = dilation

    data_root = "./data/visualisations/RAW"
    sample_name = "S019C001P051R001A113"
    debug_frame_num = 10

    videos, poses, poseoff = load_data(data_root, sample_name)
    T, _, C = poses.shape
    M, V = (2, 25)
    pose=poses[debug_frame_num]
    poses = rearrange(poses, 'T (M V) C -> C T V M', C=C, T=T, V=V, M=M)
    flows, rgb, _ = videos

    print(f"Video shape (T, H, W, C): {rgb.shape}")
    print(f"Poses shape (C, T, V, M): {poses.shape}")

    print(f"Video min-max: {rgb.min()}-{rgb.max()}")
    print(f"Video dtype: {rgb.dtype}")

    # # Test the normal flow calculations
    # norm_flows = flow_normal(rgb)
    # print(f"Normal flows shape: {norm_flows.shape}")


    poseOFFSampler = PoseOFFSampler_LK(**transform_arg['poseoff'])
    print(vars(poseOFFSampler))
    poseoff = FPS(rgb, poses)
    print(f"PoseOFF features shape: {poseoff.shape}")

    frame = draw_poseoff(rgb[debug_frame_num], poseoff, pose, window_size=transform_arg['poseoff']['window_size'])
    print(f"Poseoff frame shape: {frame.shape}")
    cv.imwrite("./TMP.png", frame)
