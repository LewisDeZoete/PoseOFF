import torch
import numpy as np
import re
from einops import rearrange
from data_gen.utils.postprocess import loop_graph, flow_mag_norm, pose_match
import cv2


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
        (channels, num_frames, num_joints, max_number_people)
        (3,        ~300(variable),        17,         2)
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

        # TODO: more efficiently by using inbuilt YOLO keypoints:
        # https://docs.ultralytics.com/tasks/pose#predict

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
        _stack_frames(self, frames): stack frames for RAFT flow inference, outut shape (N,2,C,H,W).
        _time_inference (self, video): time how long flow inference takes using CUDA timing.
    """

    def __init__(self, model, device, minibatch_size: int = 8):
        self.model = model
        self.device = device
        self.minibatch_size = minibatch_size

        # Move the model to the corresponding device
        self.model.to(self.device)

    def __call__(self, video) -> torch.tensor:
        # Stack the frames in a tensor that looks like: [[0, 1],
        stacked = self._stack_frames(video)  #                  [1, 2]] etc.
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

    def _stack_frames(self, frames: torch.Tensor):
        """
        Stack adjacent pairs of frames.

        Args:
            frames (torch.Tensor): Preprocessed input frames to be stacked prior to calculating flow, shape (N,C,H,W).

        Returns:
            frame_pairs (torch.Tensor): Stacked frames, where frame N is at index (:,0,...) and frame N+1 is at index (:,1,...).
            Shape: (N,2,C,H,W)
        """
        frame_pairs = torch.zeros(tuple([frames.shape[0] - 1, 2]) + tuple(frames.shape[1:]))
        for i in range(len(frames) - 1):
            frame_pairs[i, 0] = frames[i]
            frame_pairs[i, 1] = frames[i + 1]
        return frame_pairs

    def _time_inference(self, video):
        stacked = self._stack_frames(video)

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
        return total_ms


class PoseOFFSampler:
    """A class to sample optical flow in windows surrounding pose keypoints.
    Attributes:
        device (torch.device): The device to run the computations on.
        window_size (int): The size of the window around each pose keypoint. Default is 3.
        threshold (float): The threshold for visibility of keypoints. Default is 0.05.
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
    def __init__(
            self,
            *args,
            window_size: int = 3,
            threshold: float = 0.05,
            loop: bool = True,
            norm: bool = False,
            match_pose: bool = True,
            ntu: bool = False,
            dilation: int = 1,
            debug_vis: bool = False,
            **kwargs
    ):
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
        """Call function to process optical flow and poses to produce PoseOFF
        Args:
            flows: (torch.Tensor/np.array) Optical flow array of shape (T, C, H, W)
            poses: (torch.Tensor/np.array) Pose array of shape (C, T, V, M)
        Samples the optical flow in windows surrounding the pose keypoints.
        Returns array of shape:
            (num_pose_channels*(window_size**2)*2,
            frames, 
            keypoints, 
            num_people)"""
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

        # If it's NTU data, just return the windows
        if self.ntu and not self.debug_vis:
            poseoff = stacker.reshape(stacker.shape[0], *poses.shape[1:])
        else:
            poseoff = np.concatenate((poses, stacker.reshape(stacker.shape[0], *poses.shape[1:])), axis=0)
        
        if hasattr(self, 'loop_graph'):
            poseoff = self.loop_graph(poseoff)
        
        if hasattr(self, 'norm'):
            poseoff = self.norm(poseoff, flow_window=self.window_size)

        # Pad sequence to ensure it is of the same shape as the poses!
        if not self.ntu:
            poseoff = np.pad(poseoff, ((0, 0), (0, 1), (0, 0), (0, 0)), mode="constant")

        return poseoff

    def _get_dilation(self, pose):
        '''TODO: Write a get dilation function that picks a dilation value based on apparent size skels'''
        pass


class PoseOFFSampler_LK(PoseOFFSampler):
    """Calculate optical flow windows taking samples from pose keypoints. Child class of PoseOFFSampler.

    Attributes:
        lk_params: Dictionary of Lukas-Kanade attributes input to OpenCV LK Flow estimation.
        mag_threshold: Optical flow magnitude threshold below which optical flow vectors are set to zero.
    """
    def __init__(
            self,
            *args,
            lk_winSize: tuple[int] = (15, 15),
            lk_maxLevel: int = 3,
            mag_threshold: int = 100,
            **kwargs
    ):
        """Initialises the PoseOFF Sampler.

        Args:
            lk_winSize ((int, int): winSize parameter for CV2 lk params. Default is (15, 15).
            lk_maxLevel (int): maxLevel parameter for CV2 lk params. Default is 3.
            mag_threshold (int): Optical flow magnitude threshold, limiting how large flow arrows will be. Default is 100.
        """
        super().__init__(*args, **kwargs)
        self.lk_params = {
            "winSize": lk_winSize,
            "maxLevel": lk_maxLevel,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        self.mag_threshold = mag_threshold

    def __call__(self, video, poses):
        """Using the LK method of optical flow calculation to generate PoseOFF.
        CV implementation: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
        goodFeaturesToTrack returns list of length `max_corners`, of shape: [max_corners, 1, 2].
        For each corner, you can simply ravel to flatten the array and get (x,y) positions.
        NOTE: The raw poses (from denoised_skes_data) are of shape: (T, M, V, C)
            In the get_poseoff_samples.py loop, we reshape (poses = poses.transpose(3, 0, 2, 1)) -> (C, T, V, M)

        Args:
            video (torch.Tensor): Tensor of a video, either (T, C, H, W) or (T, H, W, C)
            poses (torch.Tensor): Pose keypoint tensor of shape (C, T, V, M)
                C is 2 for ntu denoised data, or C for poses extracted by YOLO pose...

        Returns:
            poseoff_aray: Array containing only the flow windows of shape:
                (C*window_size**2, num_pose_frames, total_keypoints)
        """
        # OpenCV expects numpy arrays...
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()

        # Remove first frame of poses (no flow data)
        poses = poses[:, 1:, ...]

        # Open CV assumes image format as ((T) H W C)
        if video.shape[-1] > 3:
            video = rearrange(video, 'T C H W -> T H W C')

        # Get some shapes of input tensors
        n_frames, height, width, _ = video.shape
        pose_channels, num_pose_frames, num_keypoints, num_people = poses.shape
        total_keypoints = num_keypoints * num_people

        if self.ntu:  # NTU does not return confidence values for keypoints
            # Remove nan values (replace with zeros), only take x,y values
            pose_points = np.nan_to_num(poses[:2], nan=0)
            # Scale between x:[0-1919] y:[0-1079]
            pose_points = (rearrange(pose_points, 'C T V M -> C T (V M)')
                        * np.array([(width - 1)/1920, (height - 1)/1080]).reshape(2, 1, 1)).astype(int)
        else: # Else we're assuming it's 2D poses (x,y, conf.)
            pose_points = ((poses[:2, ...] + 0.5).reshape(2, num_pose_frames, total_keypoints)
                        * np.array([width - 1, height - 1]).reshape(2, 1, 1)).astype(int)
            vis = poses[2, :, :].flatten() > self.threshold  # Visibility mask (frames, keypoints)

        # Exclude keypoints that are too close to the edge where the flow window is cut off
        if self.ntu:
            valid_indices = ( # (T, total_keypoints)
                (pose_points[0, :, :] >= self.half_k * self.dilation) &
                (pose_points[0, :, :] < width - self.half_k * self.dilation) &
                (pose_points[1, :, :] >= self.half_k * self.dilation) &
                (pose_points[1, :, :] < height - self.half_k * self.dilation)
            )
        else:
            valid_indices = ( # (T, total_keypoints)
                vis.reshape(num_pose_frames, total_keypoints) &
                (pose_points[0, :, :] >= self.half_k * self.dilation) &
                (pose_points[0, :, :] < width - self.half_k * self.dilation) &
                (pose_points[1, :, :] >= self.half_k * self.dilation) &
                (pose_points[1, :, :] < height - self.half_k * self.dilation)
            )

        # Get the first frame in order to calculate from from frame 0->1
        old_grey = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)

        # Create the array of just the optical flow windows ((C*H*W), T, V*M)
        flow_windows = np.zeros((self.window_size**2*2, num_pose_frames, total_keypoints))

        # Iterate over the frame numbers and keypoints
        for frame_num, frame in enumerate(video[1:]):
            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Initialise points to track
            p0 = []
            skip_points = []
            for keypoint_num in range(total_keypoints):
                if valid_indices[frame_num, keypoint_num]:
                    x,y = pose_points[0, frame_num, keypoint_num], pose_points[1, frame_num, keypoint_num]
                    # Create grid of positions about each keypoint ((x,y), window_size, window_size)
                    grid = np.array(
                        np.meshgrid(
                            np.linspace(x-self.half_k*self.dilation, x+self.half_k*self.dilation, self.window_size).astype(int),
                            np.linspace(y-self.half_k*self.dilation, y+self.half_k*self.dilation, self.window_size).astype(int)
                        )
                    )
                    p0.append(grid)
                else:
                    # Keypoint too close to screen edge, add zeros, later removed anyway
                    p0.append(np.ones((2, self.window_size, self.window_size))*10)
                    skip_points.append(keypoint_num)

            # Reshape points to track...
            p0 = rearrange(np.array(p0), 'N C H W -> (N H W) 1 C').astype('float32')

            # Estimate the optical flow (LK method)
            try:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_grey, frame_grey, p0, None, **self.lk_params)
            except Exception as e:
                print("LK Flow calculation failed, likely due to incorrect dtype (must be uint8)")
                print(e)
                break

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
        flow_windows = rearrange(flow_windows, 'C T (V M) -> C T V M', V=num_keypoints, M=num_people)

        # If it's NTU data, just return the windows
        if self.ntu and not self.debug_vis:
            poseoff = flow_windows.reshape(flow_windows.shape[0], *poses.shape[1:])
        # Otherwise, concatenate pose and flow channels
        else:
            poseoff = np.concatenate((poses, flow_windows.reshape(flow_windows.shape[0], *poses.shape[1:])), axis=0)

        # Pad sequence to ensure it is of the same shape as the poses!
        # (53, 299, 17, 2) -> (53, 300, 17, 2)
        if not self.ntu:
            poseoff = np.pad(poseoff, ((0, 0), (0, 1), (0, 0), (0, 0)), mode="constant")

        return poseoff

    def _get_dilation(self, pose):
        '''TODO: Write a get dilation function that picks a dilation value based on apparent size skels'''
        pass


def _temporal_gradient_5pt(frames: np.ndarray, pad: bool = True, dt: float = 1.0) -> np.ndarray:
    """Temporal gradient via 5-point stencil, vectorised over all spatial dims.
    Kernel: 1/12 * [-1, 8, 0, -8, 1]

    Args:
        frames: float32 array of shape (T, H, W).
        pad: if true, use pad start/end of array and use 5pt gradient across all. Default is True.
        dt: Time step (frame interval). Default is 1.0.

    Returns:
        I_t: float32 array of shape (T, H, W).
    """
    T = frames.shape[0]
    I_t = np.empty_like(frames, dtype=np.float32)

    # If using padding, pad start and end frames and use 4th order central difference
    if pad:
        # First pad by duplicating the first and last frames
        frames = np.pad(frames, ((2,2),(0,0),(0,0)), 'edge')

        # Only use 4th order central difference (two frames on each end are lost anyway)
        I_t = (-frames[4:] + 8.0 * frames[3:-1]
               - 8.0 * frames[1:-3] + frames[0:-4]) / (12.0 * dt)

    # Else, handle boundary conditions using lower order differencing
    else:
        if T >= 5:
            # Interior: 4th-order central difference
            I_t[2:-2] = (-frames[4:] + 8.0 * frames[3:-1]
                        - 8.0 * frames[1:-3] + frames[0:-4]) / (12.0 * dt)
            # Second and second-to-last: 3-point central
            I_t[1]  = (frames[2] - frames[0]) / (2.0 * dt)
            I_t[-2] = (frames[-1] - frames[-3]) / (2.0 * dt)
        elif T >= 3:
            I_t[1:-1] = (frames[2:] - frames[:-2]) / (2.0 * dt)

        # Boundary: first-order one-sided differences
        I_t[0]  = (frames[1] - frames[0]) / dt
        I_t[-1] = (frames[-1] - frames[-2]) / dt

    return I_t


def _temporal_gradient_sobel(frames: np.ndarray, dt: float = 1.0, **kwargs) -> np.ndarray:
    """Sobel filter for temporal gradient.
    Kernel: 1/8 * [-1, 8, 0, -8, 1]

    Args:
        frames: float32 array of shape (T, H, W).
        dt: Time step (frame interval). Default is 1.0.

    Returns:
        I_t: float32 array of shape (T, H, W).
    """
    I_t = np.empty_like(frames, dtype=np.float32)

    # First pad by duplicating the first and last frames
    frames = np.pad(frames, ((2,2),(0,0),(0,0)), 'edge')

    # Apply 1D sobel kernel
    I_t = (-frames[0:-4] - 2.0*frames[1:-3]
        + 2.0*frames[3:-1] + frames[4:]) / (8.0 * dt)

    return I_t


def _spatial_gradients(
        grey: np.ndarray,
        spatial_sobel_ksize: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Sobel spatial gradients for a single greyscale frame or array of frames.

    Args:
        grey: uint8 or float32 array of shape (H, W) or (T, H, W).
        ksize:      Sobel kernel size (3 or 5 recommended).

    Returns:
        I_x, I_y: float32 arrays of shape (H, W) or (T, H, W).
    """
    if len(grey.shape) == 2: grey = [grey]
    # Shape: (T, H, W) each.  Sobel on float32 frames.
    I_x = np.stack(
        [cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=spatial_sobel_ksize) for g in grey],
        axis=0,
    )  # (T, H, W)
    I_y = np.stack(
        [cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=spatial_sobel_ksize) for g in grey],
        axis=0,
    )  # (T, H, W)
    return I_x, I_y


class PoseOFFSampler_NF(PoseOFFSampler):
    """Normal-flow sampler — drop-in replacement for PoseOFFSampler_LK.

    Computes the normal-flow vector at each point of a sparse mesh grid
    centred on each valid pose keypoint, using only local intensity
    gradients (no iterative solver, no descriptor, no pyramid).

    Attributes:
        sobel_ksize (int): Kernel size for Sobel spatial-gradient filter.
        eps (float): Regularisation added to ‖∇I‖² to avoid division by zero in flat regions.
        min_mag_threshold (float): Minimum ‖∇I‖ below which normal flow is set to zero
                                (unreliable in flat regions).
        pad (bool): if true, pad start/end of array by duplicating first/last frames
                    and use 5pt gradient across all. Default is True.
        dt (float): Time step between frames for the temporal gradient.
                    Default 1.0, i.e. per-frame units.
    """

    def __init__(
            self,
            *args,
            sobel_ksize: int = 3,
            gaussian_ksize: int = 9,
            t_grad_esimator: str = "sobel",
            eps: float = 1e-6,
            min_mag_threshold: float = 1.0,
            pad: bool = True,
            dt: float = 1.0,
            **kwargs,
    ):
        """Initialise the Normal-Flow PoseOFF Sampler.

        Args:
            sobel_ksize (int): Spatial Sobel Kernel size. Default is 3.
            gaussian_ksize (int): Gaussian blur kernel size. Default is 9.
            t_grad_estimator (str): Temporal gradient calculator to used. Default is "sobel".
            eps (float): Regularisation for gradient magnitude. Default is 1e-6.
            min_mag_threshold (float): Suppress normal flow where ‖∇I‖ < this. Default 1.0.
            pad (bool): Defines whether temporal padding is used. Default is True.
            dt (float): Frame time step. Default is 1.0.
        """
        super().__init__(*args, **kwargs)
        self.sobel_ksize = sobel_ksize
        self.gaussian_ksize = gaussian_ksize
        self.temporal_estimator = _temporal_gradient_sobel if t_grad_esimator == "sobel" \
            else _temporal_gradient_5pt
        self.eps = float(eps)
        self.min_mag_threshold = min_mag_threshold
        self.pad = pad
        self.dt = dt

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, video: torch.Tensor, poses: torch.Tensor) -> np.ndarray:
        """Compute pose-guided normal-flow windows for a full video clip.

        Args:
            video (Tensor | ndarray): Shape (T, C, H, W)  [C first] or
                                      (T, H, W, C)  [C last, detected
                                      automatically].
                                      NOTE: dtype=uint8 for cv2
            poses (Tensor | ndarray): Shape (C, T, V, M)
                                      C=2 for NTU denoised data,
                                      C=3 for YOLO poses (x, y, conf).

        Returns:
            poseoff (ndarray): Same output convention as PoseOFFSampler_LK.
                NTU  → (2·W², T, V, M)
                YOLO → (pose_C + 2·W², T, V, M)  padded to original T.
        """
        # ---- 0. Convert to numpy ----------------------------------------
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()

        # ---- 1. Ensure video layout is (T, H, W, C) ---------------------
        # Heuristic: if last dim is small (≤4) it is already channel-last.
        if video.ndim == 4 and video.shape[-1] > 4:
            video = rearrange(video, 'T C H W -> T H W C')

        n_frames, height, width, n_ch = video.shape

        # ---- 2. Convert all frames to greyscale and blur float32 -----------------
        # Doing this once up-front avoids repeated per-frame conversion inside
        # the loop and lets us run the temporal gradient vectorised.
        grey = np.stack(
            [
                cv2.GaussianBlur(
                    cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32),
                    (self.gaussian_ksize, self.gaussian_ksize), 0
                )
             for f in video],
            axis=0,
        )  # (T, H, W)  float32


        # ---- 3. Precompute GLOBAL temporal gradient I_t ------------------
        # Shape: (T, H, W).  This is the only O(T·H·W) operation.
        I_t_full = self.temporal_estimator(grey, pad=self.pad, dt=self.dt)  # (T, H, W)

        # ---- 4. Precompute GLOBAL spatial gradients I_x, I_y -------------
        # Shape: (T, H, W) each.  Sobel on float32 frames.
        I_x_full, I_y_full = _spatial_gradients(grey, self.sobel_ksize)

        # ---- 5. Poses: remove first frame (no I_t at t=0 is meaningful) -
        pose_channels, num_pose_frames, num_keypoints, num_people = poses.shape
        total_keypoints = num_keypoints * num_people

        # ---- 6. Scale keypoints to pixel coordinates --------------------
        if self.ntu:
            pose_points = np.nan_to_num(poses[:2], nan=0)
            pose_points = (
                rearrange(pose_points, 'C T V M -> C T (V M)')
                * np.array([(width - 1) / 1920,
                             (height - 1) / 1080]).reshape(2, 1, 1)
            ).astype(int)
        else:
            pose_points = (
                (poses[:2, ...] + 0.5)
                .reshape(2, num_pose_frames, total_keypoints)
                * np.array([width - 1, height - 1]).reshape(2, 1, 1)
            ).astype(int)
            vis = poses[2, :, :].flatten() > self.threshold

        # ---- 7. Build validity mask -------------------------------------
        half_k_dil = self.half_k * self.dilation
        if self.ntu:
            valid_indices = (
                (pose_points[0] >= half_k_dil) &
                (pose_points[0] <  width  - half_k_dil) &
                (pose_points[1] >= half_k_dil) &
                (pose_points[1] <  height - half_k_dil)
            )  # (T, total_keypoints)
        else:
            valid_indices = (
                vis.reshape(num_pose_frames, total_keypoints) &
                (pose_points[0] >= half_k_dil) &
                (pose_points[0] <  width  - half_k_dil) &
                (pose_points[1] >= half_k_dil) &
                (pose_points[1] <  height - half_k_dil)
            )  # (T, total_keypoints)

        # ---- 8. Output buffer -------------------------------------------
        # Layout matches LK sampler: (2·W², T, total_keypoints)
        flow_windows = np.zeros(
            (self.window_size ** 2 * 2, num_pose_frames, total_keypoints),
            dtype=np.float32,
        )

        # ---- 9. Per-frame loop ------------------------------------------
        # Since I'm using central difference for temporal component,
        # gradients align with video...
        for frame_num in range(num_pose_frames):
            # Slice global gradient arrays for this frame
            It = I_t_full[frame_num]   # (H, W)
            Ix = I_x_full[frame_num]   # (H, W)
            Iy = I_y_full[frame_num]   # (H, W)

            # Gradient magnitude squared (with regularisation)
            grad_sq = Ix ** 2 + Iy ** 2   # (H, W)

            for kp_idx in range(total_keypoints):
                if not valid_indices[frame_num, kp_idx]:
                    # Invalid keypoint → leave zeros in output
                    continue

                x = pose_points[0, frame_num, kp_idx]
                y = pose_points[1, frame_num, kp_idx]

                # Build the (window_size × window_size) sampling grid
                # xs / ys are integer pixel coordinates, shape (W,) each
                xs = np.linspace(
                    x - half_k_dil, x + half_k_dil, self.window_size
                ).astype(int)
                ys = np.linspace(
                    y - half_k_dil, y + half_k_dil, self.window_size
                ).astype(int)
                # grid_x, grid_y each shape (W, W)
                grid_x, grid_y = np.meshgrid(xs, ys)

                # Look up gradient values at all grid points
                it = It[grid_y, grid_x]   # (W, W)
                ix = Ix[grid_y, grid_x]   # (W, W)
                iy = Iy[grid_y, grid_x]   # (W, W)
                gs = grad_sq[grid_y, grid_x]  # (W, W)

                # Suppress in flat regions (gradient too small — unreliable)
                flat_mask = np.sqrt(gs) < self.min_mag_threshold

                # Normal flow (equation 2):
                #   u_n = -I_t * I_x / (I_x² + I_y²)
                #   v_n = -I_t * I_y / (I_x² + I_y²)
                denom = gs + self.eps
                u_n = -it * ix / denom   # (W, W)
                v_n = -it * iy / denom   # (W, W)

                u_n[flat_mask] = 0.0
                v_n[flat_mask] = 0.0

                # Pack into output buffer:
                # first half of channel dim = u, second half = v
                # (matching the (p1-p0) convention in LK where dim 0 is x-flow)
                W2 = self.window_size ** 2
                flow_windows[:W2, frame_num, kp_idx] = u_n.ravel()
                flow_windows[W2:, frame_num, kp_idx] = v_n.ravel()

        # ---- 10. Reshape to (C·W², T, V, M) ----------------------------
        flow_windows = rearrange(
            flow_windows, 'C T (V M) -> C T V M',
            V=num_keypoints, M=num_people,
        )

        # ---- 11. Assemble output (mirrors LK sampler logic) -------------
        if self.ntu and not self.debug_vis:
            poseoff = flow_windows.reshape(flow_windows.shape[0],
                                           *poses.shape[1:])
        else:
            poseoff = np.concatenate(
                (poses,
                 flow_windows.reshape(flow_windows.shape[0], *poses.shape[1:])),
                axis=0,
            )

        # # Pad temporal dim to match original pose sequence length
        # if not self.ntu:
        #     poseoff = np.pad(
        #         poseoff, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant'
        #     )

        return poseoff

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_dilation(self, pose):
        """TODO: adaptive dilation based on apparent skeleton size."""
        pass


if __name__ == '__main__':
    # TEST WITH:
    # srun --mem=12G --gres=gpu:1 python data_gen/utils/extractors.py

    import os
    import os.path as osp
    from config.argclass import ArgClass
    import torchvision.transforms.v2 as v2
    from ultralytics import YOLO
    from data_gen.utils import LoadVideo
    import math
    import logging

    # ----------------------------
    poseoff_type = "NF" # RAFT, LK, NF
    dataset = "ucf101" # ntu, ucf101
    inference_timing = False

    # Get one representative sample (71 frames, 2 bodies)
    rgb_path = '../Datasets/NTU_RGBD/nturgb+d_rgb/' if dataset == "ntu" else "../Datasets/UCF-101/ApplyLipstick/"
    ske_name = "S001C001P001R001A058"
    rgb_name = osp.join(rgb_path, ske_name+"_rgb.avi") if dataset == "ntu" else \
        osp.join(rgb_path, "v_ApplyLipstick_g01_c03.avi")
    # ----------------------------

    os.makedirs("./logs/debug/extractors", exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f'./logs/debug/extractors/poseoff_sampler_{poseoff_type}.log',
        encoding="utf-8",
        filemode="w",
        level=logging.DEBUG
    )
    print(f"\tLogs to be written to: './logs/debug/extractors/poseoff_sampler_{poseoff_type}.log'")

    # Defining variables to overwrite, and which PoseOFF sampler to load
    logger.debug(f"PoseOFF type: {poseoff_type}")
    logger.debug(f"Dataset: {dataset}")
    logger.debug(f"Inference timing: {inference_timing}")
    overwrites = {
                    "RAFT": (
                        "PoseOFFSampler",
                        {"window_size": 5, "ntu": True if dataset == "ntu" else False}
                    ),
                    "LK": (
                        "PoseOFFSampler_LK",
                        {"window_size": 5, "dilation": 1, "lk_winSize": (15,15), "lk_maxLevel": 3, "mag_threshold": 500, "ntu": True if dataset == "ntu" else False}
                    ),
                    "NF": (
                        "PoseOFFSampler_NF",
                        {"window_size": 5, "sobel_ksize": 3, "gaussian_ksize":9, "t_grad_estimator":"sobel", "eps": 1e-6, "min_mag_threshold": 1.0, "pad": True, "dt": 1.0, "ntu": True if dataset == "ntu" else False}
                    ),
                }

    # Get the type of sampler and potential poseoff_arg_overwrites
    sampler_kind, poseoff_arg_overwrite = overwrites[poseoff_type]
    pre_flow = True if poseoff_type == "RAFT" else False # Required for RAFT PoseOFF

    # Don't tell anyone I'm importing this here...
    if pre_flow:
        from torchvision.models.optical_flow import raft_large

    # Get the argparse object
    arg = ArgClass(arg=f"./config/infogcn2/{dataset}/cnn.yaml")
    transform_arg = arg.extractor
    transform_arg['poseoff'] = transform_arg['poseoff'] | poseoff_arg_overwrite

    logger.debug("TRANSFORM ARGUMENTS:")
    for key, value in transform_arg.items():
        logger.debug(f"\t{key}: {value}")

    # Get the device
    device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

    if pre_flow:
        # Create the model, move it to device and turn to eval mode
        weights = torch.load(transform_arg['flow']['weights'], weights_only=True, map_location=device)
        model = raft_large(progress=False)
        model.load_state_dict(weights)
        model = model.eval().to(device)

    # Create the transforms required for flow estimation
    transform_flow = v2.Compose([
        LoadVideo(max_frames=300),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # map [0, 1] into [-1, 1]
    ])

    # Create the pose model
    detector = YOLO(transform_arg['pose']['weights'])
    detector.to(device)
    transform_pose = v2.Compose([
        LoadVideo(max_frames=300),
        v2.Resize(size=(384,640)), # YOLO pose has a minimum input image size
        v2.ToDtype(torch.float32),
        v2.Lambda(lambda x: x/255.0), # Normalises the image to [0,1]
    ])

    # Create the flow and pose transforms separately
    if pre_flow:
        getFlow = GetFlow(model=model, device=device, minibatch_size=transform_arg['flow']['minibatch_size'])
    transform_rgb = v2.Compose([
        LoadVideo(max_frames=300),
        v2.Resize(size=transform_arg['flow']['imsize']),
        v2.ToDtype(torch.uint8, scale=True)
    ])
    getPose = GetPoses_YOLO(detector=detector, num_joints=17)

    # Create necessary PoseOFF samplers
    getPoseOFF = locals()[sampler_kind](**transform_arg['poseoff'])


    # Pass the rgb video name through appropriate transforms!
    # NOTE: the final step here (rgb = ...) produces video appropriate for PoseOFF LK
    rgb_pose = transform_pose(rgb_name)
    rgb_flow = transform_flow(rgb_name)
    rgb = transform_rgb(rgb_name)

    # Get the necessary data to input into the PoseOFF sampler
    poses = getPose(rgb_pose)
    if pre_flow:
        flows = getFlow(rgb_flow)

    # CALCULATE POSEOFF!
    poseoff = getPoseOFF(flows, poses) if pre_flow else getPoseOFF(rgb, poses)

    # Print out the results nicely
    logger.debug(f"Sampler kind: {sampler_kind}")
    logger.debug(f"\tRGB shape: {rgb.shape}")
    if pre_flow:
        logger.debug(f"\tFlow shape: {flows.shape}")
    logger.debug(f"\tPoses shape: {poses.shape}")
    logger.debug(f"\tPoseOFF shape: {poseoff.shape}")

    if inference_timing:
        # Pad to correct length...
        T, *_ = rgb_flow.shape
        rgb_flow_padded = torch.cat(
            [rgb_flow for i in range(math.ceil(100/T))],
            axis=0
            )[:100]
        T, *_ = rgb_pose.shape
        rgb_pose_padded = torch.cat(
            [rgb_pose for i in range(math.ceil(100/T))],
            axis=0
            )[:100]

        logger.debug(f"RGB flow padded shape: {rgb_flow_padded.shape}")
        logger.debug(f"RGB pose padded shape: {rgb_pose_padded.shape}")

        # Dictionary for storing timing!
        timing_dict = {'flow': [], 'pose': []}


        for i in range(100):
            timing_dict['flow'].append(getFlow._time_inference(rgb_flow))
            timing_dict['pose'].append(getPose._time_inference(rgb_pose))
        logger.debug(timing_dict)
        # timing_dict['flow'].append(getFlow._time_inference(rgb_flow_padded))
        # timing_dict['pose'].append(getPose._time_inference(rgb_pose_padded))
        # logger.debug(timing_dict)
