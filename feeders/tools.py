"""
These are all effectively augments to be used during dataloading.
"""

import random
import numpy as np
import math

import torch
import torch.nn.functional as F


def valid_crop_resize(
    data_numpy: np.ndarray, valid_frame_num: int, p_interval: list, window_size: int
):
    """
    Perform cropping and resizing on the input data.
    This function processes a 4D numpy array representing video data with dimensions
    (C, T, V, M), where:
        - C: Number of channels
        - T: Number of frames (temporal dimension)
        - V: Number of joints (spatial dimension)
        - M: Number of bodies (e.g., multiple people)
    The function crops the temporal dimension based on a specified interval and
    resizes the data to a fixed window size.
    Args:
        data_numpy (np.array): Input data with shape (C, T, V, M).
        valid_frame_num (int): Number of valid frames in the input data.
        p_interval (list): Interval for cropping. If it contains one value,
            center cropping is performed. If it contains two values, random cropping
            is performed within the range [p_interval[0], p_interval[1]].
        window_size (int): Target size for the temporal dimension after resizing.
    Returns:
        np.array: Processed data with shape (C, window_size, V, M).
    Notes:
        - If `p_interval` contains one value, the function performs center cropping.
        - If `p_interval` contains two values, the function performs random cropping
          with constraints on the cropped length (minimum of 64 frames).
        - Resizing is performed using bilinear interpolation, which can handle both
          up-sampling and down-sampling.
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias : end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = (
            np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        )  # random float between 0.5-1
        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p)), 64), valid_size
        )  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias : begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float) # C, T, V, M
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(
        data, size=(C * V * M, window_size), mode="bilinear", align_corners=False
    ).squeeze()  # could perform both up sample and down sample
    data = (
        data.contiguous()
        .view(C, V, M, window_size)
        .permute(0, 3, 1, 2)
        .contiguous()
        .numpy()
    )

    return data



def obs_mask(data_numpy, obs_ratio: float = 1.0):
    '''Temporal masking of the sequence by duplicating last observed frame.
    Args:
        data_numpy (np.array): Input data with shape (C, T, V, M).
        obs_ratio (float): Ratio of sequence to KEEP (default: 1.0)
    Returns:
        np.array: Processed data with shape (C, T*obs_ratio, V, M).
    '''
    C, T, V, M = data_numpy.shape
    pad_frame_no = int(obs_ratio*T)-1
    data_numpy[:, pad_frame_no:, ...] = np.expand_dims(
        data_numpy[:, pad_frame_no, ...],
        axis=1
        )
    return data_numpy


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return (
        data_numpy.reshape(C, int(T / step), step, V, M)
        .transpose((0, 1, 3, 2, 4))
        .reshape(C, int(T / step), V, step * M)
    )


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_padding(data_numpy, window_size:int=64, pad_method:str='last_frame'):
    """Pads the input data to a specified window size.

    Args:
        data_numpy (numpy.ndarray): Input data array with shape (C, T, V, M).
        window_size (int, optional): The size of the window to pad to. Default is 64.
        pad_method (str, optional): Pad data_numpy time dimension using either
            ['last_frame', 'replay', 'zero_pad'](default = 'last_frame').

    Returns:
        numpy.ndarray: Padded data array with shape (C, window_size, V, M) if T < window_size,
            otherwise returns the original data array.
    """
    C, T, V, M = data_numpy.shape
    assert pad_method in ['last_frame', 'replay', 'zero_pad'], \
        "Pad method must be either 'last_frame' or 'replay'"
    if T < window_size:
        if pad_method == 'replay':
            data_numpy_paded = np.concatenate(
                [data_numpy for i in range(math.ceil(window_size/T))],
                axis=1
                )[:, :window_size]
        else:
            data_numpy_paded = np.zeros((C, window_size, V, M))
            data_numpy_paded[:, :T, ...] = data_numpy
            if pad_method == 'zero_pad':
                return data_numpy_paded
            data_numpy_paded[:, T:, ...] = np.repeat(
                np.expand_dims(data_numpy[:, -1, ...], axis=1),
                window_size-T,
                axis=1
            )
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, window_size=64, auto_pad=True):
    """
    Randomly selects a window of data from the input numpy array.

    Args:
        data_numpy (numpy.ndarray): The input data array with shape (C, T, V, M).
        window_size (int, optional): The size of the window to select. Default is 64.
        auto_pad (bool, optional): If True, pads the data if T < window_size. Default is True.

    Returns:
        numpy.ndarray: The selected window of data with shape (C, window_size, V, M).
    """
    C, T, V, M = data_numpy.shape
    if T == window_size:
        return data_numpy
    elif T < window_size:
        if auto_pad:
            return auto_padding(data_numpy, window_size)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - window_size)
        return data_numpy[:, begin : begin + window_size, :, :]


def random_move(
    data_numpy,
    angle_candidate=[-10.0, -5.0, 0.0, 5.0, 10.0],
    scale_candidate=[0.9, 1.0, 1.1],
    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
    move_time_candidate=[1],
):
    """
    Apply random transformations to the input data.
    Parameters:
    data_numpy (numpy.ndarray): Input data array with shape (C, T, V, M).
    angle_candidate (list of float): List of angle candidates for rotation in degrees.
    scale_candidate (list of float): List of scale candidates for scaling.
    transform_candidate (list of float): List of translation candidates for x and y axes.
    move_time_candidate (list of int): List of candidates for the number of times to apply the transformation.
    Returns:
    numpy.ndarray: Transformed data array with the same shape as input.
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i] : node[i + 1]] = (
            np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        )
        s[node[i] : node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i] : node[i + 1]] = np.linspace(
            T_x[i], T_x[i + 1], node[i + 1] - node[i]
        )
        t_y[node[i] : node[i + 1]] = np.linspace(
            T_y[i], T_y[i + 1], node[i + 1] - node[i]
        )

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s], [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    """
    Introduces a random temporal shift, and pads any empty frames after the shift
    with zeros.
    """
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias : bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert C == 3
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0 : T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0 : T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = rank == m
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert np.all(forward_map >= 0)

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]].transpose(
            1, 2, 0
        )
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def mirror(data_numpy, probability: float = 0.2):
    # Apply transformation based on probability
    if random.random() <= probability:
        C, T, V, M = data_numpy.shape
        W = int((C / 2) ** 0.5)

        # Flip x positions (assuming the first channel corresponds to x positions)
        data_numpy[0] *= -1

        # Flip the x-direction of flow vectors
        flow = data_numpy[3:].reshape(2, W, W, T, V, M)  # Reshape to (2, W, W, T, V, M)
        flow[0] *= -1  # Flip x-direction flow
        data_numpy[3:] = flow.reshape(
            -1, T, V, M
        )  # Reshape back to original dimensions

    return data_numpy


def average_flow(data_numpy):
    """
    Averages the flow vectors across the spatial dimensions.
    Flow channel output is 2, simpy the average x and y directions.
    """
    C, T, V, M = data_numpy.shape
    W = int(((C - 3) / 2) ** 0.5)

    # Average flow vectors
    flow = data_numpy[3:].reshape(2, W, W, T, V, M)  # Reshape to (2, W, W, T, V, M)
    flow = flow.mean(axis=(1, 2))  # Average flow vectors
    data_numpy = np.concatenate(
        [data_numpy[:3], flow.reshape(-1, T, V, M)], axis=0
    )  # Reshape back to original dimensions

    return data_numpy


def absolute_flow(data_numpy, window_mean=False):
    """
    Converts flow vectors to absolute values.
    if window_mean is True, the flow vectors are averaged across the spatial dimensions.
    """
    C, T, V, M = data_numpy.shape
    W = int(((C - 3) / 2) ** 0.5)

    # Calculate absolute flow vectors
    flow = data_numpy[3:].reshape(2, W, W, T, V, M)  # Reshape to (2, W, W, T, V, M)
    flow = np.linalg.norm(flow, axis=0)  # Calculate absolute flow vectors
    if window_mean:
        flow = flow.mean(axis=(0, 1))  # Average flow vectors across spatial dimension
    data_numpy = np.concatenate(
        [data_numpy[:3], flow.reshape(-1, T, V, M)], axis=0
    )  # Reshape back to original dimensions

    return data_numpy


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: (C + 50), T, V, M
    C: number of keypoint channels (e.g., 3 for x, y, z)
    50: number of optical flow channels (e.g., 5x5 window of u, v flow vectors flattened)
    TODO: Test that this random_rotation works for both keypoint and flow data, implement in feeder.py
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape[0] - 50, data_torch.shape[1], data_torch.shape[2], data_torch.shape[3]

    # Separate keypoint channels and flow channels
    keypoints = data_torch[:C]
    flow_vectors = data_torch[C:]

    # Create rotation matrix
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # Assuming _rot is a function that creates a rotation matrix

    # Rotate keypoints
    keypoints = keypoints.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)
    keypoints = torch.matmul(rot, keypoints)
    keypoints = keypoints.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()
    
    # Rotate flow vectors
    flow_vectors = flow_vectors.permute(1, 0, 2, 3).contiguous().view(T, 50, V*M)
    flow_vectors = torch.matmul(rot, flow_vectors)
    flow_vectors = flow_vectors.view(T, 50, V, M).permute(1, 0, 2, 3).contiguous()

    # Combine rotated keypoints and flow vectors
    data_torch = torch.cat((keypoints, flow_vectors), dim=0)

    return data_torch


if __name__ == "__main__":
    from feeders.ntu_rgb_d import Feeder
    from config.argclass import ArgClass
    import logging

    # Create logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename='./logs/debug/feeders/feeder_tools.log',
        encoding="utf-8",
        filemode="w",
        level=logging.DEBUG
    )

    # CHANGE THIS TO TEST DIFFERENT EMBEDDING CONFIGS
    model_type = "stgcn2"
    dataset = "ntu"
    evaluation = "CV"
    flow_embedding = "base"
    obs_ratio = 0.6

    arg = ArgClass(f"config/{model_type}/{dataset}/{flow_embedding}.yaml")
    arg.feeder_args['eval'] = evaluation
    arg.feeder_args['obs_ratio'] = obs_ratio
    arg.feeder_args['window_size'] = 15 # Get a window of size 15 so we can test padding

    feeder = Feeder(**arg.feeder_args, split="train")

    data, label, _,_ = feeder[30]
    C, T, V, M = data.shape
    logger.debug(f"Data shape (C, T, V, M): {data.shape}")
    logger.debug(f"Label: {label}")

    data_pad_last_frame = auto_padding(data, 100)
    data_pad_replay = auto_padding(data, 100, pad_method="replay")
    data_pad_zero_pad = auto_padding(data, 100, pad_method="zero_pad")
    data_valid_crop = valid_crop_resize(data, 29, [0.5,1], 100)

    logger.debug(f"Last frame padding shape: {data_pad_last_frame.shape}")
    logger.debug(f"Last frame padding (first frame): {data_pad_last_frame[:3, 0, 0, 0]}")
    logger.debug(f"Last frame padding (last valid frame): {data_pad_last_frame[:3, 14, 0, 0]}")
    logger.debug(f"Last frame padding (last valid frame+1): {data_pad_last_frame[:3, 15, 0, 0]}")
    logger.debug(f"Last frame padding (last frame): {data_pad_last_frame[:3, -1, 0, 0]}\n")

    logger.debug(f"Replay padding shape: {data_pad_replay.shape}")
    logger.debug(f"Replay padding (first frame): {data_pad_replay[:3, 0, 0, 0]}")
    logger.debug(f"Replay padding (last valid frame-1): {data_pad_replay[:3, 13, 0, 0]}")
    logger.debug(f"Replay padding (last valid frame): {data_pad_replay[:3, 14, 0, 0]}")
    logger.debug(f"Replay padding (last valid frame+1): {data_pad_replay[:3, 15, 0, 0]}")
    logger.debug(f"Replay padding (last frame-1): {data_pad_replay[:3, -2, 0, 0]}")
    logger.debug(f"Replay padding (last frame): {data_pad_replay[:3, -1, 0, 0]}\n")

    logger.debug(f"Zero pad padding shape: {data_pad_zero_pad.shape}")
    logger.debug(f"Zero pad padding (first frame): {data_pad_zero_pad[:3, 0, 0, 0]}")
    logger.debug(f"Zero pad padding (last valid frame-1): {data_pad_zero_pad[:3, 13, 0, 0]}")
    logger.debug(f"Zero pad padding (last valid frame): {data_pad_zero_pad[:3, 14, 0, 0]}")
    logger.debug(f"Zero pad padding (last valid frame+1): {data_pad_zero_pad[:3, 15, 0, 0]}")
    logger.debug(f"Zero pad padding (last frame-1): {data_pad_zero_pad[:3, -2, 0, 0]}")
    logger.debug(f"Zero pad padding (last frame): {data_pad_zero_pad[:3, -1, 0, 0]}")

    logger.debug(f"Valid crop resize shape: {data_valid_crop.shape}")

    # data = np.load("data/UCF-101/flowpose/Archery/v_Archery_g01_c01.npy")
    # C, T, V, M = data.shape
    # print("(Channels, Time, Joints, Bodies)")
    # print(f"Original shape: {data.shape}")

    # # Crop temporal dimension to only include valid frames
    # valid_frame = data.sum(0, keepdims=True).sum(2, keepdims=True)
    # valid_frame_num = np.sum(np.squeeze(valid_frame).sum(-1) != 0)
    # data = valid_crop_resize(data, valid_frame_num, p_interval=[0.95], window_size=64)
    # print(f"{data.shape} - valid_crop_resize")

    # for transform in [
    #     random_shift,
    #     random_choose,
    #     auto_padding,
    #     random_move,
    #     absolute_flow,
    # ]:
    #     data = transform(data)
    #     print(f"{data.shape} - {transform.__name__}")
