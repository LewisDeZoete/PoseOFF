import sys
sys.path.extend(['../'])

from tqdm import tqdm
import torch
import numpy as np
import decord
from decord import VideoReader, cpu
from .rotation import angle_between, rotation_matrix


def pre_normalisation(data, zaxis=[0, 1], xaxis=[8, 4]):
    """
    Perform pre-normalization on skeleton data.
    Parameters:
    data (numpy.ndarray): The input skeleton data with shape (N, C, T, V, M), where
                          N is the number of samples,
                          C is the number of channels,
                          T is the number of frames,
                          V is the number of joints,
                          M is the number of people.
    zaxis (list): Indices of the joints to define the z-axis for rotation. Default is [0, 1].
    xaxis (list): Indices of the joints to define the x-axis for rotation. Default is [8, 4].
    Returns:
    numpy.ndarray: The normalized skeleton data with the same shape as the input.
    """
    
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


def stack_frames(frames: torch.Tensor):
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


class LoadVideo:
    """
    Load a video from the specified path and return a tensor of frames.
    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to load from the video.
    Returns:
        torch.tensor: A tensor containing the video frames in RGB format with shape (num_frames, channels, height, width).
                    The frames are permuted to have the channel dimension as the second dimension.
    """
    def __init__(self, max_frames=300):
        self.max_frames = max_frames

    def __call__(self, video_path):
        decord.bridge.set_bridge("torch")
        vr = VideoReader(video_path, ctx=cpu(0))
        # if there's too many frames, get `max_frames` linearly spaced frames
        if len(vr) > self.max_frames:
            # output = torch.tensor(vr.get_batch(np.linspace(0, len(vr)-1, self.max_frames)).asnumpy())
            video = vr.get_batch(np.linspace(0, len(vr) - 1, self.max_frames))
        else:
            # output = torch.tensor(vr.get_batch(np.linspace(0, len(vr)-1, len(vr))).asnumpy())
            video = vr.get_batch(np.linspace(0, len(vr) - 1, len(vr)))

        # RGB Colour format
        video = torch.permute(video, (0, 3, 1, 2))

        return video


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalisation(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
