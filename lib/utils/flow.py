import torch


def stack_frames(frames: torch.Tensor):
    """
    TODO: DELETE -> moved to data_gen/preprocess.py
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