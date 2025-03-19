import numpy as np
# import torch


def loop_graph(data):
    """
    Pad empty frames with previous skeleton.
    """
    C, T, V, M = data.shape
    s = np.transpose(data, (3, 1, 2, 0))  # C, T, V, M  to  M, T, V, C

    # Pad empty frames with the previous skeleton
    for i_p, person in enumerate(s):
        if person.sum() == 0:
            continue
        if person[0].sum() == 0:
            index = person.sum(-1).sum(-1) != 0
            tmp = person[index].clone()
            person *= 0
            person[: len(tmp)] = tmp
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                if person[i_f:].sum() == 0:
                    rest = len(person) - i_f
                    num = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                    s[i_p, i_f:] = pad
                    break

    data = np.transpose(s, (3, 1, 2, 0))  # M, T, V, C to C, T, V, M
    return data


def flow_mag_norm(data_numpy, flow_window=5):
    """
    Normalize the flow vectors in the given numpy array.

    Parameters:
    data_numpy (numpy.ndarray): Input data array with shape (C, T, V, M).
                                C is the number of channels.
                                T is the number of frames.
                                V is the number of joints.
                                M is the number of people.
    flow_window (int, optional): The window size for the flow calculation. Default is 5.

    Returns:
    numpy.ndarray: The input data array with normalized flow vectors.
    """
    C, T, V, M = data_numpy.shape
    flow = data_numpy[3:, ...]
    flow = np.reshape(flow, (2, flow_window, flow_window, T, V, M))  # (2,5,5,300,17,2)
    mag = np.sqrt(
        flow[0] ** 2 + flow[1] ** 2 + 1e-8
    )  # Calculate the magnitude of each vector
    norm_flow = flow / mag  # Normalise flow vectors (divide by mag)

    # Change the flow values in the numpy array to the normalised ones
    data_numpy[3:] = norm_flow.reshape(2 * flow_window**2, T, V, M)

    return data_numpy


def pose_match(data):
    """
    Matches skeletons across video using square of distance between frames.
    Only takes pose as the input data.
    """
    C, T, V, M = data.shape
    assert C == 3
    score = data[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0 : T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data[0:2, 0 : T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = np.arange(2)
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
    new_data = np.zeros(data.shape)
    for t in range(T):
        new_data[:, t, :, :] = data[:, t, :, forward_map[t]].transpose(1, 2, 0)
    data = new_data

    # score sort
    trace_score = data[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data[:, :, :, rank]

    return data_numpy


def align_skeleton(data):
    """
    Aligns the skeleton data by transforming each sample to a new coordinate system.

    Parameters:
    data (numpy.ndarray): A 5-dimensional array with shape (N, C, T, V, M) where:
        N - Number of samples
        C - Number of channels (e.g., x, y, z coordinates)
        T - Number of time steps
        V - Number of joints/vertices
        M - Number of people in the frame

    Returns:
    numpy.ndarray: A 5-dimensional array with the same shape as the input, containing the transformed skeleton data.
    """
    N, C, T, V, M = data.shape
    trans_data = np.zeros_like(data)
    for i in range(N):
        for p in range(M):
            sample = data[i][..., p]
            # if np.all((sample[:,0,:] == 0)):
            # continue
            d = sample[:, 0, 1:2]
            v1 = sample[:, 0, 1] - sample[:, 0, 0]
            if np.linalg.norm(v1) <= 0.0:
                continue
            v1 = v1 / np.linalg.norm(v1)
            v2_ = sample[:, 0, 12] - sample[:, 0, 16]
            proj_v2_v1 = np.dot(v1.T, v2_) * v1 / np.linalg.norm(v1)
            v2 = v2_ - np.squeeze(proj_v2_v1)
            v2 = v2 / (np.linalg.norm(v2))
            v3 = np.cross(v2, v1) / (np.linalg.norm(np.cross(v2, v1)))
            v1 = np.reshape(v1, (3, 1))
            v2 = np.reshape(v2, (3, 1))
            v3 = np.reshape(v3, (3, 1))

            R = np.hstack([v2, v3, v1])
            for t in range(T):
                trans_sample = (np.linalg.inv(R)) @ (sample[:, t, :])  # -d
                trans_data[i, :, t, :, p] = trans_sample
    return trans_data


def create_aligned_dataset(
    file_list=["data/ntu/NTU60_CS-pose.npz", "data/ntu/NTU60_CV-pose.npz"]
):
    """
    Create an aligned dataset from the given list of .npz files.

    This function loads the original dataset from the specified .npz files,
    processes the data to align the skeletons, and saves the aligned dataset
    back to new .npz files with '_aligned' appended to the original filenames.

    Parameters:
    file_list (list of str): List of file paths to the .npz files containing the original datasets.
                             Default is ['data/ntu/NTU60_CS.npz', 'data/ntu/NTU60_CV.npz'].
    data_type (str): Type of data to align, either 'pose' or 'flowpose'. If 'pose', perform `align_skeleton`.
                     Default is 'pose'.

    The function expects the .npz files to contain the following keys:
    - 'x_train': Training data
    - 'y_train': Training labels
    - 'x_test': Testing data
    - 'y_test': Testing labels

    The aligned datasets are saved with the following keys:
    - 'x_train': Aligned training data
    - 'y_train': Training labels (unchanged)
    - 'x_test': Aligned testing data
    - 'y_test': Testing labels (unchanged)
    """
    for file in file_list:
        org_data = np.load(file)
        splits = ["x_train", "x_test"]
        aligned_set = {}
        if 'flowpose' in file:
            for split in splits:
                data = org_data[split]
                N, T, _ = data.shape
                data = data.reshape((N, T, 2, 25, 53)).transpose(0, 4, 1, 3, 2)
                skel_data = data[:, :3, ...]
                aligned_skel_data = align_skeleton(skel_data)
                aligned_data = np.concatenate((aligned_skel_data, data[:, 3:,...]), axis=1)
                aligned_data = aligned_data.transpose(0, 2, 4, 3, 1).reshape(N, T, -1)
                aligned_set[split] = org_data[split]
        
        else:
            for split in splits:
                data = org_data[split]
                N, T, _ = data.shape
                data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
                aligned_data = align_skeleton(data)
                aligned_data = aligned_data.transpose(0, 2, 4, 3, 1).reshape(N, T, -1)
                aligned_set[split] = aligned_data
            

        np.savez(
            file.replace(".npz", "_aligned.npz"),
            x_train=aligned_set["x_train"],
            y_train=org_data["y_train"],
            x_test=aligned_set["x_test"],
            y_test=org_data["y_test"],
        )
