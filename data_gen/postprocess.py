import numpy as np
import torch

def loop_graph(data):
        '''
        Pad empty frames with previous skeleton.
        '''
        C, T, V, M = data.shape
        s = np.transpose(data, (3, 1, 2, 0))  # C, T, V, M  to  M, T, V, C

        # Pad empty frames with the previous skeleton
        for i_p, person in enumerate(s):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].clone()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_p, i_f:] = pad
                        break
    
        data = np.transpose(s, (3, 1, 2, 0)) # M, T, V, C to C, T, V, M
        return data


def flow_mag_norm(data_numpy, flow_window=5):
    C,T,V,M = data_numpy.shape
    flow = data_numpy[3:,...]
    flow = np.reshape(flow, (2, flow_window, flow_window, T,V,M)) # (2,5,5,300,17,2)
    mag = np.sqrt(flow[0]**2 + flow[1]**2+1e-8) # Calculate the magnitude of each vector
    norm_flow = flow / mag # Normalise flow vectors (divide by mag)

    # Change the flow values in the numpy array to the normalised ones
    data_numpy[3:] = norm_flow.reshape(2*flow_window**2,T,V,M)

    return data_numpy


def pose_match(data):
    '''
    Pytorch version of the pose_match augmentation.
    Matches skeletons across video using square of distance between frames.
    Only takes pose as the input data.
    '''
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