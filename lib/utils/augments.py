import random

import torch
import numpy as np


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    '''
    Subtract mean from all non-zero values of skeleton graph.
'''
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


class random_choose(object):
    '''
    If sequence is longer than `size`, implement padding (default `random_pad=True`)
    If sequence is shorter than `size`, implement random temporal crop.
    If data is preprocessed (using `loop_graph`), this transform is superfluous.
    '''
    def __init__(self, size, auto_pad=True):
        self.size=size
        self.auto_pad=auto_pad
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        if T == self.size:
            return data_numpy
        elif T < self.size:
            if self.auto_pad:
                return auto_pading(data_numpy, self.size, random_pad=True)
            else:
                return data_numpy
        else:
            begin = random.randint(0, T - self.size)
            return data_numpy[:, begin:begin + self.size, :, :]


class random_move(object):
    '''
    TODO: Comprehensive docstring
    TODO: Ensure only the first 2 channels get randomly moved
    Randomly move skeleton keypoints a small amount.
    '''
    def __init__(self,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
        self.angle_candidate=angle_candidate
        self.scale_candidate=scale_candidate
        self.transform_candidate=transform_candidate
        self.move_time_candidate=move_time_candidate

    def __call__(self, data_numpy):
        # input: C,T,V,M
        C, T, V, M = data_numpy.shape
        move_time = random.choice(self.move_time_candidate)
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(self.angle_candidate, num_node)
        S = np.random.choice(self.scale_candidate, num_node)
        T_x = np.random.choice(self.transform_candidate, num_node)
        T_y = np.random.choice(self.transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                        [np.sin(a) * s, np.cos(a) * s]])

        # perform transformation
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

        return data_numpy


class random_shift(object):
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        data_shift = np.zeros(data_numpy.shape)
        valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
        begin = valid_frame.argmax()
        end = len(valid_frame) - valid_frame[::-1].argmax()

        size = end - begin
        bias = random.randint(0, T - size)
        data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

        return data_shift
    
class flow_mag_norm(object):
    '''
    Normalise optical flow vectors to unit vectors by dividing by their magnitude.
    TODO: double check that the input data is flow_pose!
    '''
    def __call__(self, data_numpy):
        C,T,V,M = data_numpy.shape
        flow = data_numpy[3:,...]
        flow = flow.reshape(2, np.sqrt(C/2), np.sqrt(C/2), T,V,M) # (2,5,5,300,17,2)
        mag = np.sqrt(flow[0]**2 + flow[1]**2+1e-8)
        norm_flow = flow / mag
        return(norm_flow)


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


class loop_graph(object):
    '''
    Pad empty frames with previous skeleton.
    '''
    def __init__(self, zaxis=[0, 1], xaxis=[8, 4]):
        self.zaxis=zaxis
        self.xaxis=xaxis
    
    def __call__(self, data):
        C, T, V, M = data.shape
        s = data.permute(3, 1, 2, 0)  # C, T, V, M  to  M, T, V, C

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
                        pad = torch.cat([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_p, i_f:] = pad
                        break
    
        data = s.permute(3, 1, 2, 0) # M, T, V, C to C, T, V, M
        return data


class swap_numpy(object):
    '''
    Swap between numpy array and torch tensor. Assumes torch tensor is on CPU.
    '''
    def __init__(self, device='cpu'):
        self.device=device
    def __call__(self, data):
        # If it's a numpy array, convert it to a torch tensor
        if isinstance(data, np.ndarray):
            data = torch.tensor(data).to(self.device).float()
        # else, it it's a tensor, convert it to numpy!
        elif isinstance(data, torch.Tensor):
            data = data.cpu() # Make sure it's on cpu!!
            data = np.array(data)
        return data


if __name__ == "__main__":
    import time
    import torch
    start = time.time()
    transforms = [swap_numpy(device='cpu'),
                  random_shift(),
                  random_choose(300),
                  random_move(),
                  swap_numpy(device='cpu')]
    
    for i in range(1000):
        data = torch.rand((53,150,17,2))
        for transform in transforms:
            data = transform(data)
    print(data.shape)
    print(time.time()-start)