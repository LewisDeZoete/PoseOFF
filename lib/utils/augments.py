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
    Randomly move skeleton keypoints a small amount.
    NOTE: changed default value of `transform_candidate` 
        Previously, this was [-0.2, -0.1, 0.0, 0.1, 0.2], but given I'm passing
        (x,y) keypoint values that are normalised between -0.5 and 0.5, 
        this might be too wide of a range.
    '''
    def __init__(self,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.1, -0.05, 0.0, 0.05, 0.1],
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
    '''
    Introduces a random temporal shift, and pads any empty frames after the shift
    with zeros.
    '''
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
    def __init__(self, flow_window=5):
        self.flow_window = flow_window
    def __call__(self, data_numpy):
        C,T,V,M = data_numpy.shape
        flow = data_numpy[3:,...]
        flow = np.reshape(flow, (2, self.flow_window, self.flow_window, T,V,M)) # (2,5,5,300,17,2)
        mag = np.sqrt(flow[0]**2 + flow[1]**2+1e-8) # Calculate the magnitude of each vector
        norm_flow = flow / mag # Normalise flow vectors (divide by mag)

        # Change the flow values in the numpy array to the normalised ones
        data_numpy[3:] = norm_flow.reshape(2*self.flow_window**2,T,V,M)

        return data_numpy


class loop_graph(object):
    '''
    Pad empty frames with previous skeleton.
    NOTE: TORCH VERSION
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


class mirror(object):
    '''
    Mirror augmentation for data (mirror the x positions and u vectors).
    NOTE: assuming flow windows are square...
    '''
    def __init__(self, probability: float=0.2):
        self.probability = probability
    def __call__(self, data):
        if random.random() <= self.probability+10:
            C,T,V,M = data.shape
            W = int((C/2)**(0.5))

            # Flip x positons
            data[0] = data[0]*(-1)

            # Flip the x-direction of flow vectors
            flow = data[3:]
            flow = np.reshape(flow, (2, W, W, T, V, M)) # (2,5,5,300,17,2)
            flow[0] = flow[0]*(-1)

            # Change the flow values in the numpy array to the normalised ones
            data[3:] = np.reshape(flow, (2*W**2,T,V,M))

            return data

        else:
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
    flow_window = 5
    start = time.time()
    transforms = [swap_numpy(device='cpu'),
                  flow_mag_norm(5),
                  random_shift(),
                  mirror(),
                  random_choose(300),
                  random_move(),
                  swap_numpy(device='cpu')]
    # transforms = [pose_match()]
    
    for i in range(1000):
        data = torch.rand((3+2*flow_window**2,150,17,2))
        for transform in transforms:
            data = transform(data)

    print(data.shape)
    print(time.time()-start)