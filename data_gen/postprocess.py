import numpy as np

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