import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

# Joint index:
# {0,  "Nose"}
# {1,  "Left-eye"},
# {2,  "Right-eye"},
# {3,  "Left-ear"},
# {4,  "Right-ear"},
# {5,  "Left-shoulder"},
# {6,  "Right-shoulder"},
# {7,  "Left-elbow"},
# {8,  "Right-elbow"},
# {9,  "Left-wrist"},
# {10, "Right-wrist"},
# {11, "Left-hip"},
# {12, "Right-hip"},
# {13, "Left-knee"},
# {14, "Right-knee"},
# {15, "Left-ankle"},
# {16, "Right-ankle"},

num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,11), 
                    (11,12), (6,12), (5,7), (7,9),(6,8), (8,10), 
                    (11,13), (13,15), (12,14), (14,16)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


# TODO: REMOVE (after fixing MS-G3D)
class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)

    plt.matshow(A_binary)
    plt.show()
