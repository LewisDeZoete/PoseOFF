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
inward = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,11), (11,12), (6,12), (5,7), (7,9),
          (6,8), (8,10), (11,13), (13,15), (12,14), (14,16)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
