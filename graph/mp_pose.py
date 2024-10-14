import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools


num_node = 33
self_link = [(i, i) for i in range(num_node)]
inward = [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (10,9),
          (11,12), (12,24), (23,24), (11,23), (11,13), (13,15), (15,21), (15,17), (15,19), (17,19),
          (12,14), (14,16), (16,22), (16,18), (16,20), (18,20), (23,25), (25,27), (27,29), (27,31), (29,31),
          (24,26), (26,28), (28,30), (28,32), (30,32)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        # Do I need to normalise the adjacency matrix for MP_Pose?
        # self.A = tools.normalize_adjacency_matrix(self.A_binary)

if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()


'''
Going from mpPose to coco

convert_dict = 
{
    0:(0, 'Nose'),
    # 11/12:(1, 'neck')
    12:(2,'RShoulder'),
    14:(3,'RElbow'),
    16:(4,'RWrist'),
    11:(5,'LSHoulder'),
    13:(6,'LElbow'),
    15:(7,'LWrist'),
    24:(8,'RHip'),
    26:(9,'RKnee'),
    28:(10,'RAnkle'),
    23:(11,'LHip'),
    25:(12,'LKnee'),
    27:(13,'LAnkle'),
    5:(14,'REye'),
    2:(15,'LEye'),
    8:(16,'REar'),
    7:(17,'LEar')
}
'''