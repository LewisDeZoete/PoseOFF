# import torch
# from lib.utils.objects import ArgClass
# from torch import optim
# from lib.utils.objects import ArgClass
# import matplotlib.pyplot as plt

# arg = ArgClass(arg='config/custom_pose/train_joint.yaml')

# params = torch.zeros(10,10, requires_grad=True)
# optimiser = optim.SGD([params], lr=0.1, momentum=0.9)

# scheduler1 = optim.lr_scheduler.LinearLR(optimiser, start_factor=0.5, total_iters=arg.optim['step'][0])
# scheduler2 = optim.lr_scheduler.ConstantLR(optimiser, factor=1, total_iters=arg.optim['step'][1])
# scheduler3 = optim.lr_scheduler.ExponentialLR(optimiser, gamma=arg.optim['gamma'])
# scheduler = optim.lr_scheduler.SequentialLR(optimiser, schedulers=[scheduler1, scheduler2, scheduler3], milestones=arg.optim['step'])
# # scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=arg.optim['step'], gamma=arg.optim['gamma'])

# lr = {}
# for epoch in range(100):
#     lr[epoch] = scheduler.get_last_lr()
#     scheduler.step()

# # print(scheduler.state_dict())
# fig, ax = plt.subplots()
# ax.plot(lr.keys(), lr.values())
# plt.savefig('lr.png')


from torch import randn, transpose, cat
from torch.nn import LeakyReLU

in_feats = 15
w = randn((15,15)) # Weight matrix, (n_nodes)
h = randn((15, 3)) # Nodes, (n_nodes, n_features)
a = randn() # Attention weight matrix 

# horrid sorry
# nodes = {0: 1, 1: 2, 2: 3}
# for node, feature in nodes.items():
#     nodes[node] = feature*w

# for i in range(3):
#     neighbours = list(nodes.keys())
#     neighbours.remove(i)
#     # e = LeakyReLU(a*transpose([]))

# print(cat((nodes[0], nodes[1])))