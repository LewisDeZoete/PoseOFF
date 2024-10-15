# from lib.utils.objects import ArgClass
# from model import load_model
# import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('-c', dest='config', default='custom_pose',
#                     help='config dictionary location (default=custom_pose)')
# parser.add_argument('-p', dest='phase', default='test',
#                     help='network phase [train, test] (default=test)')
# parser.add_argument('-l', dest='limb', default='joint',
#                     help='limb [joint, bone] (default=joint)')
# parser.add_argument('-s', dest='save_name', default='',
#                     help='name to save the results dictionary as after training')
# parsed = parser.parse_args()


# print("### Libraries loaded")
# # pass the argparse.Namespace object (parsed) to ArgClass to create an arg obj
# arg = ArgClass(arg=parsed)

# skel_model = load_model(arg)

import torch
from torch import optim

params = [torch.randn(2,2, requires_grad=True),
          torch.randn(3,1, requires_grad=True)]

# Create the optimiser
optimiser = optim.SGD(
                params,
                lr=0.5,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.0003)
# lr_scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=arg.step, gamma=0.1)
scheduler1 = optim.lr_scheduler.LinearLR(optimiser, start_factor=0.5, total_iters=10)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.93)
scheduler = optim.lr_scheduler.SequentialLR(optimiser, schedulers=[scheduler1, scheduler2], milestones=[10])


print(f'\tMilestones: {scheduler._milestones}')

for scheduler_part in scheduler._schedulers:
    if str(scheduler_part.__class__).split('.')[-1][:-2] == 'LinearLR':
        print('\tLinearLR:')
        print(f'\t\tStart: {scheduler_part.start_factor}')
        print(f'\t\tEnd: {scheduler_part.end_factor}')
    if str(scheduler_part.__class__).split('.')[-1][:-2] == 'ExponentialLR':
        print('\tExponentialLR')
        print(f'\t\tGamma: {scheduler_part.gamma}')