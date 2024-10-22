import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from collections import defaultdict
# from torchvision.transforms import v2
from model import load_model
from lib.data.dataset import CustomVideoDataset
from lib.utils.objects import ArgClass
# from lib.utils.transforms import GetPoses_YOLO
# from ultralytics import YOLO
from lib.training import train_simple_network
from sklearn.metrics import accuracy_score

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='config', default='custom_pose',
                    help='config dictionary location (default=custom_pose)')
parser.add_argument('-p', dest='phase', default='test',
                    help='network phase [train, test] (default=test)')
parser.add_argument('-l', dest='limb', default='joint',
                    help='limb [joint, bone] (default=joint)')
parser.add_argument('-r', dest='run_name', default='',
                    help='name to save the results dictionary as after training')
parsed = parser.parse_args()


print("### Libraries loaded")
# pass the argparse.Namespace object (parsed) to ArgClass to create an arg obj
arg = ArgClass(arg=parsed)

# Get the annotation file
classes = arg.get_classes()

print("### Arguments loaded")

# Using the load_model class to create an object to hold model related objects
# to access the actual model, we can use skel_model.model (maybe unnecessary)
skel_model = load_model(arg)
skel_model.load_model()

print("### Model created")

# Create the dataset and dataloader
train_dataset, test_dataset = torch.utils.data.random_split(CustomVideoDataset(arg), [0.8,0.2])
train_dataloader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=True)

# Get the parameters to optimise
param_groups = {'params': []}
for name, params in skel_model.model.named_parameters():
    param_groups['params'].append(params)
params = list({'other': param_groups}.values())

# Create the optimiser
optimiser = optim.SGD(
                params,
                lr=arg.base_lr,
                momentum=0.9,
                nesterov=arg.nesterov,
                weight_decay=arg.weight_decay)
# lr_scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=arg.step, gamma=0.1)
scheduler1 = optim.lr_scheduler.LinearLR(optimiser, start_factor=0.5, total_iters=10)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.93)
scheduler = optim.lr_scheduler.SequentialLR(optimiser, schedulers=[scheduler1, scheduler2], milestones=[10])
for scheduler_part in scheduler._schedulers:
    print(scheduler_part.__dict__)

loss = nn.CrossEntropyLoss()


# TRAINING! 
score_funcs = {'accuracy': accuracy_score}

results = train_simple_network(model=skel_model.model, loss_func=loss, train_loader=train_dataloader, test_loader=test_dataloader,
                                score_funcs=score_funcs, device=arg.device, epochs=100, 
                                scheduler=scheduler, optimiser=optimiser)

print('### Results')
print(results)