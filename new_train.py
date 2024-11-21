import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ModelLoader
from lib.data.dataset import SingleStreamDataset
from lib.utils.objects import ArgClass
from lib.utils.augments import swap_numpy, random_shift, random_choose, random_move

from lib.training import train_simple_network
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
parser.add_argument('-d', dest='description', default='',
                    help='Description of what is being tested in run') # TODO: DEBUG REMOVE
parsed = parser.parse_args()

print(parsed.description) # TODO: DEBUG REMOVE
print("### Libraries loaded")
# pass the argparse.Namespace object (parsed) to ArgClass to create an arg obj
arg = ArgClass(arg=parsed)

# Get the annotation file
classes = arg.classes

print("### Arguments loaded")

# Using the load_model class to create an object to hold model related objects
# to access the actual model, we can use skel_model.model (maybe unnecessary)
modelLoader = ModelLoader(arg)
modelLoader.load_model()
skel_model = modelLoader.model

print("### Model created")

# Get the correct device (since arg.device is simply an int, we want a torch.device)
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')
# loader_device = torch.device('cpu')

# Create the transforms (for multistream, I just need the FlowPoseSampler)
transforms = [swap_numpy(device=device),
              random_shift(),
              random_move(),
              swap_numpy(device=device)]

train_dataset = SingleStreamDataset(arg, stream='flowpose',transforms=transforms)
test_dataset = SingleStreamDataset(arg, stream='flowpose')

# Create the dataset and dataloader
train_idx, test_idx = torch.utils.data.random_split(range(len(train_dataset)),[0.8,0.2])
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
train_dataloader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=True)

# Get the parameters to optimise
param_groups = {'params': []}
for name, params in modelLoader.model.named_parameters():
    param_groups['params'].append(params)
params = list({'other': param_groups}.values())

# Create the optimiser
optimiser = optim.SGD(
                params,
                lr=arg.optim['base_lr'],
                momentum=0.9,
                nesterov=arg.optim['nesterov'],
                weight_decay=arg.optim['weight_decay'])

scheduler1 = optim.lr_scheduler.LinearLR(optimiser, start_factor=0.5, total_iters=arg.optim['step'])
scheduler2 = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.93)
scheduler = optim.lr_scheduler.SequentialLR(optimiser, schedulers=[scheduler1, scheduler2], milestones=[10])

loss = nn.CrossEntropyLoss()

# TRAINING! 
score_funcs = {'accuracy': accuracy_score, 
               'confusion matrix': confusion_matrix,
               'classification report': classification_report}

results = train_simple_network(model=skel_model, loss_func=loss, train_loader=train_dataloader, test_loader=test_dataloader,
                                score_funcs=score_funcs, device=arg.device, epochs=100, 
                                scheduler=scheduler, optimiser=optimiser, 
                                checkpoint_file=f'{arg.save_location}{arg.run_name}.pt')

print('### Results')
print(results)