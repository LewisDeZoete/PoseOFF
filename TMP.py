import numpy as np
import os.path as osp
from torch.utils.data import DataLoader
import torch
from config.argclass import ArgClass
from model import ModelLoader


root_path = './data/ntu/aligned_data'
dataset = 'MINI_CS_flowpose.npz'
arg = ArgClass('config/nturgbd/train_cnn.yaml')

# Define checkpoint file
arg.evaluation = 'CS'
arg.run_name = 'nturgbd_CS_cnn'
arg.checkpoint_file = osp.join(arg.save_location,
                                   arg.evaluation,
                                   arg.run_name + '.pt')

# Load the model with pretrained weights
modelLoader = ModelLoader(arg)
skel_model = modelLoader.model

# Create the feeder (using the mini dataset...)
feeder_class = arg.import_class(arg.feeder)
arg.feeder_args['data_paths']['CS'] = osp.join(root_path, dataset)
test_dataset = feeder_class(**arg.feeder_args, eval=arg.evaluation, split="test")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=arg.batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
)

for x, y, mask, index in test_dataloader:
    y_hat, x_hat, z_0, z_hat, kl_div = skel_model(x)
torch.save(y_hat, 'y_hat_tmp.pt')
torch.save(y, 'y_tmp.pt')

y_hat = torch.load('y_hat_tmp.pt')
y = torch.load('y_tmp.pt')

_, predicted_label = torch.max(y_hat, 1)
y_expanded = y.unsqueeze(1).expand_as(predicted_label)

# Compare predictions to ground truth for each frame
correct = (predicted_label == y_expanded) # Shape: (120, 64), bool

# Compute mean accuracy for each frame
per_frame_accuracy = correct.float().mean(dim=0)  # shape: (64,)

# If you want percentages:
per_frame_accuracy_percent = per_frame_accuracy * 100  # shape: (64,)

print(per_frame_accuracy_percent)


