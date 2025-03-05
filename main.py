import torch

# import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ModelLoader
from feeders import feeder
from config.argclass import ArgClass

from training.train_infogcn import train_network
from loss import LabelSmoothingCrossEntropy, masked_recon_loss
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    dest="config",
    default="ucf101",
    help="config dictionary location (default=ucf101)",
)
parser.add_argument(
    "-p",
    dest="phase",
    default="train",
    help="network phase [train, test] (default=test)",
)
parser.add_argument(
    "-m", dest="model_type", default="base", help="model type [base, cnn, avg, abs] (default=base)"
)
parser.add_argument(
    "-r",
    dest="run_name",
    default="",
    help="name to save the results dictionary as after training",
)
parser.add_argument(
    "-d",
    dest="description",
    default="",
    help="Description of what is being tested in run",
)
parser.add_argument(
    "-v", dest="verbose", action="store_true", help="Print verbose output for argparse"
)
parsed = parser.parse_args()

print(parsed.description)
print("### Libraries loaded")
# Pass the argparse.Namespace object (parsed) to ArgClass to create an arg obj
arg = ArgClass(arg=parsed, verbose=parsed.verbose)

# Define checkpoint file
if arg.run_name != "":
    arg.checkpoint_file = f"{arg.save_location}{arg.run_name}.pt"

print("### Arguments loaded")

# Using the ModelLoader class to create an object to hold model related objects
# to access the actual model, we can use skel_model.model (maybe unnecessary)
modelLoader = ModelLoader(arg)
skel_model = modelLoader.model


print("### Model created")

# Get the correct device (since arg.device is simply an int, we want a torch.device)
device = torch.device(arg.device if torch.cuda.is_available() else "cpu")

# # Create the datasets and dataloaders
train_dataset = feeder.Feeder(**arg.feeder_args, split="train")
test_dataset = feeder.Feeder(**arg.feeder_args, split="test")
generator = torch.Generator().manual_seed(
    42
)  # It shouldn't be random when you resume training
train_idx, test_idx = torch.utils.data.random_split(
    range(len(train_dataset)), [0.8, 0.2], generator=generator
)
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
test_dataset = torch.utils.data.Subset(test_dataset, test_idx)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=arg.batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=arg.batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
)

# Get the parameters to optimise
param_groups = {"params": []}
for name, params in modelLoader.model.named_parameters():
    param_groups["params"].append(params)
params = list({"other": param_groups}.values())

# Create the optimiser
optimiser = optim.SGD(
    params,
    lr=arg.optim["base_lr"],
    momentum=0.9,
    nesterov=arg.optim["nesterov"],
    weight_decay=arg.optim["weight_decay"],
)

# scheduler1 = optim.lr_scheduler.LinearLR(
#     optimiser, start_factor=0.5, total_iters=arg.optim["step"][0]
# )
# scheduler2 = optim.lr_scheduler.ConstantLR(
#     optimiser, factor=1, total_iters=arg.optim["step"][1]
# )
# scheduler3 = optim.lr_scheduler.ExponentialLR(optimiser, gamma=arg.optim["gamma"])
# scheduler = optim.lr_scheduler.SequentialLR(
#     optimiser,
#     schedulers=[scheduler1, scheduler2, scheduler3],
#     milestones=arg.optim["step"],
# )
scheduler = optim.lr_scheduler.MultiStepLR(
    optimiser, milestones=arg.optim["step"], gamma=arg.optim["gamma"]
)

# Create the loss function(s)
cls_loss = LabelSmoothingCrossEntropy(T=arg.model_args["T"])
recon_loss = masked_recon_loss
loss_funcs = {"cls_loss": cls_loss, "recon_loss": recon_loss}


# TRAINING!
# score_funcs = {'accuracy': accuracy_score,
#                'confusion matrix': confusion_matrix,
#                'classification report': classification_report}
score_funcs = ["AUC", "cls_loss", "feature_loss", "recon_loss"]

results = train_network(
    arg=arg,
    model=skel_model,
    loss_funcs=loss_funcs,
    train_loader=train_dataloader,
    test_loader=test_dataloader,
    score_funcs=score_funcs,  # TODO: Implement score functions for infogcn
    device=device,
    epochs=arg.num_epoch,
    scheduler=scheduler,
    optimiser=optimiser,
    checkpoint_file=arg.checkpoint_file,
    checkpoint_freq=arg.checkpoint_freq,
)


# print(f'\tTraining time: {results['training time'][-1]/60:0.2f)} minutes')
# print(f'\tBest train accuracy: {results['train accyracy'].max()}')
# print(f'\tBest test accuracy: {results['test accyracy'].max()}')
