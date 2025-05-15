import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ModelLoader
from config.argclass import ArgClass

from training.train_infogcn import train_network
from training.loss import LabelSmoothingCrossEntropy, masked_recon_loss
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os.path as osp
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
    help="network phase [train, test] (default=train)",
)
parser.add_argument(
    "-m", 
    dest="model_type", 
    default="base", 
    help="model type [base, cnn, avg, abs] (default=base)"
)
parser.add_argument(
    "-e",
    dest="evaluation",
    help="Evaluation benchmark used for specific dataset \
        (eg. 1-3 for ucf101, CV/CS for NTU_RGB+D)"
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

print("### Libraries loaded")
# Pass the argparse.Namespace object (parsed) to ArgClass to create an arg obj
arg = ArgClass(arg=parsed, verbose=parsed.verbose)

# Define checkpoint file
if arg.run_name != "":
    arg.checkpoint_file = osp.join(arg.save_location,
                                   arg.evaluation,
                                   arg.run_name + '.pt')

print("### Arguments loaded")

# Using the ModelLoader class to create an object to hold model related objects
# to access the actual model, we can use skel_model.model (maybe unnecessary)
modelLoader = ModelLoader(arg)
skel_model = modelLoader.model


print("### Model created")

# Get the correct device (since arg.device is simply an int, we want a torch.device)
device = torch.device(arg.device if torch.cuda.is_available() else "cpu")

# # Create the datasets and dataloaders
feeder_class = arg.import_class(arg.feeder)
train_dataset = feeder_class(**arg.feeder_args, eval=arg.evaluation, split="train")
test_dataset = feeder_class(**arg.feeder_args,eval=arg.evaluation, split="test")

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
score_funcs = ["ACC", "AUC", "cls_loss", "feature_loss", "recon_loss"]


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
    verbose=arg.verbose,
)


# print(f'\tTraining time: {results['training time'][-1]/60:0.2f)} minutes')
# print(f'\tBest train accuracy: {results['train accyracy'].max()}')
# print(f'\tBest test accuracy: {results['test accyracy'].max()}')
