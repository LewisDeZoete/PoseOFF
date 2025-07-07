import os.path as osp
import datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ModelLoader
from config.argclass import ArgClass

from training.train_infogcn import train_network
from training.loss import LabelSmoothingCrossEntropy, masked_recon_loss
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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
    "--data_path_overwrite", help="Overwrite dataset path"
)
parser.add_argument(
    "-v", dest="verbose", action="store_true", help="Print verbose output for argparse"
)

parsed = parser.parse_args()

print("### Libraries loaded")
# Pass the argparse.Namespace object (parsed) to ArgClass to create an arg obj
# `with open(f'./config/{arg.config}/{arg.phase}_{arg.model_type}.yaml', 'r')...`
arg = ArgClass(arg=parsed, verbose=parsed.verbose)

# Pass root path for the dataset objects
if arg.data_path_overwrite is not None:
    arg.feeder_args['use_mmap']=True
    for arg_key, arg_val in arg.feeder_args['data_paths'].items():
        arg.feeder_args['data_paths'][arg_key] = osp.join(arg.data_path_overwrite,
                                                          arg_val.split('/')[-1])

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


test_dataset = feeder_class(**arg.feeder_args, eval=arg.evaluation, split="test")

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

print("### Dataloaders created")

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

# MultiStep Learning Rate scheduler (default steps at 50, 60 epochs)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimiser, milestones=arg.optim["step"], gamma=arg.optim["gamma"]
)

# Create the loss function(s)
cls_loss = LabelSmoothingCrossEntropy(T=arg.model_args["T"])
recon_loss = masked_recon_loss
loss_funcs = {"cls_loss": cls_loss, "recon_loss": recon_loss}


# Score functions (train+test), accuracy, area under curve, 
# class, feature and reconstruction losses
score_funcs = ["ACC", "AUC", "cls_loss", "feature_loss", "recon_loss"]

# Simultaneous training of the network
results = train_network(
    arg=arg,
    model=skel_model,
    loss_funcs=loss_funcs,
    train_loader=train_dataloader,
    test_loader=test_dataloader,
    score_funcs=score_funcs,
    device=device,
    epochs=arg.num_epoch,
    scheduler=scheduler,
    optimiser=optimiser,
    checkpoint_file=arg.checkpoint_file,
    checkpoint_freq=arg.checkpoint_freq,
    verbose=arg.verbose,
)

print(f"\tTraining time: \
{datetime.timedelta(seconds=int(sum(results['train_time']+results['test_time'])))} minutes")
print(f"\tBest train AUC: {torch.tensor(results['train_AUC']).max().item()}")
print(f"\tBest test AUC: {torch.tensor(results['test_AUC']).max().item()}")
