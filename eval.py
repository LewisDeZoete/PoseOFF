import torch
from torch.utils.data import DataLoader

from model import ModelLoader
from config.argclass import ArgClass

from training.train_infogcn import eval_network
from training.loss import LabelSmoothingCrossEntropy, masked_recon_loss

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
test_dataset = feeder_class(**arg.feeder_args, eval=arg.evaluation, split="test")

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

# Create the loss function(s)
cls_loss = LabelSmoothingCrossEntropy(T=arg.model_args["T"])
recon_loss = masked_recon_loss
loss_funcs = {"cls_loss": cls_loss, "recon_loss": recon_loss}


# Define score functions
score_funcs = ["ACC", "AUC",
               "AUC_avg_meter", # TODO Delete this! and in training/train_infogcn.py
               "cls_loss", "feature_loss", "recon_loss"]


results = eval_network(
    arg=arg,
    model=skel_model,
    loss_funcs=loss_funcs,
    test_loader=test_dataloader,
    checkpoint_file=arg.checkpoint_file,
    score_funcs=score_funcs,  # TODO: Implement score functions for infogcn
    device=device,
    verbose=arg.verbose,
)

torch.save(results, 'TMP.pt')
# print(f'\tTraining time: {results['training time'][-1]/60:0.2f)} minutes')
# print(f'\tBest train accuracy: {results['train accyracy'].max()}')
# print(f'\tBest test accuracy: {results['test accyracy'].max()}')
