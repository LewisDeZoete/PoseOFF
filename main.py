import os.path as osp
import datetime
import argparse
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from model_utils import ModelLoader
from config.argclass import ArgClass

# from training.train_infogcn import train_network
# from training.eval_infogcn import eval_network
from training.loss import LabelSmoothingCrossEntropy, masked_recon_loss
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    dest="model_type",
    default="infogcn2",
    help="Base model type (e.g. infogcn2, msg3d, stgcn2)"
)
parser.add_argument(
    "-d",
    dest="dataset",
    default="ucf101",
    help="Config dictionary location (default=ucf101)",
)
parser.add_argument(
    "-p",
    dest="phase",
    default="train",
    help="Model phase (train/eval)"
)
parser.add_argument(
    "-f",
    dest="flow_embedding",
    default="base",
    help="Optical flow embedding method [base, cnn, avg, abs] (default=base)"
)
parser.add_argument(
    "-e",
    dest="evaluation",
    help="Evaluation benchmark used for specific dataset \
        (eg. 1-3 for ucf101, CV/CS for NTU_RGB+D)"
)
parser.add_argument(
    "-o",
    dest="obs_ratio",
    default="1.0",
    help="Observation ratio, used for both training and testing."
)
parser.add_argument(
    "-r",
    dest="run_name",
    default="",
    help="name to save the results dictionary as after training",
)
parser.add_argument(
    "--desc",
    dest="description",
    default="",
    help="Description of what is being tested in run",
)
parser.add_argument(
    "--data_path_overwrite", help="Overwrite dataset path.\
    This overwrites `config['data_paths'][arg.evaluation]` completely (must be a file)."
)
parser.add_argument(
    "-v", dest="verbose", action="store_true", help="Print verbose output for argparse"
)
parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Creates a logging file to debug everything"
)

parsed = parser.parse_args()

if parsed.debug:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f'logs/debug/main/{parsed.run_name}.log',
        encoding='utf-8',
        filemode='w',
        level=logging.INFO
    )

print("### Libraries loaded")
# Pass the argparse.Namespace object (parsed) to ArgClass to create an arg obj
# `with open(f'./config/{arg.model_type}/{arg.config}/{arg.flow_embedding}.yaml', 'r')...`
arg = ArgClass(arg=parsed, verbose=parsed.verbose)

# Pass root path for the dataset objects
if arg.data_path_overwrite is not None:
    arg.feeder_args['use_mmap'] = True
    arg.feeder_args['data_paths'][arg.evaluation] = arg.data_path_overwrite

if arg.obs_ratio != "1.0":
    arg.feeder_args['obs_ratio'] = float(arg.obs_ratio)

# Define checkpoint file (this is the same for train and eval)
assert arg.run_name != ""
arg.checkpoint_file = osp.join(  # results/{dataset}/{eval}/train/{run}.pt
    arg.save_location,
    arg.evaluation,
    "train",
    arg.run_name + ".pt"
)
if arg.phase == "eval":
    arg.eval_save_name = osp.join(  # results/{dataset}/{eval}/eval/{run}.pt
        arg.save_location,
        arg.evaluation,
        "eval",
        arg.run_name + ".pt"
    )

print("### Arguments loaded")

# Using the ModelLoader class to create an object to hold model related objects
# to access the actual model, we can use skel_model.model (maybe unnecessary)
modelLoader = ModelLoader(arg)
skel_model = modelLoader.model

print("### Model created")

# Get the correct device (since arg.device is simply an int, we want a torch.device)
device = torch.device(arg.device if torch.cuda.is_available() else "cpu")

# # Create the datasets and dataloaders
FeederClass = arg.import_class(arg.feeder)
if arg.phase == "train":
    train_dataset = FeederClass(
        **arg.feeder_args,
        eval=arg.evaluation,
        split="train"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

test_dataset = FeederClass(
    **arg.feeder_args,
    eval=arg.evaluation,
    split="test"
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=arg.batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
)

print("### Dataloaders created")

# Create the loss functions!
if arg.model_type == 'infogcn2':
    # Create the loss function(s)
    cls_loss = LabelSmoothingCrossEntropy(T=arg.model_args["T"])
    recon_loss = masked_recon_loss
    loss_funcs = {"cls_loss": cls_loss, "recon_loss": recon_loss}

    # Score functions (train+test), accuracy, area under curve,
    # class, feature and reconstruction losses
    score_funcs = ["ACC", "AUC", "cls_loss", "feature_loss", "recon_loss"]

else:
    # Create the loss function(s)
    cls_loss = nn.CrossEntropyLoss()
    loss_funcs = {"cls_loss": cls_loss}

    # Score functions (train+test), accuracy, cls_loss
    score_funcs = ["ACC", "cls_loss"]

# Train or eval phases
if arg.phase == "train":
    # Get the parameters to optimise
    param_groups = {"params": []}
    for name, params in modelLoader.model.named_parameters():
        param_groups["params"].append(params)
    params = list({"other": param_groups}.values())


    # Create the optimiser
    OptimiserClass = arg.import_class(arg.optimiser)
    optimiser = OptimiserClass( # optim.SGD
        params,
        **arg.optim_params,
    )

    # Create the learning rate scheduler (if cosine annealing, calculate total iterations)
    if arg.scheduler == "torch.optim.lr_scheduler.CosineAnnealingLR":
        total_scheduler_iters = math.ceil(arg.num_epoch * ((len(train_dataset)) / arg.batch_size))
        arg.scheduler_params['T_max'] = total_scheduler_iters
    SchedulerClass = arg.import_class(arg.scheduler)
    scheduler = SchedulerClass( # optim.lr_scheduler.CosineAnnealingLR/MultiStepLR
        optimiser,
        **arg.scheduler_params,
    )

    # Get the train_network function to train our network!
    train_network = arg.import_class(arg.training_function)

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


elif arg.phase == "eval":
    # Get the eval_network function to train our network!
    eval_network = arg.import_class(arg.testing_function)

    results = eval_network(
        arg=arg,
        model=skel_model,
        loss_funcs=loss_funcs,
        test_loader=test_dataloader,
        checkpoint_file=arg.checkpoint_file,
        score_funcs=score_funcs,
        device=device,
        save_attention=False,  # TODO: handle save_attention for main.py
    )
    # Unlike the train loop, the eval loop doesn't have a checkpoint save
    torch.save(results, arg.eval_save_name)
