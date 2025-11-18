#!/usr/bin/env python3


import torch
import torch.nn as nn
import math
import numpy as np
import time
import datetime
import os
import os.path as osp
from einops import rearrange, repeat
from tqdm import tqdm
from training.loss import AverageMeter

import logging

logger = logging.getLogger(__name__)

def run_epoch(
        arg,
        model,
        loss_funcs,
        data_loader,
        device,
        optimiser,
        results,
        scheduler=None,
        prefix="",
):
    """
    loggers contain the following:
        'cls_loss': AverageMeter object
    """
    if prefix == "train":
        model.train()
    else:
        model.eval()

    # AverageMeter objects track values (n_values, mean, etc)
    log_acc = AverageMeter() # accuracy
    log_cls_loss = AverageMeter()  # class loss

    start = time.time()
    for x, y, mask, index in data_loader:
        loss = torch.tensor(0.0)

        B, C, T, V, M = x.shape
        x = x.float().to(device) # (B, C, T, V, M)
        y = y.long().to(device)
        y_hat = model(x)

        # Calculate the loss and correct predictions
        loss = loss_funcs["cls_loss"](y_hat, y)
        correct = (torch.argmax(y_hat, dim=1) == y)

        if prefix == "train":
            # Backward
            optimiser.zero_grad()
            if arg.half:  # mixed precision -> formerly using apex amp
                with torch.autocast(device_type="cuda") as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Prevent gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            if scheduler is not None:
                scheduler.step()

        # Update loggers
        log_acc.update(correct.float().mean(), B)
        log_cls_loss.update(loss, B)

    results[prefix + "_" + "ACC"].append(log_acc.avg)
    results[prefix + "_" + "cls_loss"].append(log_cls_loss.avg)

    end = time.time()
    return end - start  # time spent on epoch

def train_network(
    arg,
    model,
    loss_funcs,
    train_loader,
    test_loader=None,
    score_funcs=None,
    device="cpu",
    epochs: int = 70,
    scheduler=None,
    optimiser=None,
    checkpoint_file: str = None,
    checkpoint_freq: int = 10,
    verbose: bool = False,
):
    """
    Neural network training and testing loop.

    Arguments:
        model: the PyTorch model / "Module" to train
        loss_funcs: the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
        train_loader: PyTorch DataLoader object that returns tuples of (input, label) pairs.
        test_loader: Optional PyTorch DataLoader to evaluate on after every epoch
        score_funcs: A list of strings of what score functions are tracked performance of the model
        epochs: the number of training epochs to perform
        device: the compute lodation to perform training

    """
    to_track = ["epoch", "train_time", "train_loss", "lr"]
    if test_loader is not None:
        to_track.append("test_time")
        to_track.append("test_loss")
    if score_funcs is not None:
        for eval_score in score_funcs:
            to_track.append("train_" + eval_score)
            if test_loader is not None:
                to_track.append("test_" + eval_score)

    results = {}
    print("\tTracking:")
    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []
        print(f"\t\t{item}")

    # Place the model on the correct compute resource (CPU or GPU)
    model.to(device)

    # If we pass checkpoint_file, make sure it's initialised
    if checkpoint_file is not None:
        checkpoint = load_checkpoint(checkpoint_file, device, verbose)
        start_epoch = (
            checkpoint["epoch"] + 1
        )  # We saved the checkpoint at the end of the epoch
        try:
            results = checkpoint[
                "results"
            ]  # Don't override the results from previous training!
            optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except KeyError:
            pass  # Only just created the checkpoint file
        del checkpoint  # might save us from OOM issues

    # Training loop
    # we start from the next epoch after the last checkpoint (or from 1) and go to the specified number of epochs
    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epoch"):
        model = model.train()  # Put our model in training mode

        # Run the training epoch
        train_time = run_epoch(
            arg=arg,
            model=model,
            loss_funcs=loss_funcs,
            data_loader=train_loader,
            device=device,
            optimiser=optimiser,
            results=results,
            scheduler=scheduler,
            prefix="train",
        )

        # Append the post-training results to the results dictionary
        results["epoch"].append(epoch)
        results["train_time"].append(train_time)

        # Step the scheduler after each training epoch and append lr
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(results["val loss"][-1])
            else:
                scheduler.step()
            results["lr"].append(scheduler.get_last_lr()[0])

        # TEST
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                test_time = run_epoch(
                    arg=arg,
                    model=model,
                    loss_funcs=loss_funcs,
                    data_loader=train_loader,
                    device=device,
                    optimiser=optimiser,
                    results=results,
                    scheduler=None,
                    prefix="test",
                )

            results["test_time"].append(test_time)
            # Print the results
            print(f"\t\t{epoch} EPOCH BEST TEST ACC: {max(results['test_ACC'])}")
            print(f"\t\t\tTrain time: {train_time:.2f} seconds")
            print(f"\t\t\tTest time: {test_time:.2f} seconds")

        if checkpoint_file is not None:
            if epoch % checkpoint_freq == 0:
                save_checkpoint(
                    checkpoint_file, epoch, model, optimiser, scheduler, results, device
                )

    print(f"\tBest train ACC: {torch.tensor(results['train_ACC']).max().item()}")
    # Get the total training time
    total_time = results['train_time']
    if test_loader is not None:
        total_time += results['test_time']
        print(f"\tBest test ACC: {torch.tensor(results['test_ACC']).max().item()}")
    print(f"\tTraining time: \
        {datetime.timedelta(seconds=int(sum(total_time)))}")
    return results


def load_checkpoint(checkpoint_file: str, device, verbose: bool=False) -> dict:
    """
    Loads a checkpoint from a specified file.

    Args:
        checkpoint_file (str): The path to the checkpoint file.
        device: The device on which to load the checkpoint (e.g., 'cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint containing at least the 'epoch' key.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist, a new checkpoint is created with 'epoch' set to 0.
    """
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        print(f"\tResuming from checkpoint at epoch {checkpoint['epoch']}")

        # Print previous results, it's easier for comparisons
        results = checkpoint['results']
        if verbose:
            for i in range(len(results['epoch'])):
                print(f"\t\t{i+1} EPOCH BEST TEST ACC: {results['test_ACC'][i]}")
                print(f"\t\t\tTrain time: {results['train_time'][i]:.2f} seconds")
                print(f"\t\t\tTest time: {results['test_time'][i]:.2f} seconds")
    except FileNotFoundError:
        os.makedirs( # Create the folders in the case that they don't exist...
            osp.dirname(checkpoint_file),
            exist_ok=True
        )
        print(f"\tCreated new checkpoint file: {checkpoint_file}")
        checkpoint = {
            "epoch": 0
        }  # Start from scratch, indexing starts from 1 in this case
        torch.save(checkpoint, checkpoint_file)
    except KeyError as e:
        print(f'Key {e} did not match... this may cause errors during training')
    return checkpoint


def save_checkpoint(
    checkpoint_file: str, epoch: int, model, optimiser, scheduler, results: dict, device
):
    """
    Save checkpoint, overriding previous checkpoint
    """
    # Load an existing checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)
    # Add results to the checkpoint dictionary
    checkpoint["epoch"] = epoch
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimiser_state_dict"] = optimiser.state_dict()
    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint["results"] = (
        results  # simply set the results to the running results dict
    )

    torch.save(checkpoint, checkpoint_file)  # SAVE!


if __name__ == "__main__":
    from config.argclass import ArgClass
    from model_utils import ModelLoader
    from torch.utils.data import DataLoader, SubsetRandomSampler
    import torch.nn as nn
    import torch.optim as optim

    import io
    from contextlib import redirect_stdout


    logging.basicConfig(
        filename='./logs/debug/train_msg3d_debug.log',
        encoding='utf-8',
        filemode='w',
        level=logging.DEBUG
    )
    logging.info("Started train_msg3d debug")
    logging.info("./training/train_msg3d.py")

    model_type = "stgcn2"
    dataset = "ntu120"
    flow_embedding = "cnn"
    evaluation = "CSet"

    # This is just to write the arg verbose output to the logging file... find a better way!
    buf = io.StringIO()
    with redirect_stdout(buf):
        ArgClass(f"./config/{model_type}/{dataset}/{flow_embedding}.yaml", verbose=True)
    logging.info(buf.getvalue())

    arg = ArgClass(f"./config/{model_type}/{dataset}/{flow_embedding}.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arg.model_args["device"] = device
    arg.feeder_args['eval'] = evaluation
    arg.checkpoint_file = "./DELETE_ME.pt"
    arg.batch_size=16 # reduce the batch size for speed...

    # Model
    modelLoader = ModelLoader(arg)
    model = modelLoader.model

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

    # Dataset and dataloader
    feeder_class = arg.import_class(arg.feeder)
    train_dataset = feeder_class(
        **arg.feeder_args,
        split="train"
    )
    # Reduce the size of the training dataset for the sake of speed...
    indices = list(range(arg.batch_size*4))
    sampler = SubsetRandomSampler(indices)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=arg.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Create the loss function(s)
    cls_loss = nn.CrossEntropyLoss()
    loss_funcs = {"cls_loss": cls_loss}

    # Score functions for tracking
    score_funcs = ["ACC", "cls_loss"]

    # Get the device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train for one epoch...
    results = train_network(
        arg=arg,
        model=model,
        loss_funcs=loss_funcs,
        train_loader=train_dataloader,
        score_funcs=score_funcs,
        device=device,
        epochs=1,
        scheduler=scheduler,
        optimiser=optimiser,
        checkpoint_file=arg.checkpoint_file,
        checkpoint_freq=1,
        verbose=True,
    )

    checkpoint = torch.load('./DELETE_ME.pt', map_location='cpu')
    os.remove('./DELETE_ME.pt')

    res = checkpoint['results']
    logging.info(
        f"Training accuracy results: {res['train_ACC']}"
    )
    print(f"Training accuracy results: {res['train_ACC']}")
