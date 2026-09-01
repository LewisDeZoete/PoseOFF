#!/usr/bin/env python3

import torch
import torch.nn as nn
import math
import numpy as np
import time
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
    data_loader,
    loss_funcs,
    device,
    results,
):
    """
    loggers contain the following:
        'acc': AverageMeter object per video frame
        'cls_loss': AverageMeter object
    """
    model.eval()

    # AverageMeter objects track values (n_values, mean, etc)
    log_acc = AverageMeter() # accuracy
    log_cls_loss = AverageMeter()  # class loss

    # Create arrays to save all of the labels and predicted labels to
    log_truth = torch.tensor([]).to(device)
    log_pred = torch.tensor([]).to(device)

    start = time.time()
    inference_time = 0.0 # Timing epoch inference time
    for x, y, mask, index in data_loader:
        loss = torch.tensor(0.0)

        B, C, T, V, M = x.shape
        x = x.float().to(device)
        y = y.long().to(device)

        # If training on GPU, get cuda inference time...
        if torch.cuda.is_available():
            y_hat, elapsed_ms = time_cuda(model, x)
            inference_time += elapsed_ms
        else:
            inference_start = time.time()
            y_hat = model(x)
            inference_time += time.time() - start

        # Calculate the loss and correct predictions
        loss = loss_funcs["cls_loss"](y_hat, y)
        correct = (torch.argmax(y_hat, dim=1) == y)

        # Get the predicted labels
        _, predict_label = torch.max(y_hat.data, 1)

        # Update loggers
        log_acc.update(correct.float().mean(), B)
        log_cls_loss.update(loss, B)

        # Append the true and predicted labels to our logs
        log_truth = torch.cat((log_truth, y))
        log_pred = torch.cat((log_pred, predict_label))

    results["ACC"].append([log_acc.avg])
    results["cls_loss"].append(log_cls_loss.avg)

    results["truth"].append(log_truth)
    results["pred"].append(log_pred)

    end = time.time()
    return end - start, inference_time  # time spent on epoch


def eval_network(
    arg,
    model,
    loss_funcs,
    test_loader,
    checkpoint_file: str,
    score_funcs=None,
    device="cpu",
    save_attention:  bool = False,
):
    """
    EVAL simple neural network (only one epoch)

    Arguments:
        model: the PyTorch model / "Module" to train
        loss_funcs: the loss function that takes in batch in two arguments,
            the model outputs and the labels, and returns a score

        score_funcs: A dictionary of scoring functions to use to evalue the performance of the model
        device: the compute lodation to perform training
    """
    # to_track contains the keys of the tracked items in the results dict
    to_track = ["epoch", "test_time", "inference_time", "loss", "lr", "truth", "pred"]
    if score_funcs is not None:
        for eval_score in score_funcs:
            to_track.append(eval_score)
    # optionally add "attention"...

    results = {}
    print("\tTracking:")
    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []
        print(f"\t\t{item}")

    # Place the model on the correct compute resource (CPU or GPU)
    model.to(device)

    # TEST
    model = model.eval()
    with torch.no_grad():
        test_time, inference_time = run_epoch(
            arg=arg,
            model=model,
            data_loader=test_loader,
            loss_funcs=loss_funcs,
            device=device,
            results=results,
        )

    results["test_time"].append(test_time)
    results["inference_time"].append(inference_time/len(test_loader))

    # Print out the score functions at the end of training
    if score_funcs is not None:
        for eval_score in score_funcs:
            print(f"Results for {eval_score}: {results[eval_score]}")

    # Print out time it took for the whole epoch, and average inference time
    print(f"Total epoch time: {results['test_time']}")
    print(f"Average batch inference time: {results['inference_time'][0]}")
    print(f"Average sample inference time: {results['inference_time'][0]/arg.batch_size}")

    return results


def time_cuda(model, x):
    # For model timing! This is the inference...
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    y_hat = model(x)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    return y_hat, elapsed_ms


if __name__ == "__main__":
    from config.argclass import ArgClass
    from model_utils import ModelLoader
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim


    logging.basicConfig(
        filename='./logs/debug/train_eval/eval_msg3d_debug.log',
        encoding='utf-8',
        filemode='w',
        level=logging.DEBUG
    )
    logging.info("Started eval_msg3d debug")
    logging.info("./training/eval_msg3d.py")

    # -----------------------------------
    model = "stgcn2"
    dataset = "ntu"
    flow_embedding = "cnn"
    evaluation = "CS"
    obs_ratio = 1.0
    modifier = "D3"
    # -----------------------------------

    if flow_embedding == "base":
        run_name=f"{model}_{dataset}_{evaluation}_{flow_embedding}"
    else:
        run_name=f"{model}_{dataset}_{evaluation}_{flow_embedding}_{modifier}" # May need to adjust this
    logging.info(f"Run name: {run_name}")


    # Create arg object
    arg = ArgClass(f"./config/{model}/{dataset}/{flow_embedding}.yaml")
    arg.feeder_args['eval'] = evaluation
    arg.feeder_args['obs_ratio'] = obs_ratio
    arg.batch_size=16 # reduce the batch size for speed...

    # Trained model checkpoint to load...
    arg.checkpoint_file = osp.join(  # results/{dataset}/{evaluation}/train/{run}.pt
        arg.save_location,
        evaluation,
        "train",
        run_name + ".pt"
    )
    assert osp.isfile(arg.checkpoint_file)
    logging.info(f"Checkpoint file: {arg.checkpoint_file}")


    # Model
    modelLoader = ModelLoader(arg)
    model = modelLoader.model

    # Dataset and dataloader
    feeder_class = arg.import_class(arg.feeder)
    test_dataset = feeder_class(
        **arg.feeder_args,
        split="test",
        debug=False if torch.cuda.is_available() else True,
        # debug=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=arg.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )
    logging.info(f"\tDataset contains {len(test_dataset)} samples")

    # Create the loss function(s)
    cls_loss = nn.CrossEntropyLoss()
    loss_funcs = {"cls_loss": cls_loss}

    # Score functions (train+test), accuracy, cls_loss
    score_funcs = ["ACC", "cls_loss"]

    # Get the device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = eval_network(
        arg=arg,
        model=model,
        loss_funcs=loss_funcs,
        test_loader=test_dataloader,
        checkpoint_file=arg.checkpoint_file,
        score_funcs=score_funcs,
        device=device,
    )
    logging.info(f"Total test time: {results['test_time']}")
    logging.info(f"Averag inference time per batch: {results['inference_time'][0]}")
    logging.info(f"Average sample inference time: {results['inference_time'][0]/arg.batch_size}")
    print(f"Total test time: {results['test_time']}")
    print(f"Averag inference time per batch: {results['inference_time'][0]}")
    print(f"Average sample inference time: {results['inference_time'][0]/arg.batch_size}")

    # torch.save(results, arg.save_name)
