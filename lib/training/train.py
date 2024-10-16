# ------------------------------------------------------------------------------
# From 'Inside Deep Learning' by Edward Raff
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tqdm.autonotebook import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import pandas as pd

import time

def run_epoch(model, optimiser, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
    """
    model -- the PyTorch model / "Module" to run for one epoch
    optimiser -- the object that will update the weights of the network
    data_loader -- DataLoader object that returns tuples of (input, label) pairs. 
    loss_func -- the loss function that takes in two arguments, the model outputs and the labels, and returns a score
    device -- the compute lodation to perform training
    results -- dictionary where the results are stored
    score_funcs -- a dictionary of scoring functions to use to evalue the performance of the model
    prefix -- a string to pre-fix to any scores placed into the _results_ dictionary. 
    desc -- a description to use for the progress bar.     
    """
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        #Move the batch to the device we are using. 
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs) #this just computed f_Θ(x(i))
        # Compute loss.
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        #Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            #moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            #add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())

    #end training epoch
    end = time.time()
    
    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1: #We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    #Else, we assume we are working on a regression problem
    
    results[prefix + " loss"].append( np.mean(running_loss) )
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append( score_func(y_true, y_pred) )
            # print(f'{prefix}: {score_func(y_true, y_pred)}')
        except:
            results[prefix + " " + name].append(float("NaN"))
            # print('NaN')
    return end-start #time spent on epoch


def train_simple_network(model, loss_func, train_loader, test_loader=None, val_loader=None,
                        score_funcs=None, device="cpu", epochs=50,
                        scheduler=None, optimiser=None, checkpoint_file=None):
    """
    Train simple neural networks
    
    Keyword arguments:\n
    model -- the PyTorch model / "Module" to train\n
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score\n
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs. \n
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch\n
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model\n
    epochs -- the number of training epochs to perform\n
    device -- the compute lodation to perform training\n
    
    """
    if score_funcs == None:
        score_funcs = {}#Empty set

    to_track = ["epoch", "training time", "train loss", "lr"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
        max_acc = 0 # If we have a test loader, track the best test accuracy!
    for eval_score in score_funcs:
        to_track.append("train " + eval_score )
        if val_loader is not None:
            to_track.append("val_loader" + eval_score)
        if test_loader is not None:
            to_track.append("test " + eval_score )
        
    total_train_time = 0 #How long have we spent in the training loop? 
    results = {}
    print('\tTracking:')
    #Initialize every item with an empty list
    for item in to_track:
        results[item] = []
        print(f'\t\t{item}')

    #Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model = model.train()#Put our model in training mode
        
        # Run the training epoch
        total_train_time += run_epoch(model, optimiser, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")

        # Append the post-training results to the results dictionary
        results["epoch"].append( epoch )
        results["training time"].append( total_train_time )

        # VAL
        if val_loader is not None:
            model = model.eval() #Set the model to "evaluation" mode, b/c we don't want to make any updates!
            with torch.no_grad():
                run_epoch(model, optimiser, val_loader, loss_func, device, results, score_funcs, prefix="val", desc="Validating")
        
        # Step the scheduler after each training epoch and append lr
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(results["val loss"][-1])
            else:
                scheduler.step()
            results['lr'].append(scheduler.get_last_lr()[0])

        # TEST
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimiser, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")
            # Save best results!
            if results['train accuracy'][-1] > max_acc:
                max_acc = results['train accuracy']
                print(f'\t\tBEST: {max_acc}')
                # torch.save(model, f'./model/checkpoints/{checkpoint_file}.pt')

                    
        if checkpoint_file is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'results' : results
                }, checkpoint_file)

    return results


def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj