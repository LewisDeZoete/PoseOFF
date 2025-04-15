import torch
import torch.nn as nn
import math
import numpy as np
import time
from einops import rearrange, repeat
from tqdm import tqdm
import sys

sys.path.extend(["./"])
from loss import AverageMeter


def run_epoch(
    arg,
    model,
    optimiser,
    data_loader,
    loss_funcs,
    device,
    results,
    prefix="",
    desc=None,
):
    """
    loggers contain the following:
        'acc': list of 10 AverageMeter objects
        'cls_loss': AverageMeter object
        'feature_loss': AverageMeter object
        'recon_loss': AverageMeter object
        'recon_2d_loss': AverageMeter object (maybe not?)
        'kl_div': AverageMeter object (maybe not?)
    """
    if prefix == "train":
        model.train()
    else:
        model.eval()

    # AverageResults objects (work like running loss)
    log_acc = [
        AverageMeter() for _ in range(10)
    ]  # This one is a list of 10 AverageMeter objects
    log_loss = AverageMeter()  # Total loss
    log_cls_loss = AverageMeter()  # class loss
    # log_auc = AverageMeter()  # AUC TODO: make an AUC logger
    log_feature_loss = AverageMeter()  # feature loss
    log_recon_loss = AverageMeter()  # reconstruction loss
    # recon_2d_loss = AverageMeter()  # 2D reconstruction loss

    # tbar = tqdm(data_loader, dynamic_ncols=True, desc=desc)

    start = time.time()
    for x, y, mask, index in data_loader:
        cls_loss, recon_loss, feature_loss = (
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
        B, C, T, V, M = x.shape
        x = x.float().to(device)
        y = y.long().to(device)
        mask = mask.long().to(device)
        y_hat, x_hat, z_0, z_hat, kl_div = model(x)
        N_cls = y_hat.size(0) // B

        # Class loss (EQ. 13)
        if arg.lambda_1:
            y = y.view(1, B, 1).expand(N_cls, B, y_hat.size(2))
            y_hat_ = rearrange(y_hat, "b i t -> (b t) i")

            # TODO: make sure the second cls_loss here is LabelSmoothingCrossEntropy
            cls_loss = arg.lambda_1 * loss_funcs["cls_loss"](y_hat_, y.reshape(-1))

        # Reconstruction/prediction loss (EQ. 10)
        if arg.lambda_2:
            N_rec = x_hat.size(0) // B
            x_gt = (
                x[:, :3, ...]
                .unsqueeze(0)  # only use the first 3 channels (x,y,conf)
                .expand(N_rec, B, 3, T, V, M)
                .reshape(N_rec * B, 3, T, V, M)
            )
            mask_recon = repeat(mask, "b c t v m -> n b c t v m", n=N_rec).clone()
            for i in range(N_rec):
                if N_rec == arg.model_args["n_step"]:
                    mask_recon[i, :, :, : i + 1, :, :] = 0.0
                else:
                    mask_recon[i, :, :, :i, :, :] = 0.0
            mask_recon = rearrange(mask_recon, "n b c t v m -> (n b) c t v m")

            recon_loss = arg.lambda_2 * loss_funcs["recon_loss"](
                x_hat, x_gt, mask_recon
            )

        # Feature loss (EQ. 11)
        if arg.lambda_3:
            N_step = arg.model_args["n_step"]
            B_, C, T, V = z_0.shape
            z_0 = repeat(z_0, "b c t v-> n b c t v", n=N_step)
            z_hat = z_hat.view(N_step, B_, C, T, V)
            mask_feature = z_hat != 0.0

            feature_loss = arg.lambda_3 * loss_funcs["recon_loss"](
                z_hat, z_0, mask_feature
            )  # F.mse_loss(z_0, z_hat)

        # KL divergence (regularization)
        # TODO: REMOVE (kl_div is always torch.tensor(0.0))
        if arg.lambda_4:
            kl_div = arg.lambda_4 * kl_div

        if prefix == "train":
            # Add up losses (EQ. 14 from paper)
            loss = cls_loss + recon_loss + feature_loss + kl_div

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
        else:
            loss = arg.lambda_1 * cls_loss + arg.lambda_2 * recon_loss

        # Update loggers
        _, predict_label = torch.max(y_hat.data, 1)
        for i, ratio in enumerate([(i + 1) / 10 for i in range(10)]):
            log_acc[i].update(
                (predict_label == y.data)
                .view(N_cls * B, -1)[:, int(math.ceil(T * ratio)) - 1]
                .float()
                .mean(),
                B,
            )
        log_loss.update(loss.data.item(), B)
        log_cls_loss.update(cls_loss.data.item(), B)
        log_feature_loss.update(feature_loss.data.item(), B)
        log_recon_loss.update(recon_loss.data.item(), B)
        # loggers['kl_div'].update(kl_div.data.item(), B)

        # TODO: RENAME AUC it the average classification accuracy over the 10 cls_heads of SODE
        AUC = np.mean([log_acc[i].avg.cpu().numpy() for i in range(10)])
        # tbar.set_description(
        #     f"[Epoch #{epoch}] "
        #     f"AUC:{AUC:.3f}, "
        #     f"CLS:{results['cls_loss'].avg:.3f}, "
        #     f"FT:{results['feature_loss'].avg:.3f}, "
        #     f"RECON:{results['recon_loss'].avg:.5f}, "
        # )

    # Calculate area under curve from the 10 accuracy values
    AUC = np.mean([log_acc[i].avg.cpu().numpy() for i in range(10)])
    # train_dict = {
    #     "train/Recon2D_loss": results['recon_loss'].avg,
    #     "train/cls_loss": results['cls_loss'].avg,
    #     "train/feature_loss": results['feature_loss'].avg,
    #     "train/kl_div": results['kl_div'].avg,
    #     "train/AUC": AUC,
    # }
    # train_dict.update({f"train/ACC_{(i + 1) / 10}": results['acc'][i].avg for i in range(10)})
    # wandb.log(train_dict)

    results[prefix + "_" + "AUC"].append(AUC)
    results[prefix + "_" + "loss"].append(loss)
    results[prefix + "_" + "cls_loss"].append(log_cls_loss.avg)
    results[prefix + "_" + "feature_loss"].append(log_feature_loss.avg)
    results[prefix + "_" + "recon_loss"].append(log_recon_loss.avg)
    # results[prefix+'kl_div'].append(loggers['kl_div'].avg)

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
):
    """
    Train simple neural networks

    Arguments:
        model: the PyTorch model / "Module" to train
        loss_funcs: the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
        train_loader: PyTorch DataLoader object that returns tuples of (input, label) pairs.
        test_loader: Optional PyTorch DataLoader to evaluate on after every epoch
        score_funcs: A dictionary of scoring functions to use to evalue the performance of the model
        epochs: the number of training epochs to perform
        device: the compute lodation to perform training

    """
    to_track = ["epoch", "training_time", "train_loss", "lr"]
    if test_loader is not None:
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
        checkpoint = load_checkpoint(checkpoint_file, device)
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
            optimiser=optimiser,
            data_loader=train_loader,
            loss_funcs=loss_funcs,
            device=device,
            results=results,
            prefix="train",
            desc="Training",
        )

        # Append the post-training results to the results dictionary
        results["epoch"].append(epoch)
        results["training_time"].append(train_time)

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
                    optimiser=optimiser,
                    data_loader=train_loader,
                    loss_funcs=loss_funcs,
                    device=device,
                    results=results,
                    prefix="test",
                    desc="Testing",
                )
            # TODO: AUC is not in both ms-g3d and infogcn results dict, change to accomidate
            print(f"\t\t{epoch} EPOCH BEST TEST ACC: {max(results['test_AUC'])}")
            print(f"\t\t\tTrain time: {train_time:.2f} seconds")
            print(f"\t\t\tTest time: {test_time:.2f} seconds")

        if checkpoint_file is not None:
            if epoch % checkpoint_freq == 0:
                save_checkpoint(
                    checkpoint_file, epoch, model, optimiser, scheduler, results, device
                )

    return results


def load_checkpoint(checkpoint_file: str, device):
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
    except FileNotFoundError:
        print(f"\tCreated new checkpoint file: {checkpoint_file}")
        checkpoint = {
            "epoch": 0
        }  # Start from scratch, indexing starts from 1 in this case
        torch.save(checkpoint, checkpoint_file)
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
    from model import ModelLoader
    from feeders import ucf101
    from torch.utils.data import DataLoader
    from loss import LabelSmoothingCrossEntropy, masked_recon_loss
    import torch.optim as optim

    arg = ArgClass("./config/ucf101/train_joint_infogcn.yaml")
    arg.checkpoint_file = "DELETE_ME.pt"

    # Model
    modelLoader = ModelLoader(arg)
    model = modelLoader.model

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

    # Dataset and dataloader
    train_dataset = ucf101.Feeder(**arg.feeder_args)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )

    # Create the loss function(s)
    cls_loss = LabelSmoothingCrossEntropy(T=arg.model_args["T"])
    recon_loss = masked_recon_loss
    loss_funcs = {"cls_loss": cls_loss, "recon_loss": recon_loss}

    # Score functions for tracking
    score_funcs = ["AUC", "cls_loss", "feature_loss", "recon_loss"]

    # Get the device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = train_network(
        arg,
        model,
        loss_funcs,
        train_dataloader,
        score_funcs=score_funcs,
        device=device,
        epochs=1,
        optimiser=optimiser,
        checkpoint_file=arg.checkpoint_file,
        checkpoint_freq=arg.checkpoint_freq,
    )
