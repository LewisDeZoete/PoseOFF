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
    model.train()

    # AverageResults objects (work like running loss)
    log_acc = [
        AverageMeter() for _ in range(10)
    ]  # This one is a list of 10 AverageMeter objects
    log_cls_loss = AverageMeter()  # class loss
    # loggers["auc"].reset()  # area under the curve
    log_feature_loss = AverageMeter()  # feature loss
    log_recon_loss = AverageMeter()  # reconstruction loss
    # recon_2d_loss = AverageMeter()  # 2D reconstruction loss

    tbar = tqdm(data_loader, dynamic_ncols=True, desc=desc)

    start = time.time()
    for x, y, mask, index in tbar:
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

        # Update loggers
        value, predict_label = torch.max(y_hat.data, 1)
        for i, ratio in enumerate([(i + 1) / 10 for i in range(10)]):
            log_acc[i].update(
                (predict_label == y.data)
                .view(N_cls * B, -1)[:, int(math.ceil(T * ratio)) - 1]
                .float()
                .mean(),
                B,
            )
        log_cls_loss.update(cls_loss.data.item(), B)
        log_feature_loss.update(feature_loss.data.item(), B)
        log_recon_loss.update(recon_loss.data.item(), B)
        # loggers['kl_div'].update(kl_div.data.item(), B)

        # AUC = np.mean([results['acc'][i].avg.cpu().numpy() for i in range(10)])
        # tbar.set_description(
        #     f"[Epoch #{epoch}] "
        #     f"AUC:{AUC:.3f}, "
        #     f"CLS:{results['cls_loss'].avg:.3f}, "
        #     f"FT:{results['feature_loss'].avg:.3f}, "
        #     f"RECON:{results['recon_loss'].avg:.5f}, "
        # )

    # Calculate area under curve from the 10 accuracy values
    AUC = np.mean([results["acc"][i].avg.cpu().numpy() for i in range(10)])
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
    epochs: int = 50,
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
    if score_funcs is None:
        score_funcs = {}  # Empty set

    to_track = ["epoch", "training_time", "train_loss", "lr"]
    if test_loader is not None:
        to_track.append("test_loss")
        max_acc = 0  # If we have a test loader, track the best test accuracy!
    if score_funcs is not None:
        for eval_score in score_funcs:
            to_track.append("train_" + eval_score)
            if test_loader is not None:
                to_track.append("test_" + eval_score)

    total_train_time = 0  # How long have we spent in the training loop?
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
        start_epoch = checkpoint["epoch"]
        try:
            optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except KeyError:
            pass  # Only just created the checkpoint file
        del checkpoint  # might save us from OOM issues

    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch"):
        model = model.train()  # Put our model in training mode

        # Run the training epoch
        total_train_time += run_epoch(
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
        results["training_time"].append(total_train_time)

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
                run_epoch(
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
            # Save best results!
            if results["train_accuracy"][-1] > max_acc:
                max_acc = results["train_accuracy"][-1]
            print(f"\t\t{epoch} EPOCH BEST TEST ACC: {max(results['test_accuracy'])}")

        if checkpoint_file is not None:
            if epoch % checkpoint_freq == 0:
                save_checkpoint(
                    checkpoint_file, epoch, model, optimiser, scheduler, results, device
                )

    return results


def load_checkpoint(checkpoint_file: str, device):
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        print(f"\tResuming from checkpoint at epoch {checkpoint['epoch']}")
    except FileNotFoundError:
        print(f"\tCreated new checkpoint file: {checkpoint_file}")
        checkpoint = {"epoch": 0}
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
    from feeders import feeder
    from torch.utils.data import DataLoader

    arg = ArgClass("./config/custom_pose/train_joint_infogcn.yaml")

    # Model
    modelLoader = ModelLoader(arg)
    model = modelLoader.model

    # optimiser
    # optimiser = optim.SGD()

    # Dataset and dataloader
    train_dataset = feeder.Feeder(**arg.feeder_args)
    train_dataloader = DataLoader(
        train_dataset, batch_size=arg.batch_size, shuffle=True
    )
