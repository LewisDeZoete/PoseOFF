import torch
import torch.nn as nn
import math
import numpy as np
import time
import datetime
import os
from einops import rearrange, repeat
from tqdm import tqdm
from training.loss import AverageMeter
# TODO: Check that this logging method works without having it inside train_network()
import logging

logger = logging.getLogger(__name__)

def run_epoch(
    arg,
    model,
    optimiser,
    data_loader,
    loss_funcs,
    device,
    results,
    prefix="",
):
    """
    loggers contain the following:
        'acc': AverageMeter object per video frame
        'cls_loss': AverageMeter object
        'feature_loss': AverageMeter object
        'recon_loss': AverageMeter object
        'recon_2d_loss': AverageMeter object
        'kl_div': AverageMeter object
    """
    if prefix == "train":
        model.train()
    else:
        model.eval()

    # AverageMeter objects track values (n_values, mean, etc)
    log_acc = [
        AverageMeter() for _ in range(arg.model_args['T'])
    ]  # One AverageMeter for each frame in the input video
    log_auc = AverageMeter()  # AUC 
    log_loss = AverageMeter()  # Total loss
    log_cls_loss = AverageMeter()  # class loss
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
        x = x.float().to(device) # (B, C, T, V, M)
        y = y.long().to(device)
        mask = mask.long().to(device)
        y_hat, x_hat, z_0, z_hat, kl_div = model(x)
        N_cls = y_hat.size(0) // B


        # Class loss (EQ. 13)
        if arg.lambda_1:
            y = y.view(1, B, 1).expand(N_cls, B, y_hat.size(2))
            y_hat_ = rearrange(y_hat, "b i t -> (b t) i")

            cls_loss = arg.lambda_1 * loss_funcs["cls_loss"](y_hat_, y.reshape(-1))

        # Reconstruction/prediction loss (EQ. 10)
        if arg.lambda_2:
            N_rec = x_hat.size(0) // B
            logger.debug(
                f"Original input shape: {x.shape}"
            )
            logger.debug(
                f"Reconstruction: cropping x to include {model.pose_channels} pose channels"
            )
            x_gt = ( # Only need to reconstruct the pose channels!!
                x[:, :model.pose_channels, ...]
                .unsqueeze(0)  # only use the pose channels (x,y,z/conf)
                .expand(N_rec, B, model.pose_channels, T, V, M)
                .reshape(N_rec * B, model.pose_channels, T, V, M)
            )  # for n_sample=1, x_gt.shape = (B, 3, T, V, M)
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
        correct = (predict_label == y.data)
        for frame_no in range(T):
            log_acc[frame_no].update(
                correct[0, :, frame_no]
                .float()
                .mean(),
                B)
        # for i, ratio in enumerate([(i + 1) / 10 for i in range(10)]):
        #     log_acc[i].update(
        #         (predict_label == y.data)
        #         .view(N_cls * B, -1)[:, int(math.ceil(T * ratio)) - 1]
        #         .float()
        #         .mean(),
        #         B,
        #     )
        log_auc.update(
            correct
            .view(N_cls, B, -1)[-1,:,:]
            .float()
            .mean(), 
            B)
        log_loss.update(loss.data.item(), B)
        log_cls_loss.update(cls_loss.data.item(), B)
        log_feature_loss.update(feature_loss.data.item(), B)
        log_recon_loss.update(recon_loss.data.item(), B)

    # Calculate area under curve from average of the 10 accuracy values
    AUC = np.mean([frame.avg.cpu().numpy() for frame in log_acc])
    # AUC = np.mean([log_acc[i].avg.cpu().numpy() for i in range(10)])

    # results[prefix + "_" + "ACC"].append({f"ACC_{(i+1)/10}":log_acc[i].avg for i in range(10)})
    results[prefix + "_" + "ACC"].append([frame.avg.data for frame in log_acc])
    results[prefix + "_" + "AUC"].append(AUC)
    results[prefix + "_" + "loss"].append(log_loss.avg)
    results[prefix + "_" + "cls_loss"].append(log_cls_loss.avg)
    results[prefix + "_" + "feature_loss"].append(log_feature_loss.avg)
    results[prefix + "_" + "recon_loss"].append(log_recon_loss.avg)
    # results[prefix+'kl_div'].append(loggers['kl_div'].avg)

    # ---------------------------------------------------------------------
    logger.info(f"{prefix} X shape: {x.shape}")
    logger.info(f"{prefix} X first joint...: {x[0,:, 15, 8, 0]}")
    logger.info(f"{prefix} Y shape: {y.shape}")
    logger.info(f"{prefix} Y batch index 10: {y[0]}")
    logger.info(f"{prefix} Y_hat shape: {y_hat.shape}")
    logger.info(f"{prefix} Y_hat shape: {torch.argmax(y_hat[0],dim=0)}")

    logger.info(f"{prefix} Class loss: {cls_loss}")
    logger.info(f"{prefix} Recon loss: {recon_loss}")
    logger.info(f"{prefix} Feature loss: {feature_loss}")
    # ---------------------------------------------------------------------
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
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
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

        # TEST only if we've finished the last epoch of training (curr_epoch==max_epoch)
        if test_loader is not None and epoch == epochs:
            model = model.eval()
            with torch.no_grad():
                test_time = run_epoch(
                    arg=arg,
                    model=model,
                    optimiser=optimiser,
                    data_loader=test_loader,
                    loss_funcs=loss_funcs,
                    device=device,
                    results=results,
                    prefix="test",
                )

            results["test_time"].append(test_time)
            # Print the results
            print(f"\t\t{epoch} EPOCH BEST TEST AUC: {max(results['test_AUC'])}")
            print(f"\t\t\tTrain time: {train_time:.2f} seconds")
            print(f"\t\t\tTest time: {test_time:.2f} seconds")
        else:
            # Print results during training
            print(f"\t\t{epoch} EPOCH BEST TRAIN AUC: {max(results['train_AUC'])}")
            print(f"\t\t\tTrain time: {train_time:.2f} seconds")

        if checkpoint_file is not None:
            if epoch % checkpoint_freq == 0:
                save_checkpoint(
                    checkpoint_file, epoch, model, optimiser, scheduler, results, device
                )

    print(f"\tBest train AUC: {torch.tensor(results['train_AUC']).max().item()}")
    # Get the total training time
    total_time = results['train_time']
    if test_loader is not None:
        total_time += results['test_time']
        print(f"\tBest test AUC: {torch.tensor(results['test_AUC']).max().item()}")
    print(f"\tTraining time: \
        {datetime.timedelta(seconds=int(sum(results['train_time']+results['test_time'])))}")
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
                # Print previous results...
                print(f"\t\t{i+1} EPOCH BEST TRAIN AUC: {max(results['train_AUC'][i])}")
                print(f"\t\t\tTrain time: {results['train_time'][i]:.2f} seconds")
    except FileNotFoundError:
        os.makedirs( # Create the folders in the case that they don't exist...
            os.path.dirname(checkpoint_file),
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
    from training.loss import LabelSmoothingCrossEntropy, masked_recon_loss
    import torch.optim as optim

    import io
    from contextlib import redirect_stdout

    # ------------------------------------------------------------------
    model_type = "infogcn2"
    dataset = "ucf101"
    flow_embedding = "base"
    evaluation = "1"
    # ------------------------------------------------------------------

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f'logs/debug/train_eval/train_infogcn2_{dataset}_{evaluation}.log',
        encoding='utf-8',
        filemode='w',
        level=logging.INFO
    )
    logging.info(f"Started train_infogcn {dataset} {flow_embedding} {evaluation}")
    logging.info("./training/train_infogcn.py")


    run_name = f"{dataset}_{evaluation}_{flow_embedding}"

    # This is just to write the arg verbose output to the logging file... find a better way!
    buf = io.StringIO()
    with redirect_stdout(buf):
        ArgClass(f"./config/{model_type}/{dataset}/{flow_embedding}.yaml", verbose=True)
    logging.info(buf.getvalue())

    arg = ArgClass(f"./config/{model_type}/{dataset}/{flow_embedding}.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arg.model_args["device"] = device
    arg.feeder_args['eval'] = evaluation
    arg.feeder_args['data_paths']['CSet'] = "./data/ntu120/aligned_data/ntu120_CSet-flowpose_D3_aligned.npz"
    arg.checkpoint_file = f"./DELETE_ME_{run_name}.pt"

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
        split="train",
        debug=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    test_dataset = feeder_class(
        **arg.feeder_args,
        split="test",
        debug=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=arg.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    # Create the loss function(s)
    cls_loss = LabelSmoothingCrossEntropy(T=arg.model_args["T"])
    recon_loss = masked_recon_loss
    loss_funcs = {"cls_loss": cls_loss, "recon_loss": recon_loss}

    # Score functions for tracking
    score_funcs = ["ACC", "AUC", "cls_loss", "feature_loss", "recon_loss"]

    # Get the device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train for one epoch...
    results = train_network(
        arg=arg,
        model=model,
        loss_funcs=loss_funcs,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        score_funcs=score_funcs,
        device=device,
        epochs=2,
        scheduler=scheduler,
        optimiser=optimiser,
        checkpoint_file=arg.checkpoint_file,
        checkpoint_freq=1,
        verbose=True,
    )

    checkpoint = torch.load(arg.checkpoint_file, map_location='cpu')
    os.remove(arg.checkpoint_file)

    res = checkpoint['results']
    logging.info(
        f"Training accuracy results: {res['train_ACC'][-1]}"
    )
