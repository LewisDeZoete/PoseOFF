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
    save_attention=False
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

    # Create arrays to save all of the labels and predicted labels to
    log_truth = torch.tensor([]).to(device)
    log_pred = torch.tensor([]).to(device)

    start = time.time()
    inference_time = 0.0 # Timing epoch inference time
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

        # If training on GPU, get cuda inference time...
        if torch.cuda.is_available():
            y_hat, x_hat, z_0, z_hat, kl_div, elapsed_ms = time_cuda(model, x)
            inference_time += elapsed_ms
        else:
            inference_start = time.time()
            y_hat, x_hat, z_0, z_hat, kl_div = model(x)
            inference_time += time.time() - start
        N_cls = y_hat.size(0) // B

        # Class loss (EQ. 13)
        if arg.lambda_1:
            y = y.view(1, B, 1).expand(N_cls, B, y_hat.size(2))
            y_hat_ = rearrange(y_hat, "b i t -> (b t) i")

            cls_loss = arg.lambda_1 * \
                loss_funcs["cls_loss"](y_hat_, y.reshape(-1))

        # Reconstruction/prediction loss (EQ. 10)
        if arg.lambda_2:
            N_rec = x_hat.size(0) // B
            x_gt = (
                x[:, :3, ...]
                .unsqueeze(0)  # only use the first 3 channels (x,y,conf)
                .expand(N_rec, B, 3, T, V, M)
                .reshape(N_rec * B, 3, T, V, M)
            )
            mask_recon = repeat(
                mask, "b c t v m -> n b c t v m", n=N_rec).clone()
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

        log_auc.update(
            correct
            .view(N_cls, B, -1)[-1, :, :]
            .float()
            .mean(),
            B)
        log_loss.update(loss.data.item(), B)
        log_cls_loss.update(cls_loss.data.item(), B)
        log_feature_loss.update(feature_loss.data.item(), B)
        log_recon_loss.update(recon_loss.data.item(), B)

        # Append the true and predicted labels to our logs (batch_size, frames(64))
        log_truth = torch.cat((log_truth, y.data[0]))
        log_pred = torch.cat((log_pred, predict_label))

        if save_attention:
            # NOTE: we average over all of the limbs and attn heads...
            attentions = model.get_attention()  # [ ((N M V) H Q K)*4 ]
            for layer_no, layer in enumerate(attentions):
                layer = rearrange(
                    layer,
                    '(N M V) H Q K -> N M V H Q K',
                    N=arg.batch_size,
                    M=2,
                    V=arg.model_args['num_point']
                )
                layer = layer.mean((2, 3))  # Average over limbs and heads
                results["attentions"][layer_no] = torch.cat(
                    (layer, results["attentions"][layer_no], layer)
                )

    # Calculate area under curve from average of the 10 accuracy values
    AUC = np.mean([frame.avg.cpu().numpy() for frame in log_acc])

    results["ACC"].append([frame.avg.data for frame in log_acc])
    results["AUC"].append(AUC)
    results["loss"].append(log_loss.avg)
    results["cls_loss"].append(log_cls_loss.avg)
    results["feature_loss"].append(log_feature_loss.avg)
    results["recon_loss"].append(log_recon_loss.avg)

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
        loss_funcs: the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score

        test_loader: Optional PyTorch DataLoader to evaluate on after every epoch
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
            save_attention=save_attention
        )

    results["test_time"].append(test_time)
    results["inference_time"].append(inference_time/len(test_loader))

    print(f"\tBest test ACC: {torch.tensor(results['ACC']).max().item()*100:.2f}%")
    print(f"\tTest AUC: {results['AUC']}")

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
    y_hat, x_hat, z_0, z_hat, kl_div = model(x)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    return y_hat, x_hat, z_0, z_hat, kl_div, elapsed_ms


if __name__ == "__main__":
    from config.argclass import ArgClass
    from model_utils import ModelLoader
    from torch.utils.data import DataLoader
    from training.loss import LabelSmoothingCrossEntropy, masked_recon_loss
    import torch.optim as optim


    logging.basicConfig(
        filename='./logs/debug/train_eval/eval_infogcn_debug.log',
        encoding='utf-8',
        filemode='w',
        level=logging.DEBUG
    )
    logging.info("Started eval_infogcn debug located at:")
    logging.info("\t./training/eval_msg3d.py")


    # -----------------------------------
    model = "infogcn2"
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
    arg.checkpoint_file = osp.join(  # results/{dataset}/{eval}/train/{run}.pt
        arg.save_location,
        evaluation,
        "train",
        run_name + ".pt"
    )

    # Model
    modelLoader = ModelLoader(arg)
    model = modelLoader.model

    # Dataset and dataloader
    feeder_class = arg.import_class(arg.feeder)
    test_dataset = feeder_class(
        **arg.feeder_args,
        split="test",
        debug=False if torch.cuda.is_available() else True,
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
    cls_loss = LabelSmoothingCrossEntropy(T=arg.model_args["T"])
    recon_loss = masked_recon_loss
    loss_funcs = {"cls_loss": cls_loss, "recon_loss": recon_loss}

    # Score functions for tracking
    score_funcs = ["ACC", "AUC", "cls_loss", "feature_loss", "recon_loss"]

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
        save_attention=False,
    )
    logging.info(f"Total test time: {results['test_time']}")
    logging.info(f"Averag inference time per batch: {results['inference_time'][0]}")
    logging.info(f"Average sample inference time: {results['inference_time'][0]/arg.batch_size}")
    print(f"Total test time: {results['test_time']}")
    print(f"Averag inference time per batch: {results['inference_time'][0]}")
    print(f"Average sample inference time: {results['inference_time'][0]/arg.batch_size}")

    # torch.save(results, arg.save_name)
