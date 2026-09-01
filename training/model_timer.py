#!/usr/bin/env python3

from config.argclass import ArgClass
from model_utils import ModelLoader
from torch.utils.data import DataLoader
import argparse
import time
import torch
import logging


def time_model(model, dataloader, device, times):
    model.eval().to(device)

    # ---- Warm-up ----
    model_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(20):
            try:
                x, *_ = next(model_iter)
            except StopIteration:
                model_iter = iter(dataloader)
                x, *_ = next(model_iter)
            x=x.to(device)
            _ = model(x)
        torch.cuda.synchronize()

    # ---- Timing ----
    with torch.no_grad():
        for x, *_ in dataloader:
            x = x.to(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # start = time.perf_counter()
            start_event.record()
            _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            # end = time.perf_counter()
            # times.append((end-start)*1000)
            times.append(start_event.elapsed_time(end_event))

    return times



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        dest="model_type",
        default="infogcn2",
        help="Base model type (e.g. infogcn2, msg3d, stgcn2)"
    )
    parser.add_argument(
        "-f",
        dest="flow_embedding",
        default="base",
        help="Optical flow embedding method [base, cnn] (default=base)"
    )
    parsed = parser.parse_args()

    # -----------------------------------
    # stgcn2, msg3d, infogcn2
    model = parsed.model_type
    flow_embedding = parsed.flow_embedding
    dataset = "ntu"
    eval = "CS"
    # -----------------------------------

    logging.basicConfig(
        filename=f"./logs/debug/model_timing/time_{model}-{flow_embedding}.log",
        encoding='utf-8',
        filemode='w',
        level=logging.INFO
    )
    logging.info(f"Started time {model} {flow_embedding}")
    logging.info("./training/model_timer.py")
    print(f"Started time {model} {flow_embedding}")
    print("./training/model_timer.py")

    # Create arg object
    arg = ArgClass(f"./config/{model}/{dataset}/{flow_embedding}.yaml")
    arg.feeder_args['eval'] = eval # Add the evaluation to the feeder args...

    # Get the cuda device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    modelLoader = ModelLoader(arg)
    model = modelLoader.model
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable model params: {model_params}")
    logging.info(f"Number of trainable model params: {model_params}")

    # Dataset and dataloader
    feeder_class = arg.import_class(arg.feeder)
    test_dataset = feeder_class(
        **arg.feeder_args,
        split="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )
    logging.info(f"\tDataset contains {len(test_dataset)} samples")

    # Time it by throwing in one sample 100 times
    times = []
    # for i in range(0, 100, 10):
    #     # Just getting one video sample and throwing it in over and over!
    #     x, *_ = test_dataset[i]
    #     x = torch.from_numpy(x).unsqueeze(0).float().to(device)
    #     print(f"X shape: {x.shape}")

    # Time the model!!
    times = time_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        times=times
    )

    print(f"Average latency: {sum(times)/len(times):.4f} ms")
    print(f"FPS: {1000 / (sum(times)/len(times)):0.4f}")
    logging.info(f"Average latency: {sum(times)/len(times):.4f} ms")
    logging.info(f"FPS: {1000 / (sum(times)/len(times)):0.4f}")
