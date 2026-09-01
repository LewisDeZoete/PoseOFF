import os
import os.path as osp
from data_gen.utils import LoadVideo, GetFlow, PoseOFFSampler, PoseOFFSampler_LK, PoseOFFSampler_NF
from config.argclass import ArgClass
import torch
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.v2 as v2
from einops import rearrange
import numpy as np
import pickle
import time
import argparse


def human_k(n):
    """Get the sample number based on the sample name."""
    if n<1000:
        return "0"
    elif n >= 1000:
        val = n/1000
        s = f"{val:.1f}".rstrip('0').rstrip('.')
        return f"{s}k"
    else:
        return str(n)

def remove_frame_drops(flow_data, frames_drop_list):
    """Remove the frame drops from the flow data."""
    # Convert the list to a set for efficient membership testing
    skip_set = set(frame-1 for frame in frames_drop_list if frame > 0)

    # Create a boolean mask indicating which frames to keep
    mask = np.array([i not in skip_set for i in range(flow_data.shape[0])])

    # Use the mask to filter the array
    return flow_data[mask]


def get_raw_poseoff_data(
        data_root_path,
        dataset,
        vid_transforms,
        poseOFFTransform,
        idx_start=0,
        idx_end=20000
):
    save_path = osp.join(data_root_path, 'flow_data', 'export_tmp')
    stat_path = osp.join(data_root_path, 'statistics')
    rgb_path = '../Datasets/NTU_RGBD{0}/nturgb+d_rgb{0}/'.format('120' if dataset == 'ntu120' else '')
    skes_name_file = osp.join(stat_path, f'ntu_rgbd{120 if dataset == "ntu120" else ""}-available.txt')
    denoised_skes_data_file = osp.join(data_root_path, 'denoised_data', 'raw_denoised_colors.pkl')
    frames_drop_file = osp.join(data_root_path, 'raw_data', 'frames_drop_skes.pkl')

    # Get the skeleton names
    skes_names = np.loadtxt(skes_name_file, dtype=str)
    num_files = skes_names.size
    print('Found %d available skeleton files.\n' % num_files, flush=True)

    # Create the save path (if it doesn't exist already)
    os.makedirs(save_path, exist_ok=True)

    # Print processing information before processing...
    print(f'Processing poseoff samples for {dataset} dataset...', flush=True)
    print(f'\tIndex {idx_start} to {idx_end}')
    try:
        print(f'\tSkeleton {skes_names[idx_start]} to {skes_names[idx_end-1]}')
    except IndexError:
        print(f'\tSkeleton {skes_names[idx_start]} to {skes_names[-1]}')

    poseoff_data = []

    # Assuming we've already denoised all the pose data
    with open(denoised_skes_data_file, 'rb') as fr:  # load raw skeletons data
        denoised_skes_data = pickle.load(fr)

    # Get the frame drop dictionary to trip the flow data
    with open(frames_drop_file, 'rb') as fr:
        frames_drop_skes = pickle.load(fr)

    # Create an enumerate object for the skeleton names
    enum = enumerate(skes_names[idx_start:idx_end])
    # Start timing (time_tmp for printing times during inference)
    start_time = time.time()
    start_time_tmp = time.time()
    for ske_number, ske_name in enum:
        ske_number += idx_start
        # Get the flow data
        rgb_name = osp.join(rgb_path, ske_name + '_rgb.avi')
        rgb_data = vid_transforms(rgb_name) # This is either FLOW or Video

        # Get the pose data
        poses = denoised_skes_data[ske_number]
        poses = rearrange(poses, 'T M V C -> C T V M')

        # Remove the frame drops from flow_data
        if ske_name in frames_drop_skes:
            flow_data = remove_frame_drops(rgb_data, frames_drop_skes[ske_name])
            print(f'\tFrames dropped for {ske_name}: {len(frames_drop_skes[ske_name])}', flush=True)
            if 0 in frames_drop_skes[ske_name]:
                print('\t\tFrame 0 in skip list, add blank frame to poses', flush=True)
                C, _, V, M = poses.shape
                poses = np.concatenate([np.zeros((C, 1, V, M)), poses], axis=1)

        # Get the poseoff data and reshape!
        poseoff = poseOFFTransform(rgb_data, poses)
        poseoff_data.append(rearrange(poseoff, 'C T V M -> T (M V C)'))

        if (ske_number+1) % 100 == 0:
            print(f'Processed {ske_number-499}-{ske_number} in {time.time()-start_time_tmp:0.2f} seconds', flush=True)
            start_time_tmp = time.time()
    # Print the time it took to process this batch of data
    print(f"Processed {idx_end-idx_start} in: {time.time()-start_time:0.2f} seconds", flush=True)

    # Save the data
    data_name = f"flow_{human_k(idx_start)}-{human_k(idx_end)}"
    poseoff_filename = osp.join(save_path, f'{data_name}.pkl')
    with open(poseoff_filename, 'wb') as f:
        pickle.dump(poseoff_data, f, pickle.HIGHEST_PROTOCOL)

    print(f'Poseoff samples for {dataset} dataset generated successfully!', flush=True)
    print(f'Data saved to {save_path}', flush=True)

    return poseoff_filename


def parse_args():
    """Parse the commandline args for processing.

    Returns:
        arg (ArgClass): ArgClass object from the infogcn/ntu/cnn.yaml config.
        transform_args (dict): Transform arguments taken from config.
            NORE: Command line arguments overwrite the default values.
    """

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Preparation')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        default='ntu',
        help='Dataset, either `ntu` or `ntu120` (default=ntu)'
    )
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        default=2000,
        type=int,
        help='Processing batch size (default 2000), \
            processes from (batch_number-1)*batch_size - batch_number*batch_size'
    )
    parser.add_argument(
        '--batch_number',
        dest='batch_number',
        default=1,
        type=int,
        help='Batch number for processing'
    )
    parser.add_argument(
        '--mag_threshold',
        dest='mag_threshold',
        default=100,
        type=int,
        help='Optical flow magnitude threshold, limiting how large flow arrows will be. Default is 100.'
    )
    parser.add_argument(
        '--dilation',
        dest='dilation',
        default=3,
        type=int,
        help='Overwrite the dilation value from the yaml config.'
    )
    parser.add_argument(
        '--flow_type',
        dest='flow_type',
        default='RAFT',
        help='The type of motion estimation method used for the PoseOFF extraction.'
    )
    parsed = parser.parse_args()

    # Parsed command line arguments
    assert parsed.dataset in ["ntu", "ntu120"], print("Dataset must be one of ['ntu', 'ntu120']")
    assert parsed.flow_type in ["RAFT", "LK", "NF"], print("Invalid flow type - must be one of ['RAFT', 'LK', 'NF']")
    dataset = parsed.dataset
    flow_type = parsed.flow_type

    # Create a dictionary to overwrite appropriate configs from CLA
    overwrites = {
        "mag_threshold": parsed.mag_threshold,
        "dilation": parsed.dilation,
        "norm": False,
        "match_pose": False,
        "ntu": True
    }

    # Define start and end indexes based on passed arguments
    idx_start = (parsed.batch_number-1)*parsed.batch_size
    idx_end = parsed.batch_number*parsed.batch_size

    print(f'Start index: {idx_start}')
    print(f'End index: {idx_end}')

    # Get the argparse object
    arg = ArgClass(arg=f"./config/infogcn2/ntu{'120' if dataset == 'ntu120' else ''}/cnn.yaml")
    transform_args = arg.extractor
    print(transform_args)

    # Overwrite the transform args with anything parsed form the CLI
    transform_args['poseoff'] = transform_args['poseoff'] | overwrites

    return dataset, flow_type, idx_start, idx_end, arg, transform_args


def main():
    # Parse CLI args
    dataset, flow_type, idx_start, idx_end, arg, transform_args = parse_args()

    # Define paths
    data_root_path = f'./data/{dataset}'

    # Get the device
    device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

    # Load the weights and create the transforms...
    if flow_type == "RAFT":
        # Create the model, move it to device and turn to eval mode
        weights = torch.load(transform_args['flow']['weights'], weights_only=True, map_location=device)
        model = raft_large(progress=False)
        model.load_state_dict(weights)
        model = model.eval().to(device)
        vid_transforms = v2.Compose([
            LoadVideo(max_frames=300),
            v2.Resize(size=transform_args['flow']['imsize']),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # map [0, 1] into [-1, 1]
            GetFlow(model=model, device=device, minibatch_size=transform_args['flow']['minibatch_size'])
            # GetFlow(model=model, device=device, minibatch_size=2) # For non-resized videos
        ])
    else:
        vid_transforms = v2.Compose([
            LoadVideo(max_frames=300),
            v2.Resize(size=transform_args['flow']['imsize']),
            v2.ToDtype(torch.uint8, scale=True),
        ])

    # Create the PoseOFFSampler transform object (This is not the nice way of doing it...)
    poseOFFTransform = {
        'RAFT': PoseOFFSampler(**transform_args['poseoff']),
        'LK': PoseOFFSampler_LK(**transform_args['poseoff']),
        'NF': PoseOFFSampler_NF(**transform_args['poseoff'])
    }[flow_type]

    # Generate the data
    poseoff_filename = get_raw_poseoff_data(
        data_root_path,
        dataset,
        vid_transforms,
        poseOFFTransform,
        idx_start=idx_start,
        idx_end=idx_end
    )
    print(f"Full filename path: {poseoff_filename}", flush=True)


if __name__ == '__main__':
    main()
