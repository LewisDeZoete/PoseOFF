from data_gen.utils import extract_data, PoseOFFSampler, PoseOFFSampler_LK, PoseOFFSampler_NF, LoadVideo
from config.argclass import ArgClass
from torchvision.transforms import v2
import torch
import argparse
import time

parser = argparse.ArgumentParser(prog="poseoff_gendata")

parser.add_argument(
    '-n',
    dest='number',
    help='Class number for processing poseoff of a specific class.'
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
    default=None,
    type=int,
    help='Overwrite the dilation value from the yaml config.'
)
parser.add_argument(
    '--flow_type',
    dest='flow_type',
    default='RAFT',
    help='The type of motion estimation method used for the PoseOFF extraction.'
)
parser.add_argument(
    '--debug',
    action='store_true',
    help='Debug mode to check the data generation process.'
)
parsed = parser.parse_args()
process_number = int(parsed.number) # Get class number command line arg
debug = parsed.debug # Get debug mode command line arg

# Get the arg object and create the classes
arg = ArgClass(arg='./config/infogcn2/ucf101/base.yaml')
transform_args = arg.extractor # grab transforms arg
print(transform_args)

# If a commandline argument is passed, overwrite the yaml config
if parsed.dilation:
    transform_args['poseoff']['dilation'] = parsed.dilation
print(f"Extracting PoseOFF samples for ucf101"
      f"dataset with dilation {transform_args['poseoff']['dilation']}...")

# Create the PoseOFFSampler transform object
if parsed.flow_type in ["LK", "NF"]:
    rgb_transforms = v2.Compose([
        LoadVideo(max_frames=300),
        v2.Resize(size=transform_args['flow']['imsize']),
        v2.ToDtype(torch.uint8, scale=True)
    ])
    poseOFFTransform = PoseOFFSampler_LK(**transform_args['poseoff']) if parsed.flow_type == "LK" \
        else PoseOFFSampler_NF(**transform_args['poseoff'])
else:
    # Create the PoseOFFSampler transform object
    poseOFFTransform = PoseOFFSampler(**transform_args['poseoff'])
    rgb_transforms = None

# ------------------------------
#         PROCESS
# ------------------------------
start = time.time()

print(f"Extracting PoseOFF samples with a dilation of {parsed.dilation}, "
      f"using {parsed.flow_type} motion estimation.")

extract_data(arg,
             process_number=process_number,
             transforms=poseOFFTransform,
             modality='poseoff',
             flow_type=parsed.flow_type,
             rgb_transforms=rgb_transforms, # This is None in the case of RAFT
             debug=debug)

print(f'Processing time: {time.time()-start:.2f}s')
