from data_gen.utils import extract_data, FlowPoseSampler
from config.argclass import ArgClass
import argparse
import time

parser = argparse.ArgumentParser(prog="flowpose_gendata")

parser.add_argument(
    '-n',
    dest='number',
    help='Class number for processing flowpose of a specific class.'
)
parser.add_argument(
    '--debug',
    action='store_true',
    help='Debug mode to check the data generation process.'
)
parser.add_argument(
    '--dilation',
    dest='dilation',
    default=None,
    type=int,
    help='Overwrite the dilation value from the yaml config.'
)
parsed = parser.parse_args()
process_number = int(parsed.number) # Get class number command line arg
debug = parsed.debug # Get debug mode command line arg

# Get the arg object and create the classes
arg = ArgClass(arg='./config/ucf101/base.yaml')
transform_args = arg.extractor['flowpose'] # grab transforms arg

# If a commandline argument is passed, overwrite the yaml config
if parsed.dilation:
    transform_args['dilation'] = parsed.dilation

# Create the FlowPoseSampler transform object
flowPoseTransform = FlowPoseSampler(**transform_args)

# ------------------------------
#         PROCESS
# ------------------------------
start = time.time()

print(f"Extracting flowpose samples with a dilation of {parsed.dilation}")

extract_data(arg,
             process_number=process_number,
             transforms=flowPoseTransform,
             modality='flowpose',
             debug=debug)

print(f'Processing time: {time.time()-start:.2f}s')
