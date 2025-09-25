import os
from data_gen.utils import LoadVideo, GetFlow, extract_data, ToNumpy
from config.argclass import ArgClass
import argparse
import time
from torchvision.models.optical_flow import raft_large
import torch
import torchvision.transforms.v2 as v2


parser = argparse.ArgumentParser(prog="flow_gendata")

parser.add_argument('-n', dest='number',
                    help='Class number for processing pose keypoints of a specific class')
parser.add_argument('--debug', action='store_true',
                    help='Debug mode to check the data generation process')
parsed = parser.parse_args()
process_number = int(parsed.number) # Get class number command line arg
debug = parsed.debug # Get debug mode command line arg

# Get the arg object and create the classes
arg = ArgClass(arg='./config/ucf101/base.yaml')
transform_args = arg.extractor['flow']

# Get the device
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

# Create the model, move it to device and turn to eval mode
weights = torch.load(transform_args['weights'], weights_only=True, map_location=device)
model = raft_large(progress=False)
model.load_state_dict(weights)
model = model.eval().to(device)
transforms = v2.Compose([
    LoadVideo(max_frames=300),
    v2.Resize(size=transform_args['imsize']),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # map [0, 1] into [-1, 1]
    GetFlow(model=model, device=device, minibatch_size=transform_args['minibatch_size']),
    ToNumpy()
    ])

# Ensure the data_paths['flow_path'] exists
os.makedirs(arg.extractor['data_paths']['flow_path'], exist_ok=True)
print(f'Ensured flow path exists: {arg.extractor["data_paths"]["flow_path"]}')


# ------------------------------
#         PROCESS
# ------------------------------
start = time.time()

extract_data(arg, 
             process_number=process_number, 
             transforms=transforms, 
             modality='flow',
             debug=debug)

print(f'Processing time: {time.time()-start:.2f}s')
