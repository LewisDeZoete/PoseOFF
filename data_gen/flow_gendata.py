import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

from lib.data.dataset import CustomVideoDataset
from lib.utils.objects import ArgClass
from lib.utils.transforms import GetFlow
import torch
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.v2 as v2
import argparse

parser = argparse.ArgumentParser(prog="flow_gendata")

parser.add_argument('-n', dest='number',
                    help='Class number for processing flow of a specific class')
parsed = parser.parse_args()
# Get the command line argument given for the class number (0-100)
arg_no = int(parsed.number)

# Get the arg object and create the classes
arg = ArgClass(arg='./data_gen/UCF-101_config.yaml')
classes = arg.get_classes()

# Get the number of videos in the class (used to get indices of dataset)
def get_range(class_no):
    len_class = 0
    for i in arg.labels.keys():
        if i.split('/')[0] == list(classes.keys())[class_no]:
            try:
                assert start_index >= 0
            except NameError:
                start_index = list(arg.labels.keys()).index(i)
            len_class += 1
    return range(start_index, (start_index+len_class))

# Get the device
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

# Create the model, move it to device and turn to eval mode
weights = torch.load(arg.flow['weights'], weights_only=True, map_location=device)
model = raft_large(progress=False)
model.load_state_dict(weights)
model = model.eval()

transforms = v2.Compose([
    # v2.Resize(size=(240,320)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # map [0, 1] into [-1, 1]
    GetFlow(model=model, device=device, minibatch_size=8)
    ])

# Create the dataset object
dataset = CustomVideoDataset(arg=arg, transforms=transforms)

for idx in get_range(arg_no):
    preprocessed_video, label = dataset[idx]
    print(f'Class: {list(classes.keys())[label]}')
    print(f'Shape: {preprocessed_video.shape}')
    print(f'Max: {preprocessed_video.max()}')
    print(f'Min: {preprocessed_video.min()}')
    break

# # Create a results list, then without calculating gradients process flow
# results = []
# torch.cuda.empty_cache()
# batch_size = 8
# results = []
# with torch.no_grad():
#     for i in range(0, stacked.shape[0], batch_size):
#         input_tensor = stacked[i:i+batch_size].to(device)

#         flow_list = model(input_tensor[:,0,...], input_tensor[:,1,...])
#         results.append(flow_list[-1])
#         print(f'iteration {i+batch_size} done')
#         torch.cuda.empty_cache()

# print(torch.cat(results).shape)

# for idx in get_range(arg_no):
#     folder = f'data/UCF-101/flow/{list(arg.labels.keys())[idx].split("/")[0]}/'
#     path = os.path.join(folder, list(arg.labels.keys())[idx].split('/')[-1].split('.')[0] + '.pt')