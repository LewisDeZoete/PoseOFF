import os
from data_gen.utils import LoadVideo, GetPoses_YOLO, extract_data
from config.argclass import ArgClass
import argparse
import time
from ultralytics import YOLO
import torch
import torchvision.transforms.v2 as v2
import numpy as np

parser = argparse.ArgumentParser(prog="pose_gendata")

parser.add_argument('-n', dest='number',
                    help='Class number for processing pose keypoints of a specific class')
parser.add_argument('--debug', action='store_true',
                    help='Debug mode to check the data generation process')
parsed = parser.parse_args()
process_number = int(parsed.number) # Get class number command line arg
debug = parsed.debug # Get debug mode command line arg

# Get the arg object and create the classes
arg = ArgClass(arg='./config/ucf101/base.yaml')
transform_args = arg.extractor['pose'] # grab transforms arg

# Get the device
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')

# Create pose detector
detector = YOLO(transform_args['weights'])
detector.to(device)
transforms = v2.Compose([
    LoadVideo(max_frames=300),
    v2.Resize(size=(384,640)), # YOLO pose has a minimum input image size
    v2.ToDtype(torch.float32),
    v2.Lambda(lambda x: x/255.0), # Normalises the image to [0,1]
    GetPoses_YOLO(detector=detector, max_frames=300, num_joints=17)
    ])


video_name = list(arg.feeder_args['labels'].keys())[500]
video_path = os.path.join('../Datasets/UCF-101/', video_name+'.avi')
data = transforms(video_path)
print(f'Data shape: {data.shape}')
print(data[:, 10, 5, 0])


# ------------------------------
#         PROCESS
# ------------------------------
start = time.time()

# get_poses(arg, process_number=process_number)
extract_data(arg,
             process_number=process_number,
             transforms=transforms,
             modality='pose',
             debug=debug)

print(f'Processing time: {time.time()-start:.2f}s')
