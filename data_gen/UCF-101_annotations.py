import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

import os
import yaml
from lib.utils.objects import ArgClass

# Get the arg class
arg = ArgClass(arg='./config/custom_pose/train_joint.yaml')

annotations = {} # temporary annotations dictionary

# Get the classes (folders in data_path, not files just to be sure)
classes = [ name for name in os.listdir(arg.dataloader['data_path']) if os.path.isdir(os.path.join(arg.dataloader['data_path'], name)) ]
classes = sorted(classes) # Sort the classes

# Get the videos for each class
for class_number, class_name in enumerate(classes):
    class_path = os.path.join(arg.dataloader['data_path'], class_name)
    videos = os.listdir(class_path)
    # Get the annotations for each video add to annotations dict
    for video in videos:
        annotations[os.path.join(class_name, video[:-4])] = class_number

# Save the annotations to a yaml file (in arg.dataloader['label_path'])
with open(arg.dataloader['label_path'], 'w') as yaml_file:
    yaml.dump(annotations, yaml_file)