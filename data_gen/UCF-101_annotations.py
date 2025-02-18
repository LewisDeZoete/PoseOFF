import sys
import os

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

import os
import yaml
from config.argclass import ArgClass

# Get the arg class
arg = ArgClass(arg='./config/custom_pose/train_joint_infogcn.yaml')

annotations = {} # temporary annotations dictionary

# Get the path to the videos
rgb_path = arg.feeder_args['data_paths']['rgb_path']

# Get the classes (folders in data_path, not files just to be sure)
classes = [ name for name in os.listdir(rgb_path) \
           if os.path.isdir(os.path.join(rgb_path, name))]
classes = sorted(classes) # Sort the classes

# Get the videos for each class
for class_number, class_name in enumerate(classes):
    class_path = os.path.join(rgb_path, class_name)
    videos = os.listdir(class_path)
    # Get the annotations for each video add to annotations dict
    for video in videos:
        annotations[os.path.join(class_name, video[:-4])] = class_number

print(f'Labels created for {len(annotations)} videos')

# Save the annotations to a yaml file (in arg.dataloader['label_path'])
with open(arg.feeder_args['label_path'], 'w') as yaml_file:
    yaml.dump(annotations, yaml_file)

print(f'Annotations saved to {arg.feeder_args["label_path"]}')