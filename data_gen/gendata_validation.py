import sys
import os
import yaml

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

from config.argclass import ArgClass

config_name = './config/custom_pose/train_joint.yaml'

arg = ArgClass(arg=config_name)

# Get the annotations
annotations = arg.feeder_args['labels']

# Get the number of videos for each class
count_dict_ann = {}
for key in annotations.keys():
    try:
        count_dict_ann[key.split('/')[0]] += 1
    except KeyError:
        count_dict_ann[key.split('/')[0]] = 1

# Now loop over all the streams (may as well do this for all streams every time)
incomplete = {'flow': [], 'pose': [], 'flowpose': []}
for class_name, val in count_dict_ann.items():
    for stream in incomplete.keys():
        if len(os.listdir(f'./data/UCF-101/{stream}/{class_name}')) < val:
            incomplete[stream].append(class_name)

# Check if any classes are incomplete
complete = True

# Pretty printing any classes that are incomplete
print('Extract test complete.\nIncomplete classes:')
for key, val in incomplete.items():
    print(f'{key}: {val}')
    if len(val) != 0:
        complete = False

# If there are no incomplete classes, change the config file to reflect that
if complete:
    with open(config_name, 'r') as f_in:
        # Load the original data
        yaml_data = yaml.safe_load(f_in)

    # Modify the data
    yaml_data['dataloader']['preprocessed'] = True

    with open(f'{config_name[:-5]}_MODIFIED.yaml', 'w') as f_out:
        # Save the modified data (it doesn't look as nice)
        yaml.dump(yaml_data, f_out, sort_keys=False)