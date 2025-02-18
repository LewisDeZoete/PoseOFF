import sys
import os
import yaml

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '../..')))

from config.argclass import ArgClass

config_name = './config/custom_pose/train_joint_infogcn.yaml'

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
    print(f'\t{len(val)} classes are incomplete')
    if len(val) != 0:
        complete = False

# If there are no incomplete classes, change the config file to reflect that
if complete:
    with open(config_name, 'r') as f_in:
        # Load the original data
        yaml_data = yaml.safe_load(f_in)

    # Modify the data
    yaml_data['feeder_args']['preprocessed'] = True

    # TODO: Remove the MODIFIED portion if you don't need the comments anymore...
    with open(f'{config_name[:-5]}_MODIFIED.yaml', 'w') as f_out:
        # Save the modified data (it doesn't look as nice)
        yaml.dump(yaml_data, f_out, sort_keys=False)

try:
    # Remove the TMP directory if it is empty
    if os.listdir('./TMP') == []:
        os.rmdir('./TMP')
        print('Removed TMP directory')
    else:
        remove_annotations = []
        # If it is not empty, get the annotations that need to be removed
        for class_file in os.listdir('./TMP'):
            with open(f'./TMP/{class_file}', 'r') as f:
                list_of_files = f.readlines()
                for line in list_of_files:
                    class_name = class_file.split('_')[-1].split('.')[0]
                    remove_annotations.append(f'{class_name}/{line.strip()}')

        # Remove the annotations from the ucf101_annotations.yaml file
        with open(arg.feeder_args['label_path'], 'r') as f_in:
            # Load the annotations data
            annotations = yaml.safe_load(f_in)

        # Remove the annotations
        for ann in remove_annotations:
            annotations.pop(f'{ann}')

        # Save the modified annotations
        with open(arg.feeder_args['label_path'], 'w') as f_out:
            # Save the modified data (it doesn't look as nice)
            yaml.dump(annotations, f_out, sort_keys=False)
        
        print(f'Removed {len(remove_annotations)} annotations from {arg.feeder_args["label_path"]}')
except FileNotFoundError:
    print('No TMP directory found. Run command: \n\t> mkdir TMP')