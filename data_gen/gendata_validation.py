import sys
import os
import yaml

# # add lib to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..')))

# Get the annotations
with open('../Datasets/UCF-101/ucf101_annotations.yaml', 'r') as yaml_file:
    annotations = yaml.safe_load(yaml_file)

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

# Pretty printing any classes that are incomplete
print('Extract test complete.\nIncomplete classes:')
for key, val in incomplete.items():
    print(f'{key}: {val}')