import os
import yaml


with open('../Datasets/UCF-101/ucf101_annotations.yaml', 'r') as yaml_file:
    annotations = yaml.safe_load(yaml_file)

# Get the number of videos in the original videos
count_dict_ann = {}
for key in annotations.keys():
    try:
        count_dict_ann[key.split('/')[0]] += 1
    except KeyError:
        count_dict_ann[key.split('/')[0]] = 1

# count_dict_flow = {}
for key, val in count_dict_ann.items():
    # count_dict_flow[key] = len(os.listdir(f'./data/UCF-101/flow/{key}'))
    if len(os.listdir(f'./data/UCF-101/flow/{key}')) < val:
        print(key)