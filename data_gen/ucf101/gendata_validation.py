import os
import os.path as osp
import numpy as np
import yaml
from config.argclass import ArgClass

config_name = './config/ucf101/base.yaml'

arg = ArgClass(arg=config_name)

# Get the annotations
annotations = arg.feeder_args['labels']

print(f"Total original samples (in annotations): {len(list(annotations.keys()))}")

# Get the number of videos for each class
count_dict_ann = {}
for key in annotations.keys():
    try:
        count_dict_ann[key.split('/')[0]] += 1
    except KeyError:
        count_dict_ann[key.split('/')[0]] = 1


# Now loop over all the modalitys (may as well do this for all modalitys every time)
incomplete = {'flow': [], 'pose': [], 'poseoff': []}
for class_name, val in count_dict_ann.items():
    for modality in incomplete.keys():
        if len(os.listdir(f"./data/ucf101/{modality}/{class_name}")) < val:
            incomplete[modality].append(class_name)

# Check if any classes are incomplete
complete = True

# Pretty printing any classes that are incomplete
print('Extract test complete.\nIncomplete classes:')
for modality, num_incomplete_classes in incomplete.items():
    print(f'{modality}: {num_incomplete_classes}')
    print(f'\t{len(num_incomplete_classes)} classes are incomplete')
    if len(num_incomplete_classes) != 0:
        complete = False

# Count the number of samples removed
poseoff_path = arg.extractor['data_paths']['poseoff_path']
zero_samples=0
# Loop through poseoff files and determine which are all zeros
for class_name in os.listdir(poseoff_path):
    for sample_name in os.listdir(osp.join(poseoff_path, class_name)):
        sample_path = osp.join(poseoff_path, class_name, sample_name)
        sample = np.load(sample_path)
        # If video contains all zeros...
        if not sample.any():
            zero_samples+=1
            # Remove it from the annotations!
            annotations.pop(osp.join(class_name, sample_name.split('.')[0]))

print(f"Total non-zero samples: {len(list(annotations.keys()))}")
print(f"\t{zero_samples} removed")

# Save the modified annotations
with open(arg.feeder_args['label_path'], 'w') as f_out:
    # Save the modified data (it doesn't look as nice)
    yaml.dump(annotations, f_out, sort_keys=False, default_flow_style=False)

# print(f"{zero_samples} samples without any data removed from annotations")


# try:
#     # Remove the TMP directory if it is empty
#     if os.listdir('./TMP') == []:
#         os.rmdir('./TMP')
#         print("\nNo annotations to remove, clearning up `./TMP` directory")
#     else:
#         print("\nSome annotations to remove (likely no human pose detected...)")
#         remove_annotations = []
#         # If it is not empty, get the annotations that need to be removed
#         for class_file in os.listdir('./TMP'):
#             with open(f'./TMP/{class_file}', 'r') as f:
#                 list_of_files = f.readlines()
#                 for line in list_of_files:
#                     class_name = class_file.split('_')[-1].split('.')[0]
#                     remove_annotations.append(f'{class_name}/{line.strip()}')
#             print(f"\tRemoving {len(list_of_files)} for class: {class_name}")

#         # Remove the annotations from the ucf101_annotations.yaml file
#         with open(arg.feeder_args['label_path'], 'r') as f_in:
#             # Load the annotations data
#             annotations = yaml.safe_load(f_in)

#         # Remove the annotations
#         for ann in remove_annotations:
#             annotations.pop(f'{ann}')

#         # Save the modified annotations
#         with open(arg.feeder_args['label_path'], 'w') as f_out:
#             # Save the modified data (it doesn't look as nice)
#             yaml.dump(annotations, f_out, sort_keys=False)

#         print(f"\nRemoved {len(remove_annotations)} annotations from {arg.feeder_args['label_path']}")
# except FileNotFoundError:
#     print('No TMP directory found. Run command: \n\t> mkdir TMP')
