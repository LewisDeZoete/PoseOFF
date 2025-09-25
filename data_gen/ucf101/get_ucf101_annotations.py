import os
import os.path as osp
import shutil
import yaml
import re
from config.argclass import ArgClass


def remove_bad_lines(lines):
    '''
    Change some of the class and video names in UCF-101 for consistency
    and remove the '.avi' from the video names.
    '''
    new_lines = []
    for line in lines:
        # Change these very specific class names
        if 'HandstandPushups' in line:
            line = line.replace('HandstandPushups', 'HandStandPushups')
        if 'HandstandWalking' in line:
            line = line.replace('HandstandWalking', 'HandStandWalking')
        # Remove both '.avi' and any class labels in the trainfiles
        new_lines.append(line.split('.')[0])
    return new_lines


def format_label_files(evaluation_lists_dir=None):
    '''
    Format the test and train lists that come with UCF-101
    '''
    assert evaluation_lists_dir is not None

    files = ['testlist01.txt', 'testlist02.txt', 'testlist03.txt', 'trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt']
    for file in files:
        with open(osp.join(evaluation_lists_dir, file), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        
        lines = remove_bad_lines(lines)

        with open(osp.join(stat_path, file), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

def annotations_yaml():
    annotations = {} # temporary annotations dictionary

    # Get the path to the videos
    rgb_path = arg.extractor['data_paths']['rgb_path']

    # Get the classes (folders in data_path, not files, just to be sure)
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


if __name__=='__main__':
    # First, make sure the data directories exist
    root_path = './data/ucf101'
    stat_path = osp.join(root_path, 'statistics')
    os.makedirs(stat_path, exist_ok=True)

    # Get the arg class
    arg = ArgClass(arg='./config/ucf101/base.yaml')

    # Now, generally the train/test lists will be downloaded with UCF-101...
    # Move them into the stats folder!
    for filename in os.listdir(arg.extractor["data_paths"]["rgb_path"]):
        if re.search(r"\d.txt$", filename): # check for filenames train/textlist0#.txt
            shutil.copy(
                osp.join(arg.extractor["data_paths"]["rgb_path"], filename),
                osp.join(stat_path, filename)
            )
    print(f"Train/test lists copied to: {stat_path}")

    # Get the annotations for the UCF-101 dataset
    annotations_yaml()

    # Format the label files that come with the UCF-101 datasets
    format_label_files(evaluation_lists_dir=stat_path)
