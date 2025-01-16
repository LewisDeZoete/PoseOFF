import yaml
from deepdiff import DeepDiff

def compare_yaml_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        yaml1 = yaml.safe_load(f1)
        yaml2 = yaml.safe_load(f2)
    
    differences = DeepDiff(yaml1, yaml2, ignore_order=True)
    return differences

# Example usage
file1 = './ucf101_annotations.yaml'
file2 = '../Datasets/UCF-101/ucf101_annotations.yaml'
diffs = compare_yaml_files(file1, file2)
print(diffs)