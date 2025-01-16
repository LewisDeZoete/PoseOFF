import yaml
import argparse
import torch

# TODO: Move to config
class ArgClass(object):
    def __init__(self, arg, verbose=False):
        '''
        Takes argument of an argparse object or a string that gives the path to a config doctionary. Converts the first layer of dicionary keys into properties of this class object instance.

        Args:
            arg (str | argparse.Namespace | dict): If `arg` is an `argparse.Namespace` object that contains the config type, phase and limb (bone/joint), or a `str`, load corresponding yaml file and convert to the config dictionary object with one level of keys as attributes. If input is a `dict`, convert directly to object with one level of keys as attributes.
            verbose (bool): If True, print the attributes and their values as they are being set (prior to classes and labels so it doesn't get messy).
        '''
        # as an argparse object
        if isinstance(arg, argparse.Namespace):
            # Get arg file
            with open(f'./config/{arg.config}/{arg.phase}_{arg.limb}.yaml', 'r') as file:
                in_dict = yaml.safe_load(file)
            # Also add the attributes in parser to the arg class
            for key in arg.__dict__:
                in_dict[key] = arg.__dict__[key]
        elif isinstance(arg, str):
            with open(arg, 'r') as file:
                in_dict = yaml.safe_load(file)
        elif isinstance(arg, dict):
            in_dict = arg
        
        for key in in_dict:
            setattr(self, key, in_dict[key])
            if verbose:
                if isinstance(in_dict[key], dict):
                    # Print subkeys, only 1 level deep
                    print(f'{key}:')
                    for subkey in in_dict[key]:
                        print(f'  {subkey}: {in_dict[key][subkey]}')
                else:
                    print(f'{key}: {in_dict[key]}')
    
        # Check if the ArgClass object even has the dataloader object
        assert hasattr(self, 'dataloader'), "Input object had no key 'dataloader'"
        assert 'label_path' in self.dataloader, "Input object has no key 'label_path' under 'dataloader'"

        # If the label_path is wrong, it'll throw FileNotFound error
        try:
            with open(self.dataloader['label_path'], 'r') as file:
                self.labels = yaml.safe_load(file)

        except FileNotFoundError:
            print(f"Could not file label file: '{self.dataloader['label_path']}'")
            self.labels = {}
        
        self.classes = {}
        for elem, key in enumerate(dict.fromkeys(key.split('_')[1] for key in self.labels.keys())):
            self.classes[key] = elem



# TODO: move to model/utils.py
def LayerCompare(dict1, dict2):
    """
    Compare the shapes of tensors in two dictionaries with the same keys.

    Args:
        dict1 (dict): The first dictionary with tensor values.
        dict2 (dict): The second dictionary with tensor values.

    Returns:
        bool: True if all shapes match, False otherwise.
        list: A list of keys where the shapes do not match.
    """
    # Check if both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        raise ValueError("Dictionaries have different keys.")

    mismatched_keys = []

    for key in dict1.keys():
        tensor1 = dict1[key]
        tensor2 = dict2[key]

        # Check if both values are tensors
        if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
            raise ValueError(f"Values for key '{key}' are not both tensors.")

        # Compare shapes
        if tensor1.shape != tensor2.shape:
            mismatched_keys.append(key)

    # Return the result
    return len(mismatched_keys) == 0, mismatched_keys


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='config', default='custom_pose',
                        help='config dictionary location (default=custom_pose)')
    parser.add_argument('-p', dest='phase', default='test',
                        help='network phase [train, test] (default=test)')
    parser.add_argument('-l', dest='limb', default='joint',
                        help='limb [joint, bone] (default=joint)')
    parser.add_argument('-s', dest='save_name', default='',
                        help='name to save the results dictionary as after training')
    parsed = parser.parse_args()
    
    results = {}
    results['name'] = parsed.save_name
    results['epoch'] = [0,1,2]
    results['total_time'] = [0.1, 0.9, 0.5]
    
    with open(f'./{parsed.save_name}.yaml', 'w') as outfile:
        yaml.safe_dump(results, outfile, default_flow_style=False, sort_keys=False)