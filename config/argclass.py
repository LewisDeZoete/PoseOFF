import yaml
import argparse
import importlib

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
            # with open(f'./config/{arg.dataset}/{arg.model_type}.yaml', 'r') as file:
            with open(f'./config/{arg.model_type}/{arg.dataset}/{arg.flow_embedding}.yaml', 'r') as file:
                in_dict = yaml.safe_load(file)
            # Also add the attributes in parser to the arg class
            for key in arg.__dict__:
                in_dict[key] = arg.__dict__[key]
        elif isinstance(arg, str):
            with open(arg, 'r') as file:
                in_dict = yaml.safe_load(file)
        elif isinstance(arg, dict):
            in_dict = arg
        
        # Turn dict keys into attributes
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
    
        # Check if the ArgClass object even has the feeder args
        assert hasattr(self, 'feeder_args'), "Input object had no key 'feeder_args'"
        assert 'label_path' in self.feeder_args.keys(), "Input object has no key 'label_path' under 'feeder_args'"

        # If the label_path is wrong, it'll throw FileNotFound error
        try:
            with open(self.feeder_args['label_path'], 'r') as file:
                self.feeder_args['labels'] = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Could not file label file: '{self.feeder_args['label_path']}'")
            print('Creating empty labels dictionary')
            self.feeder_args['labels'] = {}
        
        # # Create classes dictionary
        # self.classes = {}
        # for elem, key in enumerate(dict.fromkeys(key.split('_')[-1] for key in self.feeder_args['labels'].keys())):
        #     self.classes[key] = elem
    
    # def import_class(self, name):
    #     components = name.split('.')
    #     mod = __import__(components[0])
    #     for comp in components[1:]:
    #         mod = getattr(mod, comp)
    #         return mod

    def import_class(self, path: str):
        """Dynamically import a class from a string path."""
        module_path, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", dest="model_type", default="infogcn2",
        help="Base model_type (e.g. infogcn2, msg3d, stgcn2)"
    )
    parser.add_argument(
        "-d", dest="dataset", default="ucf101",
        help="config dictionary location (default=ucf101)",
    )
    parser.add_argument(
        "-p", dest="phase", default="train",
        help="model phase (train/eval)"
    )
    parser.add_argument(
        "-f", dest="flow_embedding", default="base",
        help="model type [base, cnn, avg, abs] (default=base)"
    )
    parser.add_argument(
        "-e", dest="evaluation",
        help="Evaluation benchmark used for specific dataset \
            (eg. 1-3 for ucf101, CV/CS for NTU_RGB+D)"
    )
    parser.add_argument(
        "-v", dest="verbose", action="store_true", help="Print verbose output for argparse"
    )
    parsed = parser.parse_args()
    
    arg = ArgClass(parsed)

    feeder = arg.import_class(arg.feeder)
    model = arg.import_class(arg.model)
    print(arg.model)
    print(arg.feeder_args['data_paths'])
