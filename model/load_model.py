import sys
sys.path.insert(0, '')

import torch
from collections import OrderedDict
from lib.utils.model_utils import count_params, import_class
from lib.utils.objects import LayerCompare

class ModelLoader:
    def __init__(self, arg):
        self.arg = arg
        # If cuda isn't available device is cpu, print device name debug
        if not torch.cuda.is_available():
            output_device='cpu'
        else: 
            output_device = self.arg.device[0] if type(
                self.arg.device) is list else self.arg.device
        
        self.output_device = output_device
        print(f'\tOutput device: {self.output_device}')


    def load_checkpoint_weights(self):
        try:
            # checkpoint_file attr will stay as string
            checkpoint = torch.load(self.arg.checkpoint_file, map_location=self.output_device)
            weights = checkpoint['model_state_dict'] # we just want state dict!
            return weights
        except FileNotFoundError as error:
            print(f"\tCheckpoint file does not yet exist")
            print(f'\t({error})')
            return None
        except KeyError:
            print(f'Checkpoint file {self.arg.checkpoint_file} does not contain a model state dict')
            return None


    def load_model(self):
        '''Load the model, using the arguments passed in `arg.model_args`
        If `arg.weights` exists, it will try to load the weights from the file passed.
        '''
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).to(self.output_device)
        print(f'\tModel total number of params: {count_params(self.model)}')

        #TODO: Turn this into a method to load a state dict from checkpoint
        # If checkpoint file is found, try and load the model_state_dict
        if hasattr(self.arg, 'checkpoint_file'):
            print(f'\tLoading weights from {self.arg.checkpoint_file}')
            weights = self.load_checkpoint_weights()
            # load_checkpoint_weights returns None if there is no weights in file
            if weights:
                # removing the 'module.' part of key name and moving weight to device
                weights = OrderedDict(
                    [[k.split('module.')[-1],
                    v.to(self.output_device)] for k, v in weights.items()])

                # Check if the loaded pretrained weights match the model's
                compatible, mis_keys = LayerCompare(self.model.state_dict(), weights)

                # Removing / ignoring weights (both from args and mismatched shape weights)
                remove_weights = [key for key in weights.keys() 
                            if any(weight_name in key for weight_name in self.arg.ignore_weights)]
                remove_weights.extend(mis_keys)
                for w in remove_weights:
                    if weights.pop(w, None) is None:
                        print(f'\t\tWeight not found: {w}')
                print(f'\tTotal number of weights removed: {len(remove_weights)}')

                try:
                    # Strict must be false given we are missing some weights
                    # in the pretrained state dictionary
                    self.model.load_state_dict(weights, strict=False)
                    print('\tPretrained model loaded')
                except RuntimeError:
                    # If we didn't put the correct weights in the `ignore_weights`
                    # section of config dict, we can remove those weights from state dict
                    state = self.model.state_dict()
                    diff = list(set(state.keys()).difference(set(weights.keys())))
                    print('\tCan not find these weights:')
                    for d in diff:
                        print('\t  ' + d)


if __name__=='__main__':
    from lib.utils.objects import ArgClass
    import time
    
    arg = ArgClass('./config/custom_pose/train_joint.yaml')
    # This checkpoint may not fit the correct keys and shape to be loaded but oh well
    # arg.checkpoint_file = 'results/ms-g3d_flow/flowpose-cnn_k5_t0.05.pt'
    arg.checkpoint_file = 'results/ms-g3d_flow/jiden.pt'
    
    # checkpoint = torch.load(arg.checkpoint_file, map_location=torch.device('cpu'))
    

    modelLoader = ModelLoader(arg)
    modelLoader.load_model()
    skel_model = modelLoader.model

    in_channels = arg.model_args['in_channels'] + 2*arg.model_args['flow_window']**2

    with torch.no_grad():
        start = time.time()
        b = 6
        x = torch.randn(b,in_channels,300,17,2).to(modelLoader.output_device)
        results = skel_model(x)
        print(f'\nOutput shape (batch size = {b}): {results.shape}')
        print(f'in {time.time()-start:0.5f} seconds')