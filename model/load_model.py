import sys
sys.path.insert(0, '')

import torch
from collections import OrderedDict
from lib.utils.model_utils import count_params, import_class
from model.infogcn2.utils import LayerCompare

class ModelLoader:
    def __init__(self, arg):
        self.arg = arg
        # If cuda isn't available device is cpu, print device name debug
        if not torch.cuda.is_available():
            output_device=torch.device('cpu')
        else: 
            output_device = torch.device(self.arg.device[0]) if type(
                self.arg.device) is list else torch.device(self.arg.device)
        
        self.output_device = output_device
        print(f'\tOutput device: {self.output_device}')

        # Load the model on object creation
        self.load_model()


    def load_model(self):
        '''
        Load the model, using the arguments passed in `arg.model_args`
        If `arg.weights` exists, it will try to load the weights from the file passed.
        '''
        Model = import_class(self.arg.model)
        # InfoGCN++ takes the device argument, we want it to be dynamic
        if 'device' in self.arg.model_args:
            self.arg.model_args['device'] = self.output_device
        self.model = Model(**self.arg.model_args).to(self.output_device)
        print(f'\tModel total number of params: {count_params(self.model)}')

        # If checkpoint file is found, try and load the model_state_dict
        if hasattr(self.arg, 'checkpoint_file'):
            print(f'\tLoading weights from {self.arg.checkpoint_file}')
            weights = self._get_weights()
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
    
    
    def _get_weights(self):
        try:
            # checkpoint_file attr will stay as string
            checkpoint = torch.load(self.arg.checkpoint_file, map_location=self.output_device)
            weights = checkpoint['model_state_dict'] # we just want state dict!
            return weights
        except FileNotFoundError as error:
            print("\tCheckpoint file does not yet exist")
            print(f'\t({error})')
            return None
        except KeyError:
            print(f'Checkpoint file {self.arg.checkpoint_file} does not contain a model state dict')
            return None


if __name__=='__main__':
    from lib.utils.objects import ArgClass
    import time
    
    arg = ArgClass('./config/custom_pose/train_joint_infogcn.yaml')
    # arg.checkpoint_file = 'results/ms-g3d_flow/flowpose-cnn_k5_t0.05.pt'
    arg.checkpoint_file = 'results/ms-g3d_flow/TMP.pt'

    # Load the model!
    modelLoader = ModelLoader(arg)
    skel_model = modelLoader.model

    with torch.no_grad():
        start = time.time()
        b = 8
        # N, C, T, V, M
        x = torch.randn((8, 3, 64, 17, 2)).to(modelLoader.output_device)
        out = skel_model(x) # tuple(y, x_hat, z_0, z_hat_shifted, self.zero)
        print(out[0].shape)
        print(f'\nOutput shape (batch size = {b}): {out[0].shape}')
        print(f'in {time.time()-start:0.5f} seconds')