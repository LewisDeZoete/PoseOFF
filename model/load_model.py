import sys
sys.path.insert(0, '')

import pickle
import torch
import torch.nn as nn
from collections import OrderedDict
from lib.utils.model_utils import count_params, import_class
from lib.utils.objects import LayerCompare

class ModelLoader:
    def __init__(self, arg):
        self.arg = arg
        
    def load_model(self):
        # If cuda isn't available device is cpu, print device name debug
        if not torch.cuda.is_available():
            output_device='cpu'
        else: 
            output_device = self.arg.device[0] if type(
                self.arg.device) is list else self.arg.device
        
        self.output_device = output_device
        print(f'\tOutput device: {self.output_device}')
        Model = import_class(self.arg.model)

        # self.model = Model(**self.arg.model_args).cuda(output_device)
        self.model = Model(**self.arg.model_args).to(output_device)
        # self.loss = nn.CrossEntropyLoss().to(output_device)
        print(f'\tModel total number of params: {count_params(self.model)}')

        # If weight parameter is passed, try and get the pretrained model
        if self.arg.weights:
            try:
                self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            except AttributeError:
                print('\tCannot parse global_step from model weights filename')
                self.global_step = 0

            print(f'\tLoading weights from {self.arg.weights}')
            weights = torch.load(self.arg.weights)

            # removing the 'module.' part of key name and moving weight to device
            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.to(output_device)] for k, v in weights.items()])

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
    
    in_channels = 3

    args = ArgClass('./config/custom_pose/train_joint.yaml')
    args.model_args['in_channels'] = in_channels
    modelLoader = ModelLoader(args)
    modelLoader.load_model()
    skel_model = modelLoader.model

    with torch.no_grad():
        start = time.time()
        b = 6
        x = torch.randn(b,in_channels,300,17,2).to(modelLoader.output_device)
        results = skel_model(x)
        print(f'\nOutput shape (batch size = {b}): {results.shape}')
        print(f'in {time.time()-start:0.5f} seconds')

    # torch.save({
    #     'epoch': 0,
    #     'model_state_dict': skel_model.state_dict(),
    #     'results': results
    # }, './pretrained_models/ms-g3d_flow/test.pt')

    # checkpoint = torch.load('./pretrained_models/ms-g3d_flow/test.pt')