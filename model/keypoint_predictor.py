import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
from model.msg3d import Model as base_model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class MS_G3D_KeypointPredictor(nn.Module):
    def __init__(self, base_model, num_keypoints, T_future, input_dim=3):
        super(MS_G3D_KeypointPredictor, self).__init__()
        self.base_model = base_model  # Pretrained MS-G3D without the classification head
        self.num_keypoints = num_keypoints
        self.T_future = T_future      # base_model.feature_dim = 192 = c2
        self.output_layer = nn.Linear(192, T_future * num_keypoints * input_dim)

    def forward(self, x):
        # Extract spatio-temporal features using MS-G3D
        features = self.base_model(x)
        # Predict future keypoints
        output = self.output_layer(features)
        # Reshape output to (batch_size, T_future, num_keypoints, 3)
        output = output.view(x.size(0), self.T_future, self.num_keypoints, -1)
        return output


if __name__=='__main__':
    from lib.utils.objects import ArgClass
    
    # Get args and model args
    arg = ArgClass('config/custom_pose/train_joint.yaml')
    model_args = arg.model_args

    # Check number of output keypoints necessary
    num_keypoints = model_args['num_point']*model_args['num_person']

    # Example usage
    base = base_model(**arg.model_args) # load a pretrained model (may improve results)
    base.fc = Identity()
    prediction_model = MS_G3D_KeypointPredictor(base_model=base, num_keypoints=num_keypoints, T_future=20, input_dim=3)

    # Dummy input for testing
    feats = model_args['in_channels'] + 2*model_args['flow_window']**2
    x = torch.randn(1, feats, 300, model_args['num_point'], model_args['num_person'])  # Example input with (num_features, num_frames, num_keypoints, num_people)

    # Forward pass
    output = prediction_model(x)
    print("Output shape:", output.shape)
    
    # Testing loss
    loss = torch.nn.MSELoss()
    target = torch.randn(1,3,20,17,2)
    output = output.view(1,20,17,2,3).permute(0,4,1,2,3)
    print(f"Output (batch, (x,y,vis), predicted_frames, keypoints, people): {output.shape}")