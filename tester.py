from lib.utils.objects import ArgClass
from model import load_model

import torch
import torchsummary

arg = ArgClass('./config/custom_pose/train_joint.yaml')
skel_model = load_model(arg)
skel_model.load_model()
model = skel_model.model

# # This method wasn't working because the model still has these attributes
# for key in ['fc', 'tcn3', 'sgcn3', 'gcn3d3']:
#     skel_model.model._modules.pop(key)
# print(skel_model.model._modules.keys())
print(model.__dict__.keys())

torchsummary.summary(model, (3,300,17,2))


# b = 6
# x = torch.randn(b,3,300,17,2).to(skel_model.output_device)

# results = skel_model.model(x)

# print(f'\nOutput shape (batch size = {b}): {results.shape}')