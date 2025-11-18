import matplotlib.pyplot as plt
from config import ArgClass
from model import ModelLoader

arg = ArgClass('config/ucf101/train_joint_infogcn.yaml')
arg.checkpoint_file = 'results/infogcn2/infogcn_abs_window_mean_flow.pt'
# arg.checkpoint_file = 'results/infogcn2/infogcn_average_flow.pt'

model = ModelLoader(arg).model

# Assuming `model` is your PyTorch model and `layer_name` is the name of the Linear layer
layer_name = 'to_joint_embedding'  # Replace with your layer name
weights = getattr(model, layer_name).weight.data.cpu().numpy()
# weights = getattr(model, layer_name).data.cpu().numpy().squeeze(0)
# weights = weights.mean(axis=1).reshape(17, 1)
print(weights.shape)

plt.imshow(weights, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title(f'Weights of {layer_name}')
plt.xlabel('Input Channels')
plt.ylabel('Output Features')
plt.savefig(f'{layer_name}.png')

