import torch

results = torch.load('./pretrained_models/ms-g3d_flow/flowpose_k3_t0.5.pt',
                     map_location='cpu')

print(f"Highest test accuracy: {100*max(results['results']['test accuracy']):0.2f}%")