import torch
from model import ModelLoader
import os
import os.path as osp
from config.argclass import ArgClass
from torch.utils.data import DataLoader
import numpy as np

model_type = 'base'
dataset = 'nturgbd'
evaluation = 'CS'
# dataset = 'ucf101'
# evaluation = '1'

# TODO: move this file into the data or model visualisations 
# create the path to save the visualiations 
save_path = f'TMP/{dataset}/{evaluation}/{model_type}'
os.makedirs(save_path, exist_ok=True)

# Set up arg class
arg = ArgClass(f"config/{dataset}/train_{model_type}.yaml")
checkpoint_file = osp.join(arg.save_location,
                            evaluation,
                            f'{dataset}_{evaluation}_{model_type}.pt')
arg.checkpoint_file = checkpoint_file
arg.evaluation=evaluation

# Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arg.model_args["device"] = device

# Load the model
model_loader = ModelLoader(arg)
model = model_loader.model
model.to(device)

# Create the directory to save things in
os.makedirs('TMP', exist_ok=True)

# load the data
if dataset == 'nturgbd':
    arg.feeder_args['data_paths'][evaluation] = f'./data/ntu/aligned_data/MINI_{evaluation}_flowpose.npz'
    feeder_class = arg.import_class(arg.feeder)
    train_dataset = feeder_class(**arg.feeder_args, eval=arg.evaluation, split="train")
    test_dataset = feeder_class(**arg.feeder_args, eval=arg.evaluation, split="test")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=60,
        shuffle=False,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=60,
        shuffle=False,
        pin_memory=True,
    )
    # # Process the data (two batches only)
    with torch.no_grad():
        for batch_no, (data, label, mask, index) in enumerate(train_dataloader):
            data = data.float().to(device) # (B, C, T, V, M)
            if model_type == 'base':
                data = data[:, :3, ...]
            label = label.to(device)
            
            y_hat, x_hat, z_0, z_hat_shifted, zero = model(data)

            attentions = model.get_attention()
else:
    data = np.load('data/UCF-101/flowpose/SoccerJuggling/v_SoccerJuggling_g09_c01.npy')
    # is_zero = (data == 0).all(axis=(0, 2, 3))  # shape: (300)
    # padding_start = np.array([np.argmax(z) if np.any(z) else data.shape[0] for z in is_zero])
    # first_padding = padding_start.max()
    # print(first_padding)

    # Reshape to remove padded frames
    data = torch.tensor(np.expand_dims(data, axis=0))
    data = data.float().to(device)
    if model_type == 'base':
        data = data[:, :3, :64, ...] # (1, 3, 64, 17, 2)
    else: 
        data = data[:, :, :64, ...] # (1, 33, 64, 17, 2)
    with torch.no_grad():
        y_hat, x_hat, z_0, z_hat_shifted, zero = model(data)
        attentions = model.get_attention()


if __name__ == "__main__":
    # Just for plotting after saving the tensors
    import torch
    import matplotlib.pyplot as plt
    from einops import rearrange
    import os

    # class_no = 0 # FOR UCF101
    class_no = 24 # kicking something

    # label0 = torch.load('TMP/label_0.pt')
    # label1 = torch.load('TMP/label_1.pt')
    # y_hat_0 = torch.load('TMP/y_hat_0.pt')
    # y_hat_1 = torch.load('TMP/y_hat_1.pt')

    sample = y_hat[class_no-1]
    plt.figure(figsize=(10, 10))
    plt.xlabel('Time (frames)')
    plt.ylabel('Class number')
    imgplot = plt.imshow(sample, cmap='gray')
    plt.title('Action class - Kicking something (label number 24)')
    # plt.title('Action class - Soccer juggling (label number 83)')
    plt.savefig(osp.join(save_path, 'cls_heads.png'))


    def plot_attn(attentions, batch_no=0, body_no=0, head=0):
        '''
        plots the attention matrix for a given layer (depth), 
        and attention head, with a particular ID (batch*joint*person entry)
        '''
        for layer in range(len(attentions)):
            attention = rearrange(attentions[layer], 
                                    '(N M V) H Q K -> N M V H Q K',
                                    N=data.shape[0],
                                    M=2, 
                                    V=arg.model_args['num_point'])
            attn_matrix = attention[batch_no, body_no].mean(0)
            attn_matrix = attn_matrix.mean(0)

            # # Or a bar plot?
            # plt.bar((torch.arange(64)+1), attn_matrix.mean(0))

            # plt.figure(figsize=(8, 8))
            # plt.title(f'Last encoding layer, Head {head}')
            # plt.xlabel('Key (Time)')
            # plt.ylabel('Query (Time)')
            # plt.imshow(attn_matrix.cpu().numpy(), cmap='gray', aspect='auto')
            # plt.colorbar(label='Attention Weight')
            # plt.tight_layout()
            # plt.savefig(osp.join(save_path, f'attn_L{layer}_H{head}.png'))

            # Attention over time
            attn_avg = attention[batch_no].mean(dim=(0, 1, 2))  # (query, key)
            saliency = attn_avg.sum(dim=0)    # sum over queries, shape: (key,)
            saliency = saliency/(torch.flip(torch.arandge(64), dims=[0]))
            plt.figure()
            plt.plot(saliency.cpu().numpy())
            plt.xlabel('Time Step')
            plt.ylabel('Total Attention Received')
            plt.title(f'Layer {layer+1} Attention Saliency Over Time')
            plt.savefig(osp.join(save_path, f'remove_attn_L{layer}'))

    plot_attn(attentions, batch_no=class_no)