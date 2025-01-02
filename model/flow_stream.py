import torch
import torch.nn as nn
import torch.nn.functional as F

class Flow_conv(nn.Module):
    """
    A PyTorch module for learning flow windows using 3D convolutions.
    Args:
        kernel_size (int or tuple): Size of the convolving kernel.
        flow_window (int, optional): Size of the flow window. Default is 5.
        original_channels (int, optional): Number of original channels in the input data. Default is 3.
        out_channels (int, optional): Number of output channels after the final linear layer. Default is 4.
    Attributes:
        kernel_size (int or tuple): Size of the convolving kernel.
        original_channels (int): Number of original channels in the input data.
        out_channels (int): Number of output channels after the final linear layer.
        flow_window (int): Size of the flow window.
        conv (nn.Sequential): Sequential container of convolutional layers and activation functions.
        fc (nn.Linear): Fully connected layer to reduce the number of output channels.
    Methods:
        forward(x):
            Forward pass of the module. Takes input tensor `x` of shape (N, T, M, V, C) and returns a tensor of the same shape with additional flow features concatenated.
            Args:
                x (torch.Tensor): Input tensor of shape (N, T, M, V, C).
            Returns:
                torch.Tensor: Output tensor of shape (N, T, M, V, C + out_channels - original_channels).
    """
    def __init__(self, kernel_size, flow_window=5, original_channels=3, out_channels=4):
        
        super(Flow_conv, self).__init__()

        # 3D conv for learning the flow windows
        # in_channels = 2 since flow is x and y motion channels
        # out_channels is the output shape of the convolutions
        self.kernel_size = kernel_size
        self.original_channels = original_channels
        self.out_channels = out_channels
        self.flow_window = flow_window
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2,
                      out_channels=16,
                      kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32, 
                      kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, self.out_channels)

    def forward(self, x):
        #(N, T, M, V, C)
        N, T, M, V, C = x.size()
        # x of shape (batch, 300, 2, 17, channels+2W**2)
        flow_data = x.view(N*T*M*V, C)[:, self.original_channels:] # (N*T*M*V, 2W**2)
        flow_data = flow_data.view(N*T*M*V, self.flow_window, self.flow_window, 2).permute(0, 3, 1, 2)

        # Apply convolutions, dropout between then flatten 
        flow_features = self.conv(flow_data)    # Apply conv
        flow_features = flow_features.view(flow_features.size(0), -1) # flatten
        flow_features = self.fc(flow_features) # Linear layer to reduce out_channels
        flow_features = flow_features.view(N,T,M,V, -1)
        
        # Stack the output of this cnn onto the original graph features
        x = torch.cat((x[...,:self.original_channels], flow_features), dim=4)

        return x


class OpticalFlowStream(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_frames):
        super(OpticalFlowStream, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool3d((num_frames, 1, 1))  # Temporal aggregation
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)  # Shape: (batch_size, out_channels, num_frames)


class TemporalAttention(nn.Module):
    def __init__(self, in_channels, num_frames):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x: (batch_size, num_frames, in_channels)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        attention_weights = self.softmax(torch.matmul(queries, keys.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        attended_features = torch.matmul(attention_weights, values)
        return attended_features

if __name__ == "__main__":
    # Example usage
    N, T, M, V, C = 1, 300, 2, 17, 53
    x = torch.randn(N, T, M, V, C)
    flow_conv = Flow_conv(kernel_size=3, 
                          flow_window = 5,
                          original_channels=3, 
                          out_channels=4)
    x = flow_conv(x)
    print(x.shape)