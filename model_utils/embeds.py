import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class Flow_conv(nn.Module):
    """
    A PyTorch module for learning flow windows using 3D convolutions.
    Args:
        kernel_size (int or tuple): Size of the convolving kernel.
        flow_window (int, optional): Size of the flow window. Default is 5.
        pose_channels (int, optional): Number of original pose channels in the input data. Default is 3.
        out_channels (int, optional): Number of output channels after the final linear layer. Default is 4.
    Attributes:
        kernel_size (int or tuple): Size of the convolving kernel.
        pose_channels (int): Number of original pose channels in the input data.
        out_channels (int): Number of output channels after the final linear layer.
        flow_window (int): Size of the flow window.
        conv (nn.Sequential): Sequential container of convolutional layers and activation functions.
        fc (nn.Linear): Fully connected layer to reduce the number of output channels.
    Methods:
        forward(x):
            Forward pass of the module. Takes input tensor `x` of shape (N*M*T, V, C) and returns a tensor of the same shape with additional flow features concatenated.
            Args:
                x (torch.Tensor): Input tensor of shape (N*M*T, V, C).
            Returns:
                torch.Tensor: Output tensor of shape (N, T, M, V, C + out_channels - pose_channels).
    """

    def __init__(
        self, kernel_size, flow_window=5, pose_channels=3, out_channels=64
    ):
        super(Flow_conv, self).__init__()

        # 3D conv for learning the flow windows
        # in_channels = 2 since flow is x and y motion channels
        # out_channels is the output shape of the convolutions
        self.kernel_size = kernel_size
        self.pose_channels = pose_channels
        self.out_channels = out_channels
        self.flow_out_channels = out_channels - pose_channels
        self.flow_window = flow_window
        # self.bn = nn.BatchNorm1d(2*(flow_window**2))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, self.flow_out_channels) # For the flow channels
        # self.lin = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, x):
        A, V, C = x.size()  # A=N*M*T
        # x of shape ((batch, 2, 300), 17, pose_channels+2W**2)
        # x = rearrange(x, "n c t v m -> (n m t) v c")
        flow_data = rearrange(x, "A v c -> (A v) c", A=A, v=V, c=C)  # (N*M*T*V, C)
        # flow_data = self.bn(flow_data[:, self.pose_channels:]) # apply batchnorm
        flow_data = rearrange(
            flow_data[:, self.pose_channels:],
            "skels (w h c) -> skels c w h",
            w=self.flow_window,
            h=self.flow_window,
            c=2,
        )  # (N*M*T*V, W, H, 2)

        # Apply convolutions, dropout between then flatten
        flow_features = self.conv(flow_data)  # Apply conv
        flow_features = flow_features.view(flow_features.size(0), -1)  # flatten
        flow_features = self.fc(flow_features)  # Linear layer to reduce out_channels
        flow_features = rearrange(flow_features, "(A v) c -> (A) v c", A=A, v=V)

        # Stack the output of this cnn onto the original graph features
        x = torch.cat((x[:, :, : self.pose_channels], flow_features), dim=-1)
        # x = self.lin(x) # connect the flow and pose channels

        return x


class NodeAttention(nn.Module):
    def __init__(self, in_features_skeleton, in_features_flow, out_features):
        super(NodeAttention, self).__init__()
        # Linear layers to project skeleton and flow features to the same space
        self.skeleton_proj = nn.Linear(in_features_skeleton, out_features)
        self.flow_proj = nn.Linear(in_features_flow, out_features)

        # Need to know shape of features... assuming first 'n' channels are skel, the rest are flow
        self.in_features_skeleton = in_features_skeleton

        # Attention mechanism
        self.attention = nn.Linear(out_features * 2, 1)  # Compute attention score

    def forward(self, x):
        N, T, M, V, C = x.size()
        x = x.view(N, T * M * V, C)
        # x of shape (batch, 300, 2, 17, channels+2W**2)
        skel_feat = x[
            ..., : self.in_features_skeleton
        ]  # Shape: (N*T*M*V, skeleton_features)
        flow_feat = x[..., self.in_features_skeleton :]  # Shape: (N*T*M*V, 2W**2)

        # Project features to the same dimensionality
        skel_feat = self.skeleton_proj(skel_feat)  # Shape: (N*T*M*V, out_features)
        flow_feat = self.flow_proj(flow_feat)  # Shape: (N*T*M*V, out_features)

        # Compute attention scores
        combined = torch.cat([skel_feat, flow_feat], dim=-1)  # Concatenate features
        attention_scores = F.softmax(
            self.attention(combined), dim=1
        )  # Shape: (batch, nodes, 1)

        # Weighted sum of features
        attended_features = (
            attention_scores * skel_feat + (1 - attention_scores) * flow_feat
        )
        attended_features = attended_features.view(
            N, T, M, V, -1
        )  # Shape: (N, T, M, V, out_features)
        return attended_features


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim, num_heads=8):
        super(TemporalAttention, self).__init__()
        assert attention_dim % num_heads == 0, (
            "Attention dim must be divisible by number of heads"
        )
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        self.query = nn.Linear(feature_dim, attention_dim)
        self.key = nn.Linear(feature_dim, attention_dim)
        self.value = nn.Linear(feature_dim, attention_dim)
        self.output_layer = nn.Linear(attention_dim, feature_dim)

    def forward(self, x):
        # x: input of shape (N*M, C, T, V)
        N, C, T, V = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().view(N * V, T, C)

        # Project to query, key, value spaces
        query = self.query(x)  # (N*V, T, attention_dim)
        key = self.key(x)  # (N*V, T, attention_dim)
        value = self.value(x)  # (N*V, T, attention_dim)

        # Split into multiple heads and reshape
        query = query.view(N * V, T, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (N*V, num_heads, T, head_dim)
        key = key.view(N * V, T, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (N*V, num_heads, T, head_dim)
        value = value.view(N * V, T, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (N*V, num_heads, T, head_dim)

        # Compute scaled dot-product attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # (N*M*V, T, T)
        # attention_scores /= torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32, device=x.device))
        attention_scores /= torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32, device=x.device)
        )

        # Normalize attention scores with softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # (N*M*V, T, T)

        # Compute attention output
        attended_values = torch.matmul(
            attention_weights, value
        )  # (N*M*V, T, attention_dim)

        # # Project back to original channel dimensions
        # attended_values = self.output_layer(attended_values)  # (N*M*V, T, C)
        # Concatenate multiple heads and project back to original dimensions
        attended_values = attended_values.permute(0, 2, 1, 3).reshape(
            N * V, T, -1
        )  # (N * V, T, attention_dim)
        attended_values = self.output_layer(attended_values)  # (N * V, T, C)

        # Reshape back to (N*M, C, T, V)
        attended_values = (
            attended_values.view(N, V, T, C).permute(0, 3, 2, 1).contiguous()
        )  # (N, C, T, V)

        return attended_values


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        # Shape: (1, max_length, d_model)

    def forward(self, x):
        # x: (N*M*V, T, C))
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=300,
    ):
        super(TemporalTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Ensures input format is (N*M, T, c2)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, x):
        # TODO: ensure the comments here reflect the correct input shape
        # x: (N*M, T, c2)
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Output shape: (N*M, T, c2)
        x = x.mean(dim=1)  # Pool over the temporal dimension (T)
        return x


if __name__ == "__main__":
    from graph.yolo_pose import AdjMatrixGraph

    graph = AdjMatrixGraph()
    A_binary = graph.A_binary

    kernel_size = 3
    flow_window = 5
    pose_channels = 3
    embed_channels = 64

    # Example usage
    N, C, T, V, M = 16, (pose_channels + 2 * (flow_window**2)), 300, 17, 2

    # Dummy data
    x = torch.randn(N, C, T, V, M)
    x = rearrange(x, "n c t v m -> (n m t) v c")

    # Define two potential joint embedding methods
    lin = nn.Linear(pose_channels + 2 * (flow_window**2), 64)
    flow = Flow_conv(
        kernel_size=kernel_size,
        flow_window=flow_window,
        pose_channels=pose_channels,
        out_channels=embed_channels,
    )

    # Pass data to joint embeddings
    flows = flow(x)
    lins = lin(x)

    # Dummy positional embedding (1, V, embed_channels)
    pos_embedding = torch.randn(1, V, embed_channels)

    print(f"Input shape: {x.shape} (N: {N}, C: {C}, T: {T}, V: {V}, M: {M})")
    print(f"Expected output shape: ({N * M * T}, {V}, {embed_channels})")
    print(f"Flow embeddings: {(flows + pos_embedding[:, :V]).shape}")
    print(f"Linear embeddings: {(lins + pos_embedding[:, :V]).shape}")
