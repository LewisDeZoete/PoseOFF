import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.msg3d.ms_gcn import MultiScale_GraphConv as MS_GCN
from models.msg3d.ms_tcn import MultiScale_TemporalConv as MS_TCN
from models.msg3d.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from models.msg3d.mlp import MLP
from model_utils import (
    Flow_conv,
    NodeAttention,
    TemporalAttention,
    TemporalTransformer,
    import_class
)

from einops import rearrange, repeat

class MS_G3D_MODULE(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1]):

        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D_MODULE(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum


class MSG3D(nn.Module):
    """PoseOFF enabled implementation of MS-G3D."""
    def __init__(
            self,
            num_class,
            num_point,
            num_person,
            num_gcn_scales,
            num_g3d_scales,
            graph,
            pose_channels=3,
            flow_channels=50,
            cnn=False,
            **kwargs
        ):
        super().__init__()

        Graph = import_class(graph)()
        A_binary = Graph.A_binary

        self.data_bn = nn.BatchNorm1d(num_person *
                                     (pose_channels + flow_channels) *
                                      num_point)

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        if cnn:
            assert flow_channels > 2
            # Adding the new Flow_Windows module
            # Input to the rest of MS-G3D will simply be in_channels+1
            # self.flow_conv = Flow_conv(kernel_size=kernel_size,
            #                            flow_window = flow_window,
            #                            original_channels=in_channels,
            #                            out_channels=self.flow_channels)
            joint_embed_channels = 3
            self.to_joint_embedding = nn.Sequential(
                Flow_conv(
                    kernel_size=3,
                    flow_window=int(
                        math.sqrt(flow_channels / 2)
                    ),
                    pose_channels=pose_channels,
                    out_channels=joint_embed_channels,
                    ),
                nn.Linear(
                    joint_embed_channels, joint_embed_channels
                    )
            )
        else:
            joint_embed_channels = 3
            self.to_joint_embedding = nn.Linear(
                pose_channels + flow_channels, joint_embed_channels
            )

        self.gcn3d1 = MultiWindow_MS_G3D(joint_embed_channels, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            # MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_GCN(num_gcn_scales, joint_embed_channels, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        # self.fc = nn.Linear(c2, num_class)
        self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, "n c t v m -> n (m v c) t", n=N, c=C, t=T, m=M,v=V)
        x = self.data_bn(x)
        x = rearrange(x, "n (m v c) t -> (n m t) v c", n=N, c=C, t=T, m=M,v=V)

        # ------------------------------------------------------------
        # Joint embedding
        x = self.to_joint_embedding(x)
        # # (N * M, in_channels+flow_out_channels, T, V)
        x = rearrange(x, "(n m t) v c -> (n m) c t v", n=N, t=T, m=M, v=V)
        # ------------------------------------------------------------

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out


if __name__ == "__main__":
    from config.argclass import ArgClass
    import logging
    import os.path as osp
    from model_utils import ModelLoader

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename='logs/debug/models/msg3d.log',
        encoding='utf-8',
        filemode='w',
        level=logging.DEBUG
    )

    model_type="msg3d"
    dataset = "ucf101"
    flow_embedding = "cnn"

    # Get the config file and use the model arguments defined within
    arg = ArgClass(f'config/{model_type}/{dataset}/{flow_embedding}.yaml', verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arg.model_args["device"] = device

    # Create the model, set to train
    modelLoader = ModelLoader(arg)
    model = modelLoader.model

    # Create dummy input
    # N, C, T, V, M
    # Batch, channels, frames, keypoints, bodies
    C = arg.model_args["flow_channels"] + arg.model_args["pose_channels"]
    T = 64
    V = arg.model_args["num_point"]
    M = 2

    x = torch.randn((8, C, T, V, 2)).to(device)
    print(f"Data input shape: {x.shape}")
    print(f"N-classes: {arg.model_args['num_class']}")
    logger.info(f"Model: {flow_embedding}")
    logger.info(f"Input channels: {C}\n")
    logger.info(f"Input shape: {x.shape}\n    (B, C, T, V, M)")

    # Pass input to model
    try:
        y_hat = model(x)
        print(f"Data passed through model! - y_hat: {y_hat.shape}")
        logger.info(f"y_hat: {y_hat.shape}")
        logger.info(f"y_hat argmax: {torch.argmax(y_hat, dim=1)}")
    except:
        logger.exception('')
