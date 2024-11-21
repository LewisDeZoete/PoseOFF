import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.model_utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory


class MS_G3D(nn.Module):
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
            MS_G3D(
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


class Flow_conv(nn.Module):
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
        # x of shape (batch, 300, 2, 17, channels+2k**2)
        flow_data = x.view(N*T*M*V, C)[:, self.original_channels:] # (N*T*M*V, 2K**2)
        flow_data = flow_data.view(N*T*M*V, self.flow_window, self.flow_window, 2).permute(0, 3, 1, 2)

        # Apply convolutions, dropout between then flatten 
        # flow_features = torch.relu(self.flow_conv1(flow_data))
        # flow_features = torch.relu(self.flow_conv2(flow_features))
        # flow_features = self.pool(flow_features)
        flow_features = self.conv(flow_data)    # Apply conv
        flow_features = flow_features.view(flow_features.size(0), -1) # flatten
        flow_features = self.fc(flow_features) # Linear layer to reduce out_channels
        flow_features = flow_features.view(N,T,M,V, -1)
        
        # Stack the output of this cnn onto the original graph features
        x = torch.cat((x[...,:self.original_channels], flow_features), dim=4)

        return x


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 flow_window=5,
                 kernel_size=3,
                 flow_channels=4):
        '''
        TODO: Test out other flow_conv models?
        TODO: Create a attribute for flow out channels (for resizing and forward method)
        '''
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * 
                                     (in_channels + (flow_window**2*2)) *
                                      num_point)

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384
        self.flow_channels = flow_channels

        # Adding the new Flow_Windows module
        # Input to the rest of MS-G3D will simply be in_channels+1
        self.flow_conv = Flow_conv(kernel_size=kernel_size, 
                                   original_channels=in_channels, 
                                   out_channels=self.flow_channels)

        # r=2? STGC blocks
        # self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.gcn3d1 = MultiWindow_MS_G3D(in_channels+flow_channels, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            # MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_GCN(num_gcn_scales, in_channels+flow_channels, c1, A_binary, disentangled_agg=True),
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

        # self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        # self.sgcn3 = nn.Sequential(
        #     MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
        #     MS_TCN(c2, c3, stride=2),
        #     MS_TCN(c3, c3))
        # self.sgcn3[-1].act = nn.Identity()
        # self.tcn3 = MS_TCN(c3, c3)

        self.fc = nn.Linear(c2, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        # (N, M*V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        # (N, T, M, V, C) 
        x = x.view(N, M, V, C, T).permute(0,4,1,2,3).contiguous()
        x = self.flow_conv(x)
        
        # (N * M, in_channels+flow_out_channels, T, V)
        # TODO: Make sure the channels here are correct!
        x = x.permute(0, 2, 4, 1, 3).contiguous().view(N*M, 3+self.flow_channels, T, V)

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)

        # x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        # x = self.tcn3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out



if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    flow_window = 5
    kernel_size = 3

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph',
        in_channels=3,
        flow_window=flow_window,
        kernel_size=kernel_size,
        flow_channels=4
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    N, C, T, V, M = 6, 3+2*flow_window**2, 100, 25, 2
    x = torch.randn(N,C,T,V,M)
    out = model.forward(x)
    print(out.shape)

    # print('Model total # params:', count_params(model))
