#!/usr/bin/env python3
# CREDIT: https://github.com/kennymckormick/pyskl/tree/main

# backbone=dict(
#     type='STGCN',
#     gcn_adaptive='init',
#     gcn_with_res=True,
#     tcn_type='mstcn',
#     graph_cfg=dict(
#         layout='nturgb+d',
#         mode='spatial',
# ))
# cls_head=dict(
#     type='GCNHead',
#     num_classes=120,
#     in_channels=256
# )
# optimiser=dict(type='SGD', lr=0.1, momentum=0.9, 'weight_decay=0.0005, nesterov=True)
# optimiser_config=dict(grad_clip=None)
# lr_config=dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
# total_epochs=16
import math
import copy as cp
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from models.stgcn2.modules import mstcn, unit_gcn, unit_tcn
from models.stgcn2.heads import GCNHead
from model_utils import (
    Flow_conv,
    bn_init,
    import_class,
)

EPS = 1e-4


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


# class STGCN(nn.Module):
#     def __init__(self,
#                  num_class=60,
#                  num_point=25,
#                  num_person=2,# * Only used when data_bn_type == 'MVC'
#                  graph=None,

#                  in_channels=3,
#                  base_channels=64,
#                  data_bn_type='VC',
#                  ch_ratio=2,
#                  num_stages=10,
#                  inflate_stages=[5, 8],
#                  down_stages=[5, 8],
#                  pretrained=None,
#                  **kwargs):
#         super().__init__()

#         self.Graph = import_class(graph)()
#         A = torch.tensor(self.Graph.A, dtype=torch.float32, requires_grad=False)
#         self.data_bn_type = data_bn_type
#         self.kwargs = kwargs

#         if data_bn_type == 'MVC':
#             self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
#         elif data_bn_type == 'VC':
#             self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
#         else:
#             self.data_bn = nn.Identity()

#         lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
#         for k, v in kwargs.items():
#             if isinstance(v, tuple) and len(v) == num_stages:
#                 for i in range(num_stages):
#                     lw_kwargs[i][k] = v[i]
#         lw_kwargs[0].pop('tcn_dropout', None)

#         self.in_channels = in_channels
#         self.base_channels = base_channels
#         self.ch_ratio = ch_ratio
#         self.inflate_stages = inflate_stages
#         self.down_stages = down_stages

#         modules = []
#         if self.in_channels != self.base_channels:
#             modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

#         inflate_times = 0
#         for i in range(2, num_stages + 1):
#             stride = 1 + (i in down_stages)
#             in_channels = base_channels
#             if i in inflate_stages:
#                 inflate_times += 1
#             out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
#             base_channels = out_channels
#             modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

#         if self.in_channels == self.base_channels:
#             num_stages -= 1

#         self.num_stages = num_stages
#         self.gcn = nn.ModuleList(modules)
#         self.pretrained = pretrained

#     def init_weights(self):
#         if isinstance(self.pretrained, str):
#             self.pretrained = cache_checkpoint(self.pretrained)
#             load_checkpoint(self, self.pretrained, strict=False)

#     def forward(self, x):
#         N, M, T, V, C = x.size()
#         x = x.permute(0, 1, 3, 4, 2).contiguous()
#         if self.data_bn_type == 'MVC':
#             x = self.data_bn(x.view(N, M * V * C, T))
#         else:
#             x = self.data_bn(x.view(N * M, V * C, T))
#         x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

#         for i in range(self.num_stages):
#             x = self.gcn[i](x)

#         x = x.reshape((N, M) + x.shape[1:])
#         return x


class TEST_MODEL(nn.Module):
    # TODO: Rename models and stuff
    def __init__(self,
                 num_class=60,
                 num_point=25,
                 num_person=2,# * Only used when data_bn_type == 'MVC'
                 graph=None,

                 pose_channels=3,
                 flow_channels=50,
                 # in_channels=3,

                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 device=None,
                 cnn=False,
                 **kwargs,
                 ):
        super().__init__()

        self.Graph = import_class(graph)()
        A = torch.tensor(self.Graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.cnn = cnn
        self.kwargs = kwargs

        in_channels=pose_channels+flow_channels

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            intermediate_channels = in_channels
            # Channel embeddings...
            if cnn:
                modules.append(
                    Flow_conv(
                        kernel_size=3,
                        flow_window=int(
                            math.sqrt(flow_channels / 2)
                        ),
                        pose_channels=pose_channels,
                        out_channels=in_channels,
                    )
                )
                modules.append(STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0]))
            else:
                modules.append(STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0]))

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.head = GCNHead(num_class, in_channels=base_channels)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        # N, M, T, V, C = x.size()
        N, C, T, V, M = x.size()

        # N M V C T
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))

        # The STGCNBlock expects shape (N*M, C, T, V)
        # Flow_conv expects shape (N*M*T, V, C)
        # TODO: Cleaner way of handing the embedding layer (possibly using einops Rearrange?)
        if self.cnn:
            x = rearrange(x, "N (M V C) T -> (N M T) V C", N=N, C=C, T=T, V=V, M=M)
            x = self.gcn[0](x)
            x = rearrange(x, "(N M T) V C -> (N M) C T V", N=N, T=T, M=M)
        else:
            x = rearrange(x, "N (M V C) T -> (N M) C T V", N=N, C=C, T=T, V=V, M=M)
            x = self.gcn[0](x)

        # x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(1, self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])

        # CLS head
        x = self.head(x)

        return x



if __name__ == '__main__':
    from config.argclass import ArgClass
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename='logs/debug/models/stgcn2.log',
        encoding='utf-8',
        filemode='w',
        level=logging.DEBUG
    )

    model_type = "stgcn2"
    dataset = "ntu"
    flow_embedding = "cnn"

    # Get the config file and use the model arguments defined within
    arg = ArgClass(f'config/{model_type}/{dataset}/{flow_embedding}.yaml', verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arg.model_args["device"] = device


    model = TEST_MODEL(
        **arg.model_args
    )
    print(f"\nModel arguments: {arg.model_args}")

    print('Model loaded')

    # Create dummy input
    # N, C, T, V, M
    C = arg.model_args["flow_channels"] + arg.model_args["pose_channels"]
    T = 160
    V = arg.model_args["num_point"]
    x = torch.randn((8, C, T, V, 2)).to(device)
    logger.info(f"Model: {flow_embedding}")
    logger.info(f"Input channels: {C}")
    logger.info(f"Input shape: {x.shape}\n    (B, C, T, V, M)")

    print('Input created')
    try:
        # Pass input to model
        y_hat = model(x)
        logger.info(f"y_hat: {y_hat.shape}")
        logger.info(f"y_hat argmax: {torch.argmax(y_hat, dim=1)}")
    except:
        logger.exception('')

