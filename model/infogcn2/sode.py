import numpy as np
import math
import random
import torch
import torch.nn.functional as F

from torch import nn

import sys
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(curr_dir, "..", ".."))
sys.path.append(path)

# TODO: If changing folder structure, update this import
# from model.modules import SA_GC, TemporalEncoder, GCN
# from model.utils import bn_init, import_class, sample_standard_gaussian,\
#     cum_mean_pooling, cum_max_pooling, identity, max_pooling
# from model.encoder_decoder import Encoder_z0_RNN, RNN
from model.infogcn2.modules import SA_GC, TemporalEncoder, GCN
from model.infogcn2.utils import (
    bn_init,
    import_class,
    sample_standard_gaussian,
    cum_mean_pooling,
    cum_max_pooling,
    identity,
    max_pooling,
)
from model.infogcn2.encoder_decoder import Encoder_z0_RNN, RNN
from model.attn import Flow_conv


from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchdiffeq import odeint as odeint
# @misc{torchdiffeq,
# 	author={Chen, Ricky T. Q.},
# 	title={torchdiffeq},
# 	year={2018},
# 	url={https://github.com/rtqichen/torchdiffeq},
# }


from torch.distributions.normal import Normal
from torch.distributions import kl_divergence


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, odeint_rtol=1e-4, odeint_atol=1e-5):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict):
        """
        # Decode the trajectory through ODE Solver
        """
        pred_y = odeint(
            self.ode_func,
            first_point,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
        )
        return pred_y


class ODEFunc(nn.Module):
    def __init__(self, dim, A, N, T=64):
        super(ODEFunc, self).__init__()
        self.layers = []
        self.A = A
        self.T = T
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        with torch.no_grad():
            self.temporal_pe = self.init_pe(T + N, dim)
            self.index = torch.arange(T).unsqueeze(-1).expand(T, dim)

    def forward(self, t, x, backwards=False):
        x = self.add_pe(x, t)  # TODO T dimension wise.
        x = torch.einsum("vu,ncu->ncv", self.A.to(x.dtype), x)
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.einsum("vu,ncu->ncv", self.A.to(x.dtype), x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.proj(x)

        return x

    def init_pe(self, length, d_model):
        with torch.no_grad():
            import math

            pe = torch.zeros(length, d_model)
            position = torch.arange(0, length).unsqueeze(1)
            div_term = torch.exp(
                (
                    torch.arange(0, d_model, 2, dtype=torch.float)
                    * -(math.log(10000.0) / d_model)
                )
            )  # length, model
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe = pe.view(length, d_model)
        return pe

    def add_pe(self, x, t):
        B, C, V = x.size()
        T = self.T
        index = self.index.to(x.device) + int(t)
        pe = torch.gather(self.temporal_pe.to(x.device), dim=0, index=index)
        pe = repeat(pe, "t c -> b t c v", v=V, b=B // T)
        x = rearrange(x, "(b t) c v -> b t c v", t=T)
        x = x + pe
        x = rearrange(x, "b t c v -> (b t) c v")
        return x


class SODE(nn.Module):
    '''
    SODE: Spatio-Temporal ODE-based Graph Convolutional Network for Skeleton-based Action Recognition.

    This class implements a neural network model that combines spatio-temporal graph convolutions, ODE-based temporal modeling, and classification for skeleton-based action recognition tasks.

    Args:
        num_class (int): Number of action classes for classification.
        num_point (int): Number of joints (nodes) in the skeleton graph.
        num_person (int): Number of persons in the input data.
        graph (str or class): Graph structure definition or class for adjacency matrix.
        pose_channels (int): Number of channels for pose input (e.g., 3 for (x, y, z)).
        flow_channels (int): Number of channels for flow input.
        embed_channels (int): Number of channels for joint embedding.
        num_head (int): Number of attention heads for temporal encoder.
        ode_method (str): ODE solver method (e.g., "rk4", "euler", "RNN").
        depth (int): Depth of the temporal encoder (number of layers).
        device (str): Device to use ("cuda" or "cpu").
        T (int): Sequence length (number of frames).
        n_step (int): Number of ODE steps for extrapolation.
        SAGC_proj (bool): Whether to use projection in spatial attention graph convolution.
        num_cls (int): Number of temporal classification segments.
        n_sample (int): Number of samples for classification/reconstruction.
        backbone (str): Backbone type for temporal encoder ("transformer" or "RNN").
        cnn (bool): Whether to use CNN-based joint embedding.

    Attributes:
        Graph: Graph structure instance.
        T (int): Sequence length.
        arange (Tensor): Tensor of time indices.
        shift_idx (Tensor): Indices for temporal shifting in extrapolation.
        mask (Tensor): Mask for shifted predictions.
        z0_prior (Normal): Prior distribution for latent variable z0.
        num_class (int): Number of classes.
        num_point (int): Number of joints.
        num_person (int): Number of persons.
        cls_idx (list): Indices for temporal classification segments.
        zero (Tensor): Zero tensor for loss placeholder.
        n_step (int): Number of ODE steps.
        arange_n_step (Tensor): Range tensor for ODE steps.
        to_joint_embedding (nn.Module): Module for joint embedding.
        pos_embedding (nn.Parameter): Learnable positional embedding.
        temporal_encoder (nn.Module): Temporal encoder module.
        n_sample (int): Number of samples.
        diffeq_solver (nn.Module): ODE solver or RNN for temporal dynamics.
        recon_decoder (nn.Sequential): Decoder for reconstructing input.
        cls_decoder (nn.Sequential): Decoder for classification.
        classifier_lst (list): List of classifiers for each temporal segment.
        spatial_pooling (function): Function for spatial pooling (default: mean).

    Methods:
        extrapolate(z_0, t): Extrapolates latent representations using ODE solver.
        origin_extrapolate(z, t): Alternative extrapolation method (not commonly used).
        get_A(k): Returns the k-th power adjacency matrix for the graph.
        KL_div(z_mu, z_std, kl_coef=1): Computes KL divergence between latent and prior (not used).
        forward(x): Forward pass of the model. Returns classification logits, reconstructed input, and latent variables.
        get_attention(): Returns attention weights from the temporal encoder.
        set_n_step(n_step): Sets the number of ODE steps for extrapolation.
    '''
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        graph=None,
        pose_channels=3,
        flow_channels=50,
        embed_channels=64,
        num_head=3,
        ode_method="rk4",
        depth=4,
        device="cuda",
        T=64,
        n_step=1,
        SAGC_proj=True,
        num_cls=10,
        n_sample=1,
        backbone="transformer",
        cnn=False,
        use_mask=True
    ):
        super(SODE, self).__init__()

        self.Graph = import_class(graph)()
        with torch.no_grad():
            A = np.stack([self.Graph.A_norm] * num_head, axis=0)
            self.T = T
            self.arange = torch.arange(T).view(1, 1, T) + 1
            shift_idx = torch.arange(0, T, dtype=int).view(1, T, 1, 1)
            shift_idx = repeat(shift_idx, "b t c v -> n b t c v", n=n_step)
            shift_idx = shift_idx - torch.arange(1, n_step + 1, dtype=int).view(
                n_step, 1, 1, 1, 1
            )
            self.mask = (
                torch.triu(torch.ones(n_step, T), diagonal=1)
                .view(n_step, 1, T, 1, 1)
                .to(device)
            )
            self.z0_prior = Normal(
                torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device)
            )
            self.shift_idx = shift_idx.to(device) % T
            self.num_class = num_class
            self.num_point = num_point
            self.num_person = num_person
            self.cls_idx = [int(math.ceil(T * i / num_cls)) for i in range(num_cls + 1)]
            self.cls_idx[0] = 0
            self.cls_idx[-1] = T
            self.zero = torch.tensor(0.0).to(device)
            self.n_step = n_step
            self.arange_n_step = torch.arange(n_step + 1).to(device)
        if cnn:
            assert flow_channels > 2  # Could be a better assertion here...
            self.to_joint_embedding = Flow_conv(
                kernel_size=3,
                flow_window=int(
                    math.sqrt(flow_channels / 2)
                ),  # flow_window = sqrt(flow_channels/2)
                original_channels=pose_channels,
                out_channels=embed_channels,
            )
        else:
            self.to_joint_embedding = nn.Linear(
                pose_channels + flow_channels, embed_channels
            )
            # self.to_joint_embedding = nn.Linear(pose_channels+flow_channels, embed_channels)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_point, embed_channels)
        )
        self.temporal_encoder = (
            TemporalEncoder(
                seq_len=T,
                dim=embed_channels,
                depth=depth,
                heads=4,
                mlp_dim=embed_channels * 2,
                dim_head=embed_channels // 4,
                device=device,
                A=A,
                num_point=num_point,
                SAGC_proj=SAGC_proj,
                use_mask=use_mask
            )
            if backbone == "transformer"
            else Encoder_z0_RNN(embed_channels, A, device)
        )
        self.n_sample = n_sample
        print(f"\tTemporal encoder: {backbone}")

        method = ode_method
        ode_func = ODEFunc(
            embed_channels,
            torch.from_numpy(self.Graph.A_norm).to(device),
            N=n_step,
            T=T,
        ).to(device)
        self.diffeq_solver = (
            DiffeqSolver(ode_func, method=method)
            if method != "RNN"
            else RNN(embed_channels, A, n_step)
        )

        self.recon_decoder = nn.Sequential(
            GCN(embed_channels, embed_channels, A),
            GCN(embed_channels, embed_channels, A),
            nn.Conv2d(embed_channels, 3, 1),
        )

        if n_step:
            in_dim = embed_channels * (n_step + 1)
            mid_dim = embed_channels * (n_step + 1) // 2
            out_dim = embed_channels * (n_step + 1) // 2
        elif n_step == 0:
            in_dim = embed_channels
            mid_dim = embed_channels
            out_dim = embed_channels

        self.cls_decoder = nn.Sequential(
            GCN(in_dim, mid_dim, A),
            GCN(mid_dim, out_dim, A),
        )

        # amp is not working with for loop.
        self.c0 = nn.Conv1d(out_dim, num_class, 1)
        self.c1 = nn.Conv1d(out_dim, num_class, 1)
        self.c2 = nn.Conv1d(out_dim, num_class, 1)
        self.c3 = nn.Conv1d(out_dim, num_class, 1)
        self.c4 = nn.Conv1d(out_dim, num_class, 1)
        self.c5 = nn.Conv1d(out_dim, num_class, 1)
        self.c6 = nn.Conv1d(out_dim, num_class, 1)
        self.c7 = nn.Conv1d(out_dim, num_class, 1)
        self.c8 = nn.Conv1d(out_dim, num_class, 1)
        self.c9 = nn.Conv1d(out_dim, num_class, 1)
        self.num_cls = num_cls
        self.classifier_lst = [
            self.c0,
            self.c1,
            self.c2,
            self.c3,
            self.c4,
            self.c5,
            self.c6,
            self.c7,
            self.c8,
            self.c9,
        ]
        self.spatial_pooling = torch.mean

    def extrapolate(self, z_0, t):
        """
        z : n c t v
        """
        B, C, T, V = z_0.size()
        z_0 = rearrange(z_0, "b c t v -> (b t) c v")

        zs = self.diffeq_solver(z_0, t)  # z_i = 2, (b t), c, v
        zs = rearrange(zs, "n (b t) c v -> n b t c v", t=T)
        z_hat = zs[1:]
        z_hat_shifted = torch.gather(
            z_hat.clone(), dim=2, index=self.shift_idx.expand_as(z_hat).long()
        )
        z_hat_shifted = self.mask * z_hat_shifted
        z_hat_shifted = rearrange(z_hat_shifted, "n b t c v -> (n b) c t v")
        z_hat = rearrange(z_hat, "n b t c v -> (n b) c t v")
        z_0 = rearrange(z_0, "(b t) c v -> b c t v", t=T)

        return z_0, z_hat, z_hat_shifted

    def origin_extrapolate(self, z, t):
        """
        z : n c t v m
        """
        if self.training:
            z_hat = []
            for i in range(self.N + 1):
                z_i = self.diffeq_solver(z, t[:2])
                z_hat = z_hat.append(z_i[-1])
            z_hat = torch.cat(z_hat, dim=0)
            # AR pred
            z_0 = z[:, :, self.obs - 1 : self.obs, :, :]
        else:
            z_hat = z[:, :, : self.obs, :, :]

        z_0 = z[:, :, self.obs - 1 : self.obs, :, :]
        z_pred = self.diffeq_solver(z_0, t[: self.obs + 1])
        z_hat = torch.cat((z_hat, z_pred), dim=0)

        return z_hat

    def get_A(self, k):
        A_outward = self.Graph.A_outward_binary
        I = np.eye(self.Graph.num_node)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def KL_div(self, z_mu, z_std, kl_coef=1):
        # TODO: remove (never called, z_mu and z_std must be torch Distribution's)
        z_distr = Normal(z_mu, z_std)
        kldiv_z0 = kl_divergence(z_distr, self.z0_prior)
        if torch.isnan(kldiv_z0).any():
            print(z_mu)
            print(z_std)
            raise Exception("kldiv_z0 is Nan!")
        loss = kldiv_z0.mean()
        return loss

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, T, V, M), where
                N: batch size,
                C: number of channels,
                T: number of time steps,
                V: number of joints (vertices),
                M: number of persons (instances).

        Returns:
            tuple:
                y (torch.Tensor): Classification output tensor of shape (N, num_cls, T).
                x_hat (torch.Tensor): Reconstructed input tensor.
                z_0 (torch.Tensor): Initial latent representation.
                z_hat_shifted (torch.Tensor): Extrapolated latent representation after shifting.
                self.zero (torch.Tensor): Zero tensor (purpose defined elsewhere in the class).
        """
        N, C, T, V, M = x.size()

        # Joint embedding
        x = rearrange(x, "n c t v m -> (n m t) v c", n=N, m=M, v=V)
        x = self.to_joint_embedding(x)

        # Add positional embedding
        x = x + self.pos_embedding[:, : self.num_point]

        # encoding
        x = rearrange(x, "(n m t) v c -> (n m) c t v", m=M, n=N)
        z = self.temporal_encoder(x)
        # kl_div = self.KL_div(z_mu, z_std)
        # z = self.latent_sample(z_mu, z_std)

        # extrapolation
        z_0, z_hat, z_hat_shifted = self.extrapolate(z, self.arange_n_step.to(z.dtype))

        # reconstruction
        x_hat = self.recon_decoder(z_hat_shifted)
        x_hat = rearrange(
            x_hat, "(n m l) c t v -> n l c t v m", m=M, l=self.n_sample
        ).mean(1)

        # classification
        if self.n_step:
            z_hat_cls = rearrange(z_hat, "(n b) c t v -> b (n c) t v", n=self.n_step)
            z_cls = self.cls_decoder(torch.cat([z_0, z_hat_cls], dim=1))
        else:
            z_cls = self.cls_decoder(z_0)
        z_cls = rearrange(
            z_cls, "(n m l) c t v -> (n l) m c t v", m=M, l=self.n_sample
        ).mean(1)  # N, 2*D, T
        z_cls = self.spatial_pooling(z_cls, dim=-1)

        y_lst = []
        for i in range(self.num_cls):
            y_lst.append(
                self.classifier_lst[i](
                    z_cls[:, :, self.cls_idx[i] : self.cls_idx[i + 1]]
                )
            )  # N, num_cls, T
        y = torch.cat(y_lst, dim=-1)
        y = rearrange(y, "(n l) c t -> n l c t", l=self.n_sample).mean(1)
        return y, x_hat, z_0, z_hat_shifted, self.zero

    def get_attention(self):
        return self.temporal_encoder.get_attention()
        # self.A_vector = self.get_A(k)

    def set_n_step(self, n_step):
        self.n_step_ode.set_n_step(n_step)


if __name__ == "__main__":
    from config.argclass import ArgClass

    model_type = 'cnn_TMP'
    dataset = 'nturgbd'

    arg = ArgClass(f"config/{dataset}/train_{model_type}.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arg.model_args["device"] = device

    model = SODE(**arg.model_args)
    model = model.to(device)

    # N, C, T, V, M
    C = arg.model_args["flow_channels"] + arg.model_args["pose_channels"]
    V = arg.model_args["num_point"]
    print(f"Model: {model_type}")
    print(f"Input channels: {C}\n")

    x = torch.randn((8, C, 64, V, 2)).to(device)
    print(f'Input shape: {x.shape}\n    (B, C, T, V, M)')
    y_hat, x_hat, z_0, z_hat_shifted, zero = model(x)
    print(f'\ny_hat: {y_hat.shape},\nx_hat: {x_hat.shape},\nz_0: {z_0.shape},\nz_hat_shifted: {z_hat_shifted.shape}\n')
    print(f'Attention length: {len(model.get_attention())}')
    print(f'Attention [0] shape: {model.get_attention()[0].shape}')
