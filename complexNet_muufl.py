"""
complexNet_muufl.py
--------------------
Quantum-Inspired Complex Spiking Neural Network for the MUUFL Gulfport dataset
(official GatorSense scene-labeled benchmark, 325 x 220, 11 classes).

Key differences vs. Trento variant:
  - HSI branch  : input_dim PCA components (default 15)
  - LiDAR branch: lidar_dim channels (1-2 elevation returns from official file)
  - Classifier  : 11 output classes
  - Mamba block : applied after the second deep complex-conv stage (same as Trento)

FC input size derivation (patch_size = 15, T = 25):
  conv5 (stride=2): H/W: 15 -> 15//2+1 = 8
  conv8 (stride=2): H/W:  8 -> (15//2)//2+1 = 4
  xr5 shape: (T, B, 256, 4, 4)
  flatten -> B x (25 * 256 * 4 * 4) = B x 102400
  (This only depends on patch_size, not scene width, so it is unchanged
   from the Trento/Houston variants.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu
from mamba_block import SimpleMamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate gradient for spiking neurons
# ─────────────────────────────────────────────────────────────────────────────
class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        temp = (1 / 2.5) * torch.sign(abs(input) < 2.5)
        return grad_input * temp.float()


# ─────────────────────────────────────────────────────────────────────────────
# Complex-valued layers
# ─────────────────────────────────────────────────────────────────────────────
class ComplexConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)

    def forward(self, input_r, input_i):
        assert input_r.size() == input_i.size()
        return (self.conv_r(input_r) - self.conv_i(input_i),
                self.conv_r(input_i) + self.conv_i(input_r))


class ComplexLinear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input_r, input_i):
        return (self.fc_r(input_r) - self.fc_i(input_i),
                self.fc_r(input_i) + self.fc_i(input_r))


class _ComplexBatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features       = num_features
        self.eps                = eps
        self.momentum           = momentum
        self.affine             = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias   = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias",   None)

        if self.track_running_stats:
            self.register_buffer("running_mean",  torch.zeros(num_features, 2))
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer("num_batches_tracked",
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean",  None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, input_r, input_i):
        assert input_r.size() == input_i.size()
        assert len(input_r.shape) == 4

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = \
                        1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])
            mean   = torch.stack((mean_r, mean_i), dim=1)
            with torch.no_grad():
                self.running_mean = (exponential_average_factor * mean
                                     + (1 - exponential_average_factor)
                                     * self.running_mean)
            input_r = input_r - mean_r[None, :, None, None]
            input_i = input_i - mean_i[None, :, None, None]

            n   = input_r.numel() / input_r.size(1)
            Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0, 2, 3])

            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 0])
                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1])
                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2])
        else:
            mean = self.running_mean
            Crr  = self.running_covar[:, 0] + self.eps
            Cii  = self.running_covar[:, 1] + self.eps
            Cri  = self.running_covar[:, 2]
            input_r = input_r - mean[None, :, 0, None, None]
            input_i = input_i - mean[None, :, 1, None, None]

        det        = Crr * Cii - Cri.pow(2)
        s          = torch.sqrt(det)
        t          = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = (
            Rrr[None, :, None, None] * input_r + Rri[None, :, None, None] * input_i,
            Rii[None, :, None, None] * input_i + Rri[None, :, None, None] * input_r)

        if self.affine:
            input_r, input_i = (
                self.weight[None, :, 0, None, None] * input_r
                + self.weight[None, :, 2, None, None] * input_i
                + self.bias[None, :, 0, None, None],
                self.weight[None, :, 2, None, None] * input_r
                + self.weight[None, :, 1, None, None] * input_i
                + self.bias[None, :, 1, None, None])

        return input_r, input_i


# ─────────────────────────────────────────────────────────────────────────────
# Multi-step LIF neuron
# ─────────────────────────────────────────────────────────────────────────────
class MultiStepLIFNode(nn.Module):
    def __init__(self, tau, detach_reset, backend):
        super().__init__()
        self.spike_fn = Surrogate_BP_Function.apply

    def forward(self, x_seq: torch.Tensor, leake, input, threshold=1):
        spike_seq = []
        mem = torch.zeros_like(x_seq[0])
        for t in range(x_seq.shape[0]):
            mem     = 0.7 * mem + 0.7 * x_seq[t]
            mem_thr = mem - 1.0
            x       = self.spike_fn(mem_thr)
            mem     = mem - x
            spike_seq.append(x.unsqueeze(0))
        return torch.cat(spike_seq, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Main network
# ─────────────────────────────────────────────────────────────────────────────
class ComplexNet(nn.Module):
    """
    Parameters
    ----------
    num_steps  : int  - number of SNN time steps  (T)
    input_dim  : int  - HSI channels after PCA     (default 15)
    lidar_dim  : int  - LiDAR channels (1 or 2 elevation returns from the
                        official MUUFL file; check LIDAR_CHANNELS printed
                        by dataload_muufl.py and set this to match)
    num_cls    : int  - number of output classes   (default 11 for MUUFL)
    fc_size    : int  - flattened FC input size.
                        For patch=15, T=25: 25*256*4*4 = 102400
                        (independent of scene width/lidar channel count)
    """

    def __init__(self, num_steps, input_dim=15, lidar_dim=1,
                 num_cls=11, fc_size=102400):
        super().__init__()

        self.T = num_steps
        bias_flag = False

        # ── HSI branch (real part) ───────────────────────────────────────────
        self.conv1    = nn.Conv2d(input_dim, 64, kernel_size=3,
                                  stride=1, padding=1, bias=bias_flag)
        self.BN1      = BatchNorm2d(64)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        # ── LiDAR branch (imaginary part) ───────────────────────────────────
        self.conv2    = nn.Conv2d(lidar_dim, 64, kernel_size=1,
                                  stride=1, padding=0, bias=bias_flag)
        self.BN2      = BatchNorm2d(64)
        self.proj_lif2 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")

        # ── Shallow complex conv ─────────────────────────────────────────────
        self.proj_lif3 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")
        self.conv4     = ComplexConv2d(64, 64, kernel_size=3,
                                       stride=1, padding=1, bias=bias_flag)
        self.BN3       = ComplexBatchNorm2d(64)
        self.proj_lif4 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")

        # ── First downsampling ───────────────────────────────────────────────
        self.conv5     = ComplexConv2d(64, 128, kernel_size=3,
                                       stride=2, padding=1, bias=bias_flag)
        self.BN4       = ComplexBatchNorm2d(128)
        self.proj_lif5 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")
        self.proj_lif6 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")

        # ── Deep complex conv block 1 ────────────────────────────────────────
        self.conv6     = ComplexConv2d(128, 128, kernel_size=3,
                                       stride=1, padding=1, bias=bias_flag)
        self.BN5       = ComplexBatchNorm2d(128)
        self.proj_lif7 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")
        self.proj_lif8 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")

        # ── Deep complex conv block 2  +  Mamba ─────────────────────────────
        self.conv7     = ComplexConv2d(128, 128, kernel_size=3,
                                       stride=1, padding=1, bias=bias_flag)
        self.BN6       = ComplexBatchNorm2d(128)
        self.proj_lif9  = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")
        self.proj_lif10 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")
        self.mamba      = SimpleMamba(128)   # temporal sequence modelling

        # ── Second downsampling ──────────────────────────────────────────────
        self.conv8      = ComplexConv2d(128, 256, kernel_size=3,
                                        stride=2, padding=1, bias=bias_flag)
        self.BN7        = ComplexBatchNorm2d(256)
        self.proj_lif11 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")
        self.proj_lif12 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend="cupy")

        # ── Classifier ───────────────────────────────────────────────────────
        self.fc1 = ComplexLinear(fc_size, num_cls)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=5)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=5)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, hsi, lidar):
        """
        hsi   : (B, C_hsi,   H, W)
        lidar : (B, C_lidar, H, W)
        """
        B, C, H, W = hsi.shape

        # ── HSI encoding (real part) ─────────────────────────────────────────
        inputs2        = hsi.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        inputs2_encode = self.conv1(inputs2.flatten(0, 1))
        inputs2_encode = self.BN1(inputs2_encode)
        xr             = inputs2_encode.reshape(self.T, B, -1, H, W)
        xr             = self.proj_lif1(xr, 0.7, 1.0)

        # ── LiDAR encoding (imaginary part) ─────────────────────────────────
        inputs3        = lidar.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        inputs3_encode = self.conv2(inputs3.flatten(0, 1))
        inputs3_encode = self.BN2(inputs3_encode)
        xi             = inputs3_encode.reshape(self.T, B, -1, H, W)
        xi             = self.proj_lif2(xi, 0.7, 1.0)

        # ── Shallow complex conv ─────────────────────────────────────────────
        xr1, xi4 = self.conv4(xr.flatten(0, 1), xi.flatten(0, 1))
        xr1, xi4 = self.BN3(xr1, xi4)
        xr1      = xr1.reshape(self.T, B, -1, H, W)
        xr1      = self.proj_lif3(xr1, 0.7, 1.0)
        xi4      = xi4.reshape(self.T, B, -1, H, W)
        xi4      = self.proj_lif4(xi4, 0.7, 1.0)

        # ── Downsample ───────────────────────────────────────────────────────
        H2 = H // 2 + 1
        W2 = W // 2 + 1
        xr2, xi5 = self.conv5(xr1.flatten(0, 1), xi4.flatten(0, 1))
        xr2, xi5 = self.BN4(xr2, xi5)
        xr2      = xr2.reshape(self.T, B, -1, H2, W2)
        xr2      = self.proj_lif5(xr2, 0.7, 1.0)
        xi5      = xi5.reshape(self.T, B, -1, H2, W2)
        xi5      = self.proj_lif6(xi5, 0.7, 1.0)

        # ── Deep complex conv block 1 ─────────────────────────────────────────
        xr3, xi6 = self.conv6(xr2.flatten(0, 1), xi5.flatten(0, 1))
        xr3, xi6 = self.BN5(xr3, xi6)
        xr3      = xr3.reshape(self.T, B, -1, H2, W2)
        xr3      = self.proj_lif7(xr3, 0.7, 1.0)
        xi6      = xi6.reshape(self.T, B, -1, H2, W2)
        xi6      = self.proj_lif8(xi6, 0.7, 1.0)

        # ── Deep complex conv block 2  +  Mamba ──────────────────────────────
        xr4, xi7 = self.conv7(xr3.flatten(0, 1), xi6.flatten(0, 1))
        xr4, xi7 = self.BN6(xr4, xi7)
        xr4      = xr4.reshape(self.T, B, -1, H2, W2)
        xr4      = self.proj_lif9(xr4, 0.7, 1.0)
        xi7      = xi7.reshape(self.T, B, -1, H2, W2)
        xi7      = self.proj_lif10(xi7, 0.7, 1.0)
        xr4      = self.mamba(xr4)
        xi7      = self.mamba(xi7)

        # ── Downsample ────────────────────────────────────────────────────────
        H3 = (H // 2) // 2 + 1
        W3 = (W // 2) // 2 + 1
        xr5, xi8 = self.conv8(xr4.flatten(0, 1), xi7.flatten(0, 1))
        xr5, xi8 = self.BN7(xr5, xi8)
        xr5      = xr5.reshape(self.T, B, -1, H3, W3)
        xr5      = self.proj_lif11(xr5, 0.7, 1.0)
        xi8      = xi8.reshape(self.T, B, -1, H3, W3)
        xi8      = self.proj_lif12(xi8, 0.7, 1.0)

       # Scal1 (21×21 only)
        xr6 = xr1.transpose(0,1).reshape(B,-1)
        xi9 = xi4.transpose(0,1).reshape(B,-1)

        xr6, xi9 = self.fc1(xr6, xi9)

        x = torch.sqrt(torch.pow(xr6,2) + torch.pow(xi9,2))
        return x