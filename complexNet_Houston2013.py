
"""
@CreatedDate:   2022/04
@Author: lyh
"""
import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode
import torch
import torch.nn as nn
import torch.nn.functional as F
# from spikingjelly.activation_based import neuron, surrogate, encoding
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu
from torchvision import datasets, transforms
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def complex_relu(input_r, input_i):
    return relu(input_r), relu(input_i)

class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        # temp = torch.abs(1 - torch.abs(torch.arcsin(input))) < 0.7
        temp = (1 / 2.5) * torch.sign(abs(input) < 2.5)
        return grad_input * temp.float()
class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_r, input_i):
        assert (input_r.size() == input_i.size())
        return self.conv_r(input_r) - self.conv_i(input_i), self.conv_r(input_i) + self.conv_i(input_r)
        # return neuron.LIFNode(self.conv_r(input_r) - self.conv_i(input_i)), neuron.LIFNode(self.conv_r(input_i) + self.conv_i(input_r))

class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input_r, input_i):
        return self.fc_r(input_r) - self.fc_i(input_i), self.fc_r(input_i) + self.fc_i(input_r)

class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
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
        assert (input_r.size() == input_i.size())
        assert (len(input_r.shape) == 4)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])

            mean = torch.stack((mean_r, mean_i), dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r - mean_r[None, :, None, None]
            input_i = input_i - mean_i[None, :, None, None]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0, 2, 3])

            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

            input_r = input_r - mean[None, :, 0, None, None]
            input_i = input_i - mean[None, :, 1, None, None]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :, None, None] * input_r + Rri[None, :, None, None] * input_i, \
                           Rii[None, :, None, None] * input_i + Rri[None, :, None, None] * input_r

        if self.affine:
            input_r, input_i = self.weight[None, :, 0, None, None] * input_r + self.weight[None, :, 2, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 0, None, None], \
                               self.weight[None, :, 2, None, None] * input_r + self.weight[None, :, 1, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 1, None, None]

        return input_r, input_i

class MultiStepLIFNode(nn.Module):
    def __init__(self,tau, detach_reset, backend):
        super().__init__()
        self.spike_fn = Surrogate_BP_Function.apply

    def forward(self, x_seq: torch.Tensor, leake, input,threshold=1):

        spike_seq = []
        mem=torch.zeros(x_seq[0].shape).to(device)
        for t in range(x_seq.shape[0]):
            # mem=leake[t]*mem+input[t]*x_seq[t]
            mem = 0.7 * mem + 0.7 * x_seq[t]
            mem_thr = mem - 1.0
            x= self.spike_fn(mem_thr)
            mem=mem-x
            spike_seq.append(x.unsqueeze(0))

        spike_seq = torch.cat(spike_seq, 0)

        return spike_seq
class ComplexNet(nn.Module):
    def __init__(self, num_steps,input_dim):
        super(ComplexNet, self).__init__()

        self.T = num_steps

        bias_flag = False

        self.BNli = BatchNorm2d(1)
        # 首先，进行直接编码
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # 然后得到虚部 BN-LIF-CONV-BN-RELU-CONV
        self.BN1 = BatchNorm2d(64)
        self.conv2 = nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0, bias=bias_flag)
        self.proj_lif2 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')
        self.BN2 = BatchNorm2d(64)

        self.proj_lif3 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')
        # 从这里开始进行复数卷积操作
        self.conv4 = ComplexConv2d(64,64,kernel_size=3,stride=1,padding=1,bias=bias_flag)
        self.BN3 = ComplexBatchNorm2d(64)
        self.proj_lif4 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')

        # 下采样
        self.conv5 = ComplexConv2d(64,128,kernel_size=3,stride=2,padding=1,bias=bias_flag)
        self.BN4 = ComplexBatchNorm2d(128)
        self.proj_lif5 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')

        # 下采样之后的复数卷积
        self.conv6 = ComplexConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.BN5 = ComplexBatchNorm2d(128)
        self.proj_lif6 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')
        self.conv7 = ComplexConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.BN6 = ComplexBatchNorm2d(128)
        self.proj_lif7 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')

        # 下采样
        self.conv8 = ComplexConv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias_flag)
        self.BN7 = ComplexBatchNorm2d(256)
        self.proj_lif8 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')
        self.proj_lif9 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')
        self.proj_lif10 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')
        self.proj_lif11 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')
        self.proj_lif12 = MultiStepLIFNode(tau=1.0, detach_reset=True, backend='cupy')

        # 全连接层分类
        # self.fc1 = ComplexLinear(11520, 16) #pacth size 9 time5
        # self.fc1 = ComplexLinear(23040, 16)
        # self.fc1 = ComplexLinear(34560, 16) # time10 patchsize9
        self.fc1 = ComplexLinear(92160, 15) # time15 patchsize9

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)

    def forward(self, hsi,lidar):
        # inputs1 = np.transpose(input, (0, 2, 3, 1))
        # inputs2 = inputs1[..., 0:3]
        # inputs2 = np.transpose(inputs2,(0,3,1,2)).to(device)
        # inputs3 = inputs1[..., 3:]
        # inputs3 = np.transpose(inputs3,(0,3,1,2)).to(device)
        B,C,H,W = hsi.shape

        inputs2 = hsi.unsqueeze(0).repeat(self.T,1,1,1,1).to(hsi.device)
        # inputs2 = inputs2.flatten(0,1)
        # 通过卷积直接编码
        inputs2_encode = self.conv1(inputs2.flatten(0,1))
        inputs2_encode = self.BN1(inputs2_encode)
        # inputs2_encode = self.BN1(inputs2_encode)
        xr = inputs2_encode.reshape(self.T,B,-1,H,W)
        xr = self.proj_lif1(xr,0.7,1.0)
        # xr = self.proj_lif1(xr)

        inputs3 = lidar.unsqueeze(0).repeat(self.T,1,1,1,1).to(hsi.device)
        inputs3_encode = self.conv2(inputs3.flatten(0, 1))
        inputs3_encode = self.BN2(inputs3_encode)
        # inputs3_encode = self.BN2(inputs3_encode)
        xi = inputs3_encode.reshape(self.T, B, -1, H, W)
        xi = self.proj_lif2(xi, 0.7, 1.0)
        # xi = self.proj_lif2(xi)

        # 浅层网络复数卷积
        # xr1 = xr.reshape(self.T,B,-1,H,W)
        # xi4 = xi.reshape(self.T,B,-1,H,W)
        xr1,xi4 = self.conv4(xr.flatten(0,1),xi.flatten(0,1))
        xr1,xi4 = self.BN3(xr1,xi4)
        # 这里需要明确实部和虚部是否需要使用同一个LIF神经元,这里实部和虚部使用的是不同的LIF神经元。
        xr1 = xr1.reshape(self.T, B, -1, H, W)
        xr1 = self.proj_lif3(xr1,0.7,1.0)
        # xr1 = self.proj_lif3(xr1)
        xi4 = xi4.reshape(self.T, B, -1, H, W)
        xi4 = self.proj_lif4(xi4,0.7,1.0)
        # xi4 = self.proj_lif4(xi4)

        # xr1 10,64,64,17,17  xi4 10,64,64,17,17

        # 下采样  17×17 下采样之后大小为9
        xr2,xi5 = self.conv5(xr1.flatten(0,1),xi4.flatten(0,1))
        xr2, xi5 = self.BN4(xr2,xi5)
        xr2 = xr2.reshape(self.T,B,-1,H//2+1,W//2+1)
        xr2 = self.proj_lif5(xr2, 0.7, 1.0)
        # xr2 = self.proj_lif5(xr2)
        xi5 = xi5.reshape(self.T, B, -1, H//2+1,W//2+1)
        xi5 = self.proj_lif6(xi5, 0.7, 1.0)
        # xi5 = self.proj_lif6(xi5)

        # 第一个深层复数卷积
        xr3,xi6 = xr2.flatten(0,1),xi5.flatten(0,1)
        xr3,xi6 = self.conv6(xr3,xi6)
        xr3,xi6 = self.BN5(xr3,xi6)
        xr3 = xr3.reshape(self.T, B, -1, H // 2+1, W // 2+1)
        xr3 = self.proj_lif7(xr3, 0.7, 1.0)
        # xr3 = self.proj_lif7(xr3)
        xi6 = xi6.reshape(self.T, B, -1, H // 2+1, W // 2+1)
        xi6 = self.proj_lif8(xi6, 0.7, 1.0)
        # xi6 = self.proj_lif8(xi6)

        # 第二个深层复数卷积
        xr4, xi7 = self.conv7(xr3.flatten(0, 1), xi6.flatten(0, 1))
        xr4, xi7 = self.BN6(xr4,xi7)
        xr4 = xr4.reshape(self.T, B, -1, H // 2+1, W // 2+1)
        xr4 = self.proj_lif9(xr4, 0.7, 1.0)
        # xr4 = self.proj_lif9(xr4)
        xi7 = xi7.reshape(self.T, B, -1, H // 2+1, W // 2+1)
        xi7 = self.proj_lif10(xi7, 0.7, 1.0)
        # xi7 = self.proj_lif10(xi7)

        # 下采样
        xr5, xi8 = self.conv8(xr4.flatten(0, 1), xi7.flatten(0, 1))
        xr5, xi8 = self.BN7(xr5,xi8)
        xr5 = xr5.reshape(self.T, B, -1, (H // 2)//2+1, (W // 2)//2+1)
        xr5 = self.proj_lif11(xr5, 0.7, 1.0)
        # xr5 = self.proj_lif11(xr5)
        xi8 = xi8.reshape(self.T, B, -1, (H // 2)//2+1, (W // 2)//2+1)
        xi8 = self.proj_lif12(xi8, 0.7, 1.0)
        # xi8 = self.proj_lif12(xi8)

        # 全连接层
        xr6,xi9 = xr5.transpose(0,1),xi8.transpose(0,1)
        xr6, xi9 = xr6.reshape(B,-1),xi9.reshape(B,-1)

        xr6, xi9 = self.fc1(xr6, xi9)

        # 通过复数概率幅度以及量子叠加态原理实现最终预测
        x = torch.sqrt(torch.pow(xr6,2) + torch.pow(xi9,2))

        # 第一次跑是直接返回概率幅
        # return F.log_softmax(x,dim=1)
        return x
