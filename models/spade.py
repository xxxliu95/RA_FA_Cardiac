import torch
from torch.nn import Module, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import interpolate, relu

import warnings
warnings.filterwarnings("ignore")

class SPADE(Module):
    def __init__(self, spade_filter, spade_kernel, k):
        super().__init__()
        num_filters = spade_filter
        kernel_size = spade_kernel
        self.conv = Conv2d(8, num_filters, kernel_size, 1, 1, 1)#spectral_norm(Conv2d(8, num_filters, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_gamma = Conv2d(num_filters, k, kernel_size, 1, 1, 1)#spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_beta = Conv2d(num_filters, k, kernel_size, 1, 1, 1)#spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))

    def forward(self, x, seg):
        N, C, H, W = x.size()

        sum_channel = torch.sum(x.reshape(N, C, H*W), dim=-1)
        mean = sum_channel / (H*W)
        std = torch.sqrt((sum_channel**2 - mean**2) / (H*W))

        mean = torch.unsqueeze(torch.unsqueeze(mean, -1), -1)
        std = torch.unsqueeze(torch.unsqueeze(std, -1), -1)
        x = (x - mean) / std

        seg = interpolate(seg, size=(H,W), mode='nearest')
        seg = relu(self.conv(seg))
        seg_gamma = self.conv_gamma(seg)
        seg_beta = self.conv_beta(seg)

        x = x * (1 + seg_gamma) + seg_beta #torch.matmul(seg_gamma, x) + seg_beta

        return x