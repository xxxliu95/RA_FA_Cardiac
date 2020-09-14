import torch
from torch.nn import Module, Conv2d, LeakyReLU
from torch.nn.functional import relu
from torch.nn.utils import spectral_norm
from .spade import SPADE

class SPADEResBlk(Module):
    def __init__(self, spade_resblk_kernel, k, k_out, skip=False):
        super().__init__()
        kernel_size = spade_resblk_kernel
        self.skip = skip
        self.lrelu = LeakyReLU(0.2, inplace=True)
        
        self.spade1 = SPADE(128, 3, k)
        self.conv1 = Conv2d(k, k_out, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)

        self.spade2 = SPADE(128, 3, k_out)
        self.conv2 = Conv2d(k_out, k_out, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)

        self.spade_skip = SPADE(128, 3, k)
        self.conv_skip = Conv2d(k, k_out, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
    
    def forward(self, x, seg):
        x_skip = x
        x = self.spade1(x, seg)
        x = self.lrelu(x)
        x = self.conv1(x)
        x = self.spade2(x, seg)
        x = self.lrelu(x)
        x = self.conv2(x)

        if x_skip.shape[1] != x.shape[1]:
            x_skip = self.spade_skip(x_skip, seg)
            x_skip = self.conv_skip(x_skip)
        
        return x_skip + x