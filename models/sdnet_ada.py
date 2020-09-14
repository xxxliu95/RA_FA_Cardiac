import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time

from models.unet_parts import *
from models.blocks import *
from models.rounding import *
from models.spectral_norm import *
from models.distance_corr import *
from models.spade_resblk import *

device = torch.device('cuda:0')


# content
class Segmentor(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Segmentor, self).__init__()
        """
        """
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes+1  # check again

        self.conv1 = conv_bn_relu(self.num_output_channels, 64, 3, 1, 1)
        self.conv2 = conv_bn_relu(64, 64, 3, 1, 1)
        self.pred = nn.Conv2d(64, self.num_classes, 1, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)
        out = F.softmax(out, dim=1)

        return out

class AEncoder(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, norm, upsample):
        super(AEncoder, self).__init__()
        """
        Build an encoder to extract anatomical information from the image.
        """
        self.width = width
        self.height = height
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.upsample = upsample

        self.unet = UNet(n_channels=1, n_classes=self.num_output_channels, bilinear=True)
        self.rounding = RoundLayer()

    def forward(self, x):
        out = self.unet(x)
        out = F.softmax(out, dim=1)
        out = self.rounding(out)

        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# style

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class StyleEncoder(nn.Module):
    def __init__(self, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        dim = 64
        self.model = []
        self.model += [Conv2dBlock(1, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # for i in range(n_downsample - 2):
        #     self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

# decoder
class Ada_Decoder(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, decoder_type, anatomy_out_channels, z_length, num_mask_channels):
        super(Ada_Decoder, self).__init__()
        """
        """
        self.dec = Decoder(anatomy_out_channels, res_norm='adain', activ='relu', pad_type='reflect')
        # MLP to generate AdaIN parameters
        self.mlp = MLP(z_length, self.get_num_adain_params(self.dec), 256, 3, norm='none', activ='relu')

    def forward(self, a, z, type):
        # reconstruct an image
        images_recon = self.decode(a, z)
        return images_recon

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class Decoder(nn.Module):
    def __init__(self, dim, output_dim=1, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # upsampling blocks
        for i in range(3):
            self.model += [Conv2dBlock(dim, dim // 2, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class Discriminator(nn.Module):
    def __init__(self, ndf, num_classes):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.num_classes = num_classes + 1
        self.main = []
        # input is (nc) x 224 x 224
        self.main += [nn.Conv2d(self.num_classes, ndf, 4, 2, 1, bias=False)] #64x112x112
        self.main += [nn.LeakyReLU(0.2, inplace=True)]
        self.main += [SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))] #128x56x56
        self.main += [SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))] #256x28x28
        self.main += [SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))] #512x14x14
        self.main += [SpectralNorm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))] #1024x7x7
        # state size. (ndf*16) x 14 x 14
        self.out = nn.Linear(self.ndf * 8 * 7 * 7, 1)

        self.main =  nn.Sequential(*self.main)

    def forward(self, x):
        b_size = x.size(0)
        down_out = self.main(x)
        down_out = down_out.view(b_size, -1)
        output = self.out(down_out)
        return output.view(-1, 1).squeeze(1)


class SDNet(nn.Module):
    def __init__(self, width, height, num_classes, ndf, z_length, norm, upsample, decoder_type, anatomy_out_channels, num_mask_channels):
        super(SDNet, self).__init__()
        """
        Args:
            width: input width
            height: input height
            upsample: upsampling type (nearest | bilateral)
            nclasses: number of semantice segmentation classes
        """
        self.h = height
        self.w = width
        self.ndf = ndf
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.decoder_type = decoder_type
        self.num_mask_channels = num_mask_channels

        self.m_encoder = StyleEncoder(z_length, norm='none', activ='relu', pad_type='reflect')
        self.a_encoder = AEncoder(self.h, self.w, self.ndf, self.anatomy_out_channels, self.norm, self.upsample)
        self.segmentor = Segmentor(self.anatomy_out_channels, self.num_classes)
        self.decoder = Ada_Decoder(self.decoder_type, self.anatomy_out_channels, self.z_length, self.num_mask_channels)

    def forward(self, x, mask, script_type):
        # z_out = torch.randn(x.shape[0], self.z_length, 1, 1).to(device)
        z_out = self.m_encoder(x)
        a_out = self.a_encoder(x)
        seg_pred = self.segmentor(a_out)


        logvar_out = None
        mu_out = None
        #t0 = time.time()
        if script_type == 'training':
            reco = self.decoder(a_out, z_out, self.decoder_type)
            mu_out_tilde = self.m_encoder(reco)
        elif script_type == 'val' or script_type == 'test':
            z_out = self.m_encoder(x)
            reco = self.decoder(a_out, z_out, self.decoder_type)
            mu_out_tilde = None

        return reco, z_out, mu_out_tilde, a_out, seg_pred, mu_out, logvar_out

    def reconstruct(self, a_out, z_out):
        reco = self.decoder(a_out, z_out, self.decoder_type)

        return reco