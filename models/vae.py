import torch
import torch.nn as nn

from models.blocks import *

class VAE_encoder(nn.Module):
    def __init__(self, z_len):
        super(VAE_encoder, self).__init__()
        self.z_len = z_len
            
        self.block1 = conv_bn_relu(1, 32, 3, 2, 1)      #1 x 28 x 28
        self.block2 = conv_bn_relu(32, 32, 3, 2, 1)     #32 x 14 x 14
        self.block3 = conv_bn_relu(32, 64, 3, 2, 1)    #32 x 7 x 7
        self.block4 = conv_bn_relu(64, 256, 3, 2, 1)   #64 x 4 x 4
        #256 x 2 x 2
        self.mu = nn.Linear(1024, self.z_len)
        self.logvar = nn.Linear(1024, self.z_len)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        mu, logvar = self.encode(x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]))
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class VAE_decoder(nn.Module):
    def __init__(self, z_len, dim, upsample_method):
        super(VAE_decoder, self).__init__()
        self.z_len = z_len
        self.dim = dim
        self.upsample_method = upsample_method

        #bottleneck
        self.fc = nn.Linear(self.z_len, 1024)
        self.norm = nn.BatchNorm1d(1024)
        self.activ = nn.ReLU(inplace=True)
        #upsample decoder
        self.upsample4 = Interpolate((self.dim//8, self.dim//8), mode=self.upsample_method)
        self.block4 = conv_bn_relu(256, 64, 3, 1, 1)
        self.upsample3 = Interpolate((self.dim//4, self.dim//4), mode=self.upsample_method)
        self.block3 = conv_bn_relu(64, 32, 3, 1, 1)
        self.upsample2 = Interpolate((self.dim//2, self.dim//2), mode=self.upsample_method)
        self.block2 = conv_bn_relu(32, 32, 3, 1, 1)
        self.upsample1 = Interpolate((self.dim, self.dim), mode=self.upsample_method)
        self.block1 = conv_no_activ(32, 1, 3, 1, 1)
        #deconv decoder
        self.deconv4 = deconv_bn_relu(256, 64, 4, 2, 1)
        self.deconv3 = deconv_bn_relu(64, 32, 3, 2, 1)
        self.deconv2 = deconv_bn_relu(32, 32, 4, 2, 1)
        self.deconv1 = deconv(32, 1, 4, 2, 1)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        x = self.fc(z)
        #upsample decoder
        x = self.upsample4(x.view(-1, 256, 2, 2))
        x = self.block4(x)
        x = self.upsample3(x)
        x = self.block3(x)
        x = self.upsample2(x)
        x = self.block2(x)
        x = self.upsample1(x)
        x = self.block1(x)
        #deconv decoder
        # x = self.deconv4(x.view(-1, 256, 2, 2))
        # x = self.deconv3(x)
        # x = self.deconv2(x)
        # x = self.deconv1(x)
        #final output normalization
        x = self.sigmoid(x)


        return x


class VAE(nn.Module):
    def __init__(self, dim, z_len, upsample_method):
        super(VAE, self).__init__()

        self.dim = dim
        self.z_len = z_len
        self.upsample_method = upsample_method
        self.encoder = VAE_encoder(self.z_len)
        self.decoder = VAE_decoder(self.z_len, self.dim, self.upsample_method)

    def forward(self, x, test_z, name):
        z, mu, logvar = self.encoder(x)
        if name == 'train':
            x = self.decoder(z)
        else:
            x = self.decoder(test_z)

        return z, mu, logvar, x

