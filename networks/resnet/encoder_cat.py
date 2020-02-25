import torch
import torch.nn as nn
from torchvision import models
from networks.resnet.Decoder import resnet18_decoder


class CVAE(nn.Module):
    def __init__(self, conf, out_channels=2, pretrained=0):
        super(CVAE, self).__init__()

        if pretrained:
            self.encoder = models.resnet18(pretrained=True)
            self.cond_encoder = models.resnet18(pretrained=True)
        else:
            self.encoder = models.resnet18(pretrained=False)
            self.cond_encoder = models.resnet18(pretrained=False)

        # SET THE CORRECT NUMBER OF INPUT CHANNELS
        self.conv_1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cond_conv_1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bottleneck = nn.Conv2d(1024, 512, kernel_size=(1, 1), bias=False)
        self.drop = nn.Dropout(p=0.5)

        self.encoder.conv1 = self.conv_1
        self.cond_encoder.conv1 = self.cond_conv_1

        # REMOVING LAST LAYER
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.cond_encoder = nn.Sequential(*list(self.cond_encoder.children())[:-1])

        # LAYERS FOR MU AND SIGMA
        self.mean = nn.Linear(512, 512)
        self.logvar = nn.Linear(512, 512)

        # DEFINE THE RGB DECODER
        self.decoder = resnet18_decoder(stride=2, out_channels=out_channels)

    def forward(self, color, inputs, prediction=False):

        z_grey = self.cond_encoder(inputs)

        if isinstance(color, type(None)):  # TEST TIME
            # z1 is sampled from Normal distribution,
            # we don't have color input on testing!
            z_rand = torch.randn(inputs.shape[0], 512, 1, 1).cuda()
            z = self.bottleneck(torch.cat((z_grey, z_rand), dim=1))
            return self.decoder(z), 0, 0, z_grey, z_rand, z
        else:
            # splitting latent space
            latent = torch.squeeze(self.encoder(color))
            latent = self.drop(latent)
            mu = self.mean(latent)
            logvar = self.logvar(latent)
            # repar trick
            stddev = torch.sqrt(torch.exp(logvar))
            eps = torch.randn(stddev.size()).normal_().cuda()
            z_color = torch.add(mu, torch.mul(eps, stddev))

            z_color = z_color.view(z_color.shape[0], 512, 1, 1)
            z = self.bottleneck(torch.cat((z_grey, z_color), dim=1))
            return self.decoder(z), mu, logvar, z_grey, z_color, z
