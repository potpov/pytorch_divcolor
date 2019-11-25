import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):

    # define layers
    def __init__(self, conf):
        super(VAE, self).__init__()

        self.hidden_size = conf['HIDDENSIZE']

        # Encoder layers
        self.enc_conv0 = nn.Conv2d(2, 64, 5, stride=2, padding=2)
        self.enc_bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.enc_bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.enc_bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.enc_bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(1024)
        self.enc_fc1 = nn.Linear(4*4*1024, self.hidden_size * 2)

        # Decoder layers
        self.dec_upsamp0 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.dec_conv0 = nn.Conv2d(conf['HIDDENSIZE'], 1024, 3, stride=1, padding=1)
        self.dec_bn0 = nn.BatchNorm2d(1024)  # 4x4x1024
        self.dec_upsamp1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(512)  # 8 x 8 x 512
        self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(512, 256, 5, stride=1, padding=2)
        self.dec_bn2 = nn.BatchNorm2d(256)  # 16 x 16 x 256
        self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Conv2d(256, 128, 5, stride=1, padding=2)
        self.dec_bn3 = nn.BatchNorm2d(128)  # 32 x 32 x 128
        self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = nn.Conv2d(128, 64, 5, stride=1, padding=2)  # 64 x 64 x 64
        self.dec_bn4 = nn.BatchNorm2d(64)
        self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv5 = nn.Conv2d(64, 2, 5, stride=1, padding=2)  # 128 x 128x 2 final shape

    def encoder(self, x):
        x = F.relu(self.enc_conv0(x))
        x = self.enc_bn0(x)
        x = F.relu(self.enc_conv1(x))
        x = self.enc_bn1(x)
        x = F.relu(self.enc_conv2(x))
        x = self.enc_bn2(x)
        x = F.relu(self.enc_conv3(x))
        x = self.enc_bn3(x)
        x = F.relu(self.enc_conv4(x))
        x = self.enc_bn4(x)
        x = x.view(-1, 4*4*1024)
        x = self.enc_fc1(x)
        mu = x[..., :self.hidden_size]
        logvar = x[..., self.hidden_size:]
        return mu, logvar

    def decoder(self, z):
        x = z.view(-1, self.hidden_size, 1, 1)
        x = self.dec_upsamp0(x)
        x = F.relu(self.dec_conv0(x))
        x = self.dec_bn0(x)  # 4x4x1024
        x = self.dec_upsamp1(x)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_bn1(x)  # 8 x 8 x 512
        x = self.dec_upsamp2(x)
        x = F.relu(self.dec_conv2(x))
        x = self.dec_bn2(x)  # 16 x 16 x 256
        x = self.dec_upsamp3(x)
        x = F.relu(self.dec_conv3(x))
        x = self.dec_bn3(x)  # 32 x 32 x 128
        x = self.dec_upsamp4(x)
        x = F.relu(self.dec_conv4(x))  # 64 x 64 x 64
        x = self.dec_bn4(x)
        x = self.dec_upsamp5(x)  # 128 x 128 x 64
        x = torch.tanh(self.dec_conv5(x))  # 128 x 128 x 2 final shape
        return x

    #  define forward pass
    def forward(self, color, z_in):
        # OPTION 1: TRAINING THE VAE
        if self.training:
            mu, logvar = self.encoder(color)
            stddev = torch.sqrt(torch.exp(logvar))
            eps = torch.randn(stddev.size()).normal_().cuda()
            z = torch.add(mu, torch.mul(eps, stddev))
            # in training we need mu and var from encoder to force it to a gaussian distribution
            return mu, logvar, self.decoder(z)

        # OPTION 2 TESTING THE NETWORK
        if isinstance(color, type(None)):
            return None, None, self.decoder(z_in)  # in testing we just decode an input z

        # OPTION 3 TRAINING THE MDN
        mu, logvar = self.encoder(color)
        return mu, logvar, None
