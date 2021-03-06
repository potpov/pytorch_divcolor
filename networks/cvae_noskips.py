import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):

    # define layers
    def __init__(self, conf):
        super(CVAE, self).__init__()
        self.hidden_size = conf['HIDDEN_SIZE']
        self.train_batch_size = conf['TEST_BATCHSIZE']

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
        self.enc_fc1 = nn.Linear(4 * 4 * 1024, self.hidden_size * 2)

        # Cond encoder layers
        self.cond_enc_conv0 = nn.Conv2d(9, 64, 5, stride=2, padding=2)
        self.cond_enc_bn0 = nn.BatchNorm2d(64)
        self.cond_enc_conv1 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.cond_enc_bn1 = nn.BatchNorm2d(128)
        self.cond_enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.cond_enc_bn2 = nn.BatchNorm2d(256)
        self.cond_enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.cond_enc_bn3 = nn.BatchNorm2d(512)
        self.cond_enc_conv4 = nn.Conv2d(512, self.hidden_size, 3, stride=2, padding=1)

        # Decoder layers
        self.dec_upsamp1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(self.hidden_size, 64, 5, stride=1, padding=2)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(64, 32, 5, stride=1, padding=2)
        self.dec_bn2 = nn.BatchNorm2d(32)
        self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Conv2d(32, 16, 5, stride=1, padding=2)
        self.dec_bn3 = nn.BatchNorm2d(16)
        self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = nn.Conv2d(16, 8, 5, stride=1, padding=2)
        self.dec_bn4 = nn.BatchNorm2d(8)
        self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(8, 2, 1, stride=1, padding=0)

    def encoder(self, x):
        """
        :param x: AB COLOR IMAGE, shape: 2 x imgw x imgh
        :return: mu and log var for the hidden space
        """
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
        x = x.view(-1, 4 * 4 * 1024)
        # x = self.enc_dropout1(x)
        x = self.enc_fc1(x)
        mu = x[..., :self.hidden_size]
        logvar = x[..., self.hidden_size:]
        return mu, logvar

    def cond_encoder(self, x):
        """
        :param x: GREY LEVEL OR SPECTRAL IMAGES. shape: 1 x imgw x imgh
        :return: skip activations + z hidden size
        """
        x = F.relu(self.cond_enc_conv0(x))
        x = self.cond_enc_bn0(x)
        x = F.relu(self.cond_enc_conv1(x))
        x = self.cond_enc_bn1(x)
        x = F.relu(self.cond_enc_conv2(x))
        x = self.cond_enc_bn2(x)
        x = F.relu(self.cond_enc_conv3(x))
        x = self.cond_enc_bn3(x)
        # z = F.relu(self.cond_enc_conv4(sc_feat8))
        z = self.cond_enc_conv4(x)
        return z

    def decoder(self, z):
        x = self.dec_upsamp1(z)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_bn1(x)
        x = self.dec_upsamp2(x)
        x = F.relu(self.dec_conv2(x))
        x = self.dec_bn2(x)
        x = self.dec_upsamp3(x)
        x = F.relu(self.dec_conv3(x))
        x = self.dec_bn3(x)
        x = self.dec_upsamp4(x)
        x = F.relu(self.dec_conv4(x))
        x = self.dec_bn4(x)
        x = self.dec_upsamp5(x)

        x = self.final_conv(x)
        return x

    def forward(self, color, inputs):
        """
        when training we accept color and greylevel, they are
        both encoded to z1 and z2. decoder gets z1*z2 in
        to recreate the color image. we also use skips from the b&w image encoder.
        on testing we get only the greyscale image, encoder returns z2.
        a random z1 is sampled and mul is executed. finally the result is decoded to colorize the image
        :param color: AB channel
        :param inputs: L channel or spectral images
        :return: predicted AB channel
        """
        z_grey = self.cond_encoder(inputs)
        if isinstance(color, type(None)):  # TEST TIME
            # z1 is sampled from Normal distribution,
            # we don't have color input on testing!
            z_rand = torch.randn(self.train_batch_size, self.hidden_size, 1, 1).repeat(1, 1, 4, 4).cuda()
            z = z_grey * z_rand
            return self.decoder(z), 0, 0
        else:
            mu, logvar = self.encoder(color)
            stddev = torch.sqrt(torch.exp(logvar))
            eps = torch.randn(stddev.size()).normal_().cuda()
            z_color = torch.add(mu, torch.mul(eps, stddev))
            z_color = z_color.reshape(-1, self.hidden_size, 1, 1).repeat(1, 1, 4, 4)
            z = z_grey * z_color
            return self.decoder(z), mu, logvar