import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 8 * 8 * 512)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # main block
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.res(x)
        out += self.shortcut(x)
        return out


class CVAE_skips(nn.Module):

    # define layers
    def __init__(self, conf):
        super(CVAE_skips, self).__init__()
        self.hidden_size = 64
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
        self.dec_conv1 = nn.Conv2d(512 + self.hidden_size, 256, 5, stride=1, padding=2)  # 512 (skips) + z (color emb)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(512, 128, 5, stride=1, padding=2)  # 256 (out) + 256 (skips)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Conv2d(256, 64, 5, stride=1, padding=2)  # 128 (out) + 128 (skips)
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = nn.Conv2d(128, 64, 5, stride=1, padding=2)  # final shape 64 x 64 x 2 (ab channels)
        self.dec_bn4 = nn.BatchNorm2d(64)
        self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, 2, 1, stride=1, padding=0)

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
        sc_feat64 = self.cond_enc_bn0(x)
        x = F.relu(self.cond_enc_conv1(x))
        sc_feat32 = self.cond_enc_bn1(x)
        x = F.relu(self.cond_enc_conv2(sc_feat32))
        sc_feat16 = self.cond_enc_bn2(x)
        x = F.relu(self.cond_enc_conv3(sc_feat16))
        sc_feat8 = self.cond_enc_bn3(x)
        # z = F.relu(self.cond_enc_conv4(sc_feat8))
        z = self.cond_enc_conv4(sc_feat8)
        return sc_feat64, sc_feat32, sc_feat16, sc_feat8, z

    def decoder(self, z, sc_feat64, sc_feat32, sc_feat16, sc_feat8):
        x = self.dec_upsamp1(z)
        x = torch.cat([x, sc_feat8], 1)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_bn1(x)
        x = self.dec_upsamp2(x)
        x = torch.cat([x, sc_feat16], 1)
        x = F.relu(self.dec_conv2(x))
        x = self.dec_bn2(x)
        x = self.dec_upsamp3(x)
        x = torch.cat([x, sc_feat32], 1)
        x = F.relu(self.dec_conv3(x))
        x = self.dec_bn3(x)
        x = self.dec_upsamp4(x)
        x = torch.cat([x, sc_feat64], 1)
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
        sc_feat64, sc_feat32, sc_feat16, sc_feat8, z_grey = self.cond_encoder(inputs)
        if isinstance(color, type(None)):  # TEST TIME
            # z1 is sampled from Normal distribution,
            # we don't have color input on testing!
            z_rand = torch.randn(self.train_batch_size, self.hidden_size, 1, 1).repeat(1, 1, 4, 4).cuda()
            z = z_grey * z_rand
            return self.decoder(z, sc_feat64, sc_feat32, sc_feat16, sc_feat8), 0, 0
        else:
            mu, logvar = self.encoder(color)
            stddev = torch.sqrt(torch.exp(logvar))
            eps = torch.randn(stddev.size()).normal_().cuda()
            z_color = torch.add(mu, torch.mul(eps, stddev))
            z_color = z_color.reshape(-1, self.hidden_size, 1, 1).repeat(1, 1, 4, 4)
            z = z_grey * z_color
            return self.decoder(z, sc_feat64, sc_feat32, sc_feat16, sc_feat8), mu, logvar


class CVAE_noskip(nn.Module):

    # define layers
    def __init__(self, conf):
        super(CVAE_noskip, self).__init__()
        self.hidden_size = 128
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


class CVAE_shallow(nn.Module):

    # define layers
    def __init__(self, conf):
        super(CVAE_shallow, self).__init__()
        self.hidden_size = 128
        self.train_batch_size = conf['TEST_BATCHSIZE']

        # Encoder layers
        self.enc_conv0 = nn.Conv2d(2, 256, kernel_size=3, stride=4, padding=1)
        self.enc_bn0 = nn.BatchNorm2d(256)
        self.enc_conv1 = nn.Conv2d(256, 1024, kernel_size=3, stride=4, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(1024)
        self.enc_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.enc_fc1 = nn.Linear(4 * 4 * 1024, self.hidden_size * 2)

        # Cond encoder layers
        self.cond_enc_conv0 = nn.Conv2d(9, 256, kernel_size=3, stride=4, padding=1)
        self.cond_enc_bn0 = nn.BatchNorm2d(256)
        self.cond_enc_conv1 = nn.Conv2d(256, 1024, kernel_size=3, stride=4, padding=1)
        self.cond_enc_bn1 = nn.BatchNorm2d(1024)
        self.cond_enc_conv2 = nn.Conv2d(1024, self.hidden_size, kernel_size=3, stride=2, padding=1)

        # Decoder layers
        self.dec_upsamp1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(self.hidden_size, 32, 3, stride=1, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_upsamp2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(32, 2, 3, stride=1, padding=1)

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
        x = x.view(-1, 4 * 4 * 1024)
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
        z = F.relu(self.cond_enc_conv2(x))
        return z

    def decoder(self, z):
        x = self.dec_upsamp1(z)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_bn1(x)
        x = self.dec_upsamp2(x)
        x = F.relu(self.dec_conv2(x))
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


class CVAE(nn.Module):

    # define layers
    def __init__(self, conf):
        super(CVAE, self).__init__()
        self.hidden_size = conf['HIDDEN_SIZE']
        self.train_batch_size = conf['TEST_BATCHSIZE']

        self.encoder = nn.Sequential(
            ResidualBlock(2, 64,  stride=2),
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),
            nn.ReLU(),
            ResidualBlock(128, 256, stride=2),
            nn.ReLU(),
            ResidualBlock(256, 512, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(8 * 8 * 512, self.hidden_size * 2)
        )

        # AVOID RELU AFTER LAST RESIDUAL!
        self.cond_encoder = nn.Sequential(
            ResidualBlock(9, 64, stride=2),
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),
            nn.ReLU(),
            ResidualBlock(128, 256, stride=2),
            nn.ReLU(),
            ResidualBlock(256, 512, stride=2),
            nn.ReLU(),
            ResidualBlock(512, self.hidden_size, stride=2),
        )

        self.decoder = nn.Sequential(
            ResidualBlock(self.hidden_size, 64, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            ResidualBlock(64, 32, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            ResidualBlock(32, 16, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            ResidualBlock(16, 8, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),

            nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1),
        )

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
            x = self.encoder(color)
            mu = x[..., :self.hidden_size]
            logvar = x[..., self.hidden_size:]
            stddev = torch.sqrt(torch.exp(logvar))
            eps = torch.randn(stddev.size()).normal_().cuda()
            z_color = torch.add(mu, torch.mul(eps, stddev))
            z_color = z_color.reshape(-1, self.hidden_size, 1, 1).repeat(1, 1, 4, 4)
            z = z_grey * z_color
            return self.decoder(z), mu, logvar