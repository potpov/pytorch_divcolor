import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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

        self.mean = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.logvar = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # AVOID RELU AFTER LAST RESIDUAL!
        self.cond_encoder = nn.Sequential(

            nn.Conv2d(9, 9, kernel_size=7, stride=1, padding=3),

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
            # nn.Tanh()
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
            # x = self.encoder(color)
            # mu = x[..., :self.hidden_size]
            # logvar = x[..., self.hidden_size:]

            h = self.encoder(color)
            mu = self.mean(h)
            logvar = self.logvar(h)

            stddev = torch.sqrt(torch.exp(logvar))
            eps = torch.randn(stddev.size()).normal_().cuda()
            z_color = torch.add(mu, torch.mul(eps, stddev))

            z_color = z_color.reshape(-1, self.hidden_size, 1, 1).repeat(1, 1, 4, 4)
            z = z_grey * z_color
            return self.decoder(z), mu, logvar