import torch.nn as nn


class MDN(nn.Module):

    # define layers
    def __init__(self, conf):
        super(MDN, self).__init__()

        self.hidden_size = conf['HIDDENSIZE']
        self.nmix = conf['NMIX']
        self.nout = (self.hidden_size+1)*self.nmix

        # COLORIZATION NET PART
        self.features = nn.Sequential(
            # OLD NET PIECE
            # conv1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),

            # conv2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            # conv3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            # conv4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),

            # NEW MDN Layers
            nn.Conv2d(512, 384, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 320, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(320),
            nn.Conv2d(320, 288, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(288),
            nn.Conv2d(288, 256, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 96, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        # final layers
        self.mdn_dropout1 = nn.Dropout(p=.7)
        self.mdn_fc1 = nn.Linear(4 * 4 * 64, self.nout)

    # define forward pass
    def forward(self, feats):
        x = self.features(feats)
        x = x.view(-1, 4 * 4 * 64)
        x = self.mdn_dropout1(x)
        x = self.mdn_fc1(x)
        return x
