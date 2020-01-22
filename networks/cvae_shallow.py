import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):

    # define layers
    def __init__(self, conf):
        super(CVAE, self).__init__()
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