""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, ch=64, bilinear=True, dropout=0.2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, ch)
        self.down1 = Down(ch, 2*ch)
        self.down2 = Down(2*ch, 4*ch)
        self.img = FeatureConv(2048, 8*ch, 6*4*ch)
        self.drop = nn.Dropout(p=dropout)
        self.down3 = Down(8*ch, 8*ch)
        self.down4 = Down(8*ch, 8*ch)
        self.up1 = Up(16*ch, 4*ch, bilinear)
        self.up2 = Up(12*ch, 2*ch, bilinear)
        self.up3 = Up(4*ch, ch, bilinear)
        self.up4 = Up(2*ch, ch, bilinear)
        self.outc = OutConv(ch, n_classes)

    def forward(self, scans, features):
        # scans [batch_size, 2, 24, 48] - channels range and return type
        # features [batch_size, 2048, 3, 12]

        x1 = self.inc(scans)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        feats = self.drop(self.img(features))
        x3 = torch.cat([feats.reshape(feats.shape[0], -1, x3.shape[-2], x3.shape[-1]),x3], dim=1)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
