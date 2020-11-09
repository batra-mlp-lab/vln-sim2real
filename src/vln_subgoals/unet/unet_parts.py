""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F



class RadialPad(nn.Module):
  """ Circular padding of heading, replication padding of range."""

  def __init__(self, padding):
    """
    Args:
        padding (int): the size of the padding."""
    super(RadialPad, self).__init__()
    self.padding = padding

  def forward(self, x):
    # x has shape [batch, channels, range_bins, heading_bins]
    x1 = F.pad(x, [0, 0, self.padding, self.padding], mode='replicate')
    x2 = F.pad(x1, [self.padding, self.padding, 0, 0], mode='circular')
    return x2


class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.double_conv = nn.Sequential(
      RadialPad(1),
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      RadialPad(1),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)


class FeatureConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, mid_channels, out_channels):
    super(FeatureConv, self).__init__()
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(mid_channels, out_channels, kernel_size=(3,1), padding=0),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)


class Down(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels):
    super(Down, self).__init__()
    self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)


class Up(nn.Module):
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels, bilinear=True):
    super(Up, self).__init__()

    # if bilinear, use the normal convolutions to reduce the number of channels
    if bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    else:
      self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    # Radial padding
    x1 = F.pad(x1, [0, 0, diffY // 2, diffY - diffY // 2], mode='replicate')
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, 0, 0], mode='circular')
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)
