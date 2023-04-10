import torch
from torch import nn
import torch.nn.functional as F


class DownConv(nn.Module):
  """Down sampling with conv."""
  def __init__(self, in_channels, out_channels, kernel_size=4, stride=0, padding=0, neg_slope=0.2, norm=True):
    super().__init__()
    if norm:
      self.convblock = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
          nn.InstanceNorm2d(out_channels),
          nn.LeakyReLU(neg_slope)
      )
    else:
      self.convblock = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
          nn.LeakyReLU(neg_slope)
      )

  def forward(self, x):
    return self.convblock(x)


class UpConv(nn.Module):
  """Up sampling with conv-transpose."""
  def __init__(self, in_channels, out_channels, kernel_size=4, stride=0, padding=0, p=0.5, norm=True):
    super().__init__()
    if norm:
        self.convblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.Dropout(p),
            nn.ReLU()
        )
    else:
        self.convblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Dropout(p),
            nn.ReLU()
        )

  def forward(self, x):
    return self.convblock(x)