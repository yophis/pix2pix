import torch
from torch import nn
import torch.nn.functional as F

from pix2pix_parts import UpConv, DownConv


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        chs = [6, 64, 128, 256, 512]
        downs = []
        for i in range(4):
            layer = DownConv(chs[i], chs[i+1], kernel_size=4, padding=1,
                             stride=(1 if i == 3 else 2), norm=(False if i == 0 else True))
            downs.append(layer)
        self.downs = nn.ModuleList(downs)
        
        self.out_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        for i in range(4):
          x = self.downs[i](x)
        out = F.sigmoid(self.out_conv(x))
        return out


class Generator(nn.Module):
    def __init__(self):
      super().__init__()
      down_chs = [3, 64, 128, 256] + [512]*5
      downs = []
      for i in range(8):
          layer = DownConv(down_chs[i], down_chs[i+1], stride=2, padding=1, 
                           norm=(False if i in [0, 7] else True))
          downs.append(layer)
      self.downs = nn.ModuleList(downs)

      up_chs = [512]*4 + [256, 128, 64]
      ups = [UpConv(512, 512, stride=2, padding=1)]
      for i in range(6):
          layer = UpConv(up_chs[i]*2, up_chs[i+1], stride=2, padding=1, p=(0.5 if i < 3 else 0))
          ups.append(layer)
      ups.append(nn.ConvTranspose2d(64*2, 3, kernel_size=4, stride=2, padding=1))
      self.ups = nn.ModuleList(ups)

    def forward(self, x):
      copies = []
      # contration (down sampling)
      for i in range(8):
        x = self.downs[i](x)
        # print('Contrating:', i+1, x.shape)
        if i < 7:
          copies.append(x)

      # expansion (up sampling)
      for i in range(8):
        if i > 0:  # skip connection
          x = torch.cat([x, copies[7-i]], dim=1)
        x = self.ups[i](x)
      
      out = F.tanh(x)
      return out