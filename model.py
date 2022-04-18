import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

class CalibNet(nn.Module):
    def __init__(self, img_dim, ang_dim):
        self.img_dim = img_dim
        self.ang_dim = ang_dim

        self.base = nn.Sequential(
            self._block(img_dim[0], 24, 5, 2),
            self._block(24, 36, 5, 2),
            self._block(36, 48, 5, 2),
            self._block(48, 64, 3, 1),
            self._block(64 ,64, 3, 1),
            #nn.Flatten(),
            #nn.Linear()
        ) 

    def _block(self, in_channels, out_channels, k_size, stride, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k_size, stride, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.base(x)
        print(x.size())
        return x











