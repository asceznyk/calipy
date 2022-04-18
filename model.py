import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

class CalibNet(nn.Module):
    def __init__(self, img_dim, ang_dim):
        super(CalibNet, self).__init__()
        self.img_dim = img_dim
        self.ang_dim = ang_dim

        self.base_cnn = nn.Sequential(
            self._block(img_dim[0], 12, 5, 2),
            self._block(12, 24, 5, 2),
            self._block(24, 36, 5, 2),
            self._block(36, 48, 5, 2),
            self._block(48, 64, 3, 1),
            self._block(64, 64, 3, 1),
            nn.Flatten(),
        )

        self.base_dense = nn.Sequential(
            self.linear_block(200, 100),
            self.linear_block(100, 50),
            self.linear_block(50, 10),
            self.linear_block(10, ang_dim[0])
        )

        self._w = None

    def cnn_block(self, in_channels, out_channels, k_size, stride, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k_size, stride, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def linear_block(self, in_units, out_units):
        return nn.Sequential(
            nn.Linear(in_units, out_units),
            nn.ReLU()
        )

    def _dense(self, x):
        if not self._w:
            self._w = nn.Parameter(torch.randn(200, x.size()[1]))
        return F.linear(x, weight=self._w) 

    def forward(self, x):
        x = self.base_cnn(x)
        x = self._dense(x)
        x = self.base_dense(x)
        return x











