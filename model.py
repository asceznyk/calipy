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
            self.cnn_block(img_dim[0], 12, 5, 2),
            self.cnn_block(12, 24, 5, 2),
            self.cnn_block(24, 36, 5, 2),
            self.cnn_block(36, 48, 5, 2),
            self.cnn_block(48, 64, 3, 1),
            self.cnn_block(64, 64, 3, 1),
            nn.Flatten(),
        )

        self.base_dense = nn.Sequential(
            nn.Linear(64 * 5 * 9, 200),
            self.linear_block(200, 100),
            self.linear_block(100, 50),
            self.linear_block(50, 10),
            nn.Linear(10, ang_dim[0]),
            nn.Sigmoid()
        )

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

    def forward(self, x, y=None):
        x = self.base_cnn(x)
        p = self.base_dense(x)

        loss = None
        if y is not None: 
            print(p)
            print(torch.nan_to_num(y))
            loss = F.binary_cross_entropy(p, torch.nan_to_num(y))

        return p, loss



