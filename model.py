import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super(ResBlock, self).__init__()
        self.skip = None
        
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )

        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        
    def forward(self, x):
        return F.relu(self.block(x) + x if self.skip is None else self.skip(x))

class CalibNet(nn.Module):
    def __init__(self, img_dim, ang_dim):
        super(CalibNet, self).__init__()
        self.img_dim = img_dim
        self.ang_dim = ang_dim

        self.base_cnn = nn.Sequential(
            ResBlock(img_dim[0], 4),
            ResBlock(4, 8),
            self.cnn_block(8, 8, 5, 1),
            ResBlock(8, 16),
            self.cnn_block(16, 16, 5, 1),
            ResBlock(16, 32), 
            self.cnn_block(32, 32, 5, 1),
            ResBlock(32, 64),
            self.cnn_block(64, 64, 5, 1),
            ResBlock(64, 128),
            self.cnn_block(128, 128, 5, 2),
            ResBlock(128, 128),
            self.cnn_block(128, 128, 3, 3),
            self.cnn_block(128, 256, 3, 3),
            self.cnn_block(256, 256, 3, 3),
        )
        
        self.base_dense = nn.Sequential(
            nn.Flatten(),
            self.linear_block(256 * 3 * 4, 200),
            self.linear_block(200, 100),
            self.linear_block(100, 50),
            self.linear_block(50, 10),
            nn.Linear(10, ang_dim)
        )
        
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.bias.data.normal_(mean=0.028, std=0.022)

    def cnn_block(self, c_in, c_out, k_size, stride, bias=False):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, k_size, stride, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c_out)
        )

    def linear_block(self, in_units, out_units, bias=True):
        return nn.Sequential(
            nn.Linear(in_units, out_units, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y=None):
        x = self.base_cnn(x)
        p = self.base_dense(x)

        loss = None
        if y is not None: 
            print(p)
            print(torch.nan_to_num(y))
            loss = F.mse_loss(p, torch.nan_to_num(y))

        return p, loss


