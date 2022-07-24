import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from utils import *

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, leaky=1):
        super(ResBlock, self).__init__()
        self.skip = None
        self.leaky = leaky
        
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )

        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU() if leaky else nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        
    def forward(self, x):
        act = F.leaky_relu if self.leaky else F.relu 
        return act(self.block(x) + x if self.skip is None else self.skip(x))

class CalibResNet(nn.Module):
    def __init__(self, img_dim, ang_dim):
        super(CalibResNet, self).__init__()
        self.img_dim = img_dim
        self.ang_dim = ang_dim
        
        self.base_cnn = nn.Sequential(
            torch.nn.Sequential(*(list(models.resnet18().children())[:-1])),
        )
        
        self.base_dense = nn.Sequential(
            nn.Flatten(),
            self.linear_block(512, 100),
            self.linear_block(100, 50),
            self.linear_block(50, 10),
            nn.Linear(10, ang_dim)
        )
        
        self.base_dense[-1].bias.data = torch.tensor([0.0277611 * max_scale , 0.02836007 * max_scale])  

    def linear_block(self, in_units, out_units, bias=True):
        return nn.Sequential(
            nn.Linear(in_units, out_units, bias=bias),
            nn.LeakyReLU()
        )

    def forward(self, x, y=None):
        x = self.base_cnn(x)
        p = self.base_dense(x)

        loss = None
        if y is not None: 
            loss = F.mse_loss(p, torch.nan_to_num(y))

        return p, loss
    
class CalibConvNet(nn.Module):
    def __init__(self, img_dim, ang_dim):
        super(CalibConvNet, self).__init__()
        self.img_dim = img_dim
        self.ang_dim = ang_dim
        
        self.base_cnn = nn.Sequential(
            self.cnn_block(img_dim[0], 24, 5, 2),
            self.cnn_block(24, 36, 5, 2),
            self.cnn_block(36, 48, 5, 2),
            self.cnn_block(48, 64, 3, 2),
            self.cnn_block(64, 64, 3, 2),
        )
        
        self.base_dense = nn.Sequential(
            nn.Flatten(),
            self.linear_block(4 * 6 * 64, 100),
            self.linear_block(100, 50),
            self.linear_block(50, 10),
            nn.Linear(10, ang_dim)
        ) 
        
        self.base_dense[-1].bias.data = torch.tensor([0.0277611 * max_scale , 0.02836007 * max_scale]) 
        
    def cnn_block(self, c_in, c_out, k_size, stride, pad=0, bias=False):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, k_size, stride, padding=pad, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(c_out),
        )

    def linear_block(self, in_units, out_units, bias=True):
        return nn.Sequential(
            nn.Linear(in_units, out_units, bias=bias),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y=None):
        x = self.base_cnn(x)
        p = self.base_dense(x)

        loss = None
        if y is not None: 
            loss = F.mse_loss(p, torch.nan_to_num(y))

        return p, loss



