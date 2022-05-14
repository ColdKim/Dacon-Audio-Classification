import torch.nn as nn
from torch.nn import functional as F
import torch

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(conv_bn_relu, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.relu(x)
        return x
    

class Network(nn.Module):
    def __init__(self, **karg):
        super(Network, self).__init__()
        self.N = karg['N']
        self.AveragePooling = nn.AvgPool2d(2)
        self.MaxPooling = nn.MaxPool2d(2)
        
        
        self.input_conv = conv_bn_relu(in_channels=1, out_channels=self.N, kernel_size=3)
        
        self.block1 = nn.Sequential(
            conv_bn_relu(in_channels=self.N*1, out_channels=self.N*2, kernel_size=3),
            conv_bn_relu(in_channels=self.N*2, out_channels=self.N*4, kernel_size=3),
            conv_bn_relu(in_channels=self.N*4, out_channels=self.N*2, kernel_size=3),
            conv_bn_relu(in_channels=self.N*2, out_channels=self.N*1, kernel_size=3),
        )
        
        self.block2 = nn.Sequential(
            conv_bn_relu(in_channels=self.N*1, out_channels=self.N*2, kernel_size=3),
            conv_bn_relu(in_channels=self.N*2, out_channels=self.N*4, kernel_size=3),
            conv_bn_relu(in_channels=self.N*4, out_channels=self.N*2, kernel_size=3),
            conv_bn_relu(in_channels=self.N*2, out_channels=self.N*1, kernel_size=3),
        )
        
        self.block3 = nn.Sequential(
            conv_bn_relu(in_channels=self.N*1, out_channels=self.N*2, kernel_size=3),
            conv_bn_relu(in_channels=self.N*2, out_channels=self.N*4, kernel_size=3),
            conv_bn_relu(in_channels=self.N*4, out_channels=self.N*2, kernel_size=3),
            conv_bn_relu(in_channels=self.N*2, out_channels=self.N*1, kernel_size=3),
        )
        
        
        self.pool_block = nn.Sequential(
            conv_bn_relu(in_channels=self.N*1, out_channels=self.N*2, kernel_size=3),
            nn.Dropout(0.15),
            nn.AvgPool2d(2),
            conv_bn_relu(in_channels=self.N*2, out_channels=self.N*2, kernel_size=3),
            nn.Dropout(0.15),
            nn.AvgPool2d(2),
            conv_bn_relu(in_channels=self.N*2, out_channels=self.N*2, kernel_size=3),
            nn.Dropout(0.15),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.N*2, out_features=6),
        )
        
    def forward(self, x, out=''):
        x = self.input_conv(x)
        x = self.MaxPooling(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool_block(x)
        x=self.output(x)
        
        if out=='sigmoid':
            x = F.sigmoid(x)
        return x