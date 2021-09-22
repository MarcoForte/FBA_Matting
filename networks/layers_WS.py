import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-5):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
        self.out_channels = out_channels
        self.eps = eps
    
    def normalize_weight(self):
        weight = F.batch_norm(
            self.weight.view(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        self.weight.data = weight


    def forward(self, x):
        if self.training:
            self.normalize_weight()
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def train(self, mode: bool = True):
        super().train(mode=mode)
        self.normalize_weight()

def norm(dim):
    return nn.GroupNorm(32, dim)


class CustomGroupNorm(torch.nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.randn(num_channels)).float()
        self.bias = nn.Parameter(torch.randn(num_channels)).float()
        

    def forward(self, imput):
        b, _, _, _ = imput.size()
        x = imput.view(b, self.num_groups, -1)
        mean = torch.mean(x, dim=-1, keepdim=True)
        x = x - mean
        std = (torch.mean(torch.pow(x,2), dim=-1, keepdim=True) + self.eps).sqrt()
        x = x / std

        x = x.view(imput.size())
        x = x*self.weight.view(1,self.num_channels,1,1) + self.bias.view(1,self.num_channels,1,1)
        return x