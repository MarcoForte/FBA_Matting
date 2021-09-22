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
