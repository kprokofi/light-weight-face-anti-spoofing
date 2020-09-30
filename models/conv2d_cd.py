'''MIT License

Copyright (C) 2020 Prokofiev Kirill

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0):

        super().__init__()
        self.theta = theta
        self.bias = bias or None
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        if self.groups > 1:
            self.weight = nn.Parameter(kaiming_init(out_channels, in_channels//in_channels, kernel_size))
        else:
            self.weight = nn.Parameter(kaiming_init(out_channels, in_channels, kernel_size))
        self.padding = padding
        self.i = 0

    def forward(self, x):
        out_normal = F.conv2d(input=x, weight=self.weight, bias=self.bias, dilation=self.dilation,
                              stride=self.stride, padding=self.padding, groups=self.groups)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.bias, dilation=self.dilation,
                                stride=self.stride, padding=0, groups=self.groups)
            return out_normal - self.theta * out_diff

def kaiming_init(c_out, c_in, k):
    return torch.randn(c_out, c_in, k, k)*math.sqrt(2./c_in)
