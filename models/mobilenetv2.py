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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_tools import *

__all__ = ['mobilenetv2']


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,
                 prob_dropout, type_dropout, sigma, mu, theta):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.dropout2d = Dropout(dist=type_dropout, sigma=sigma, mu=mu, p=prob_dropout)
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d_cd(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False, theta=theta),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                Conv2d_cd(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False, theta=theta),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.dropout2d(self.conv(x))
        else:
            return self.dropout2d(self.conv(x))


class MobileNetV2(MobileNet):
    def __init__(self, cfgs, **kwargs):
        super().__init__(**kwargs)
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = make_divisible(32 * self.width_mult, 4 if self.width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2, theta=self.theta)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = make_divisible(c * self.width_mult, 4 if self.width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel,
                                    s if i == 0 else 1, t,
                                    prob_dropout=self.prob_dropout,
                                    type_dropout=self.type_dropout,
                                    mu=self.mu, sigma=self.sigma, theta=self.theta))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.conv_last = conv_1x1_bn(input_channel, self.embeding_dim)


def mobilenetv2(**kwargs):
    """
    Constructs a MobileNetV2 model
    """
    cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
    return MobileNetV2(cfgs, **kwargs)
