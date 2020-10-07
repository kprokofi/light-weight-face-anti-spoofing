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

from .conv2d_cd import Conv2d_cd
from .dropout import Dropout

__all__ = ['mobilenetv2']

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride, theta):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 3, stride, 1, bias=False, theta=theta),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_3x3_in(inp, oup, stride, theta):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 3, stride, 1, bias=False, theta=theta),
        nn.InstanceNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_in(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.InstanceNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


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


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1., prob_dropout=0.1, type_dropout='bernoulli',
                 prob_dropout_linear=0.5, embeding_dim=1280, mu=0.5, sigma=0.3,
                 theta=0, multi_heads=True, scaling=1):
        super().__init__()
        # setting of inverted residual blocks
        self.multi_heads = multi_heads
        self.scaling = scaling
        self.prob_dropout_linear = prob_dropout_linear
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2, theta=theta)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel,
                                    s if i == 0 else 1, t,
                                    prob_dropout=prob_dropout,
                                    type_dropout=type_dropout,
                                    mu=mu, sigma=sigma, theta=theta))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, embeding_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.spoofer = nn.Linear(embeding_dim, 2)
        if self.multi_heads:
            self.lightning = nn.Linear(embeding_dim, 5)
            self.spoof_type = nn.Linear(embeding_dim, 11)
            self.real_atr = nn.Linear(embeding_dim, 40)

    def forward(self, x):
        x = self.features(x)
        x = self.conv_last(x)
        return x

    def forward_to_onnx(self,x):
        x = self.features(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        spoof_out = self.spoofer(x)
        if isinstance(spoof_out, tuple):
            spoof_out = spoof_out[0]
        probab = F.softmax(spoof_out*self.scaling, dim=-1)
        return probab

    def make_logits(self, features):
        output = self.avgpool(features)
        output = output.view(output.size(0), -1)
        spoof_out = self.spoofer(output)
        if self.multi_heads:
            type_spoof = self.spoof_type(output)
            lightning_type = self.lightning(output)
            real_atr = torch.sigmoid(self.real_atr(output))
            return spoof_out, type_spoof, lightning_type, real_atr
        return spoof_out

    def spoof_task(self, features):
        output = self.avgpool(features)
        output = output.view(output.size(0), -1)
        spoof_out = self.spoofer(output)
        return spoof_out

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)
