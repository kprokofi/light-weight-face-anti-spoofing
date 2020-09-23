'''MIT License

Copyright (C) 2020 Prokofiev Kirill, Sovrasov Vladislav
 
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
from torch.nn import Parameter


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""
    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return (cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7), )


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

def label_smoothing(classes, y_hot, smoothing=0.1, dim=-1):
    lam = 1 - smoothing
    new_y = torch.where(y_hot.bool(), lam, smoothing/(classes-1))
    return new_y

class AMSoftmaxLoss(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc', 'adacos', 'cross_entropy']
    def __init__(self, margin_type='cos', device='cuda:0', num_classes=2, label_smooth=False, smoothing=0.1, ratio=[1,1], gamma=0., m=0.5, s=30, t=1.):
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m >= 0
        self.m = torch.Tensor([m/i for i in ratio]).to(device)
        assert s > 0
        if self.margin_type in ['arc','cos',]:
            self.s = s
        elif self.margin_type == 'adacos':
            self.s = math.sqrt(2) * math.log(num_classes - 1)
            if self.s <= 1:
                self.s = 15
        else:
            assert self.margin_type == 'cross_entropy'
            self.s = 1
        assert t >= 1
        self.t = t
        self.label_smooth = label_smooth
        self.smoothing = smoothing

    def forward(self, cos_theta, target):
        ''' target - one hot vector '''
        if type(cos_theta) == tuple:
            cos_theta = cos_theta[0]
        self.classes = target.size(1)
        if self.label_smooth:
            target = label_smoothing(classes=self.classes, y_hot=target, smoothing=self.smoothing)
        if self.margin_type in ('cos', 'arc', 'adacos'):
            # fold one_hot to one vector [batch size] (need to do it when label smooth or augmentations used)
            fold_target = target.argmax(dim=1)
            # unfold it to one-hot()
            one_hot_target = F.one_hot(fold_target, num_classes=self.classes)
            m = self.m * one_hot_target
            if self.margin_type == 'cos':
                phi_theta = cos_theta - m
                output = phi_theta
            elif self.margin_type == 'arc':
                theta = torch.acos(cos_theta)
                phi_theta = torch.cos(theta + self.m)
                output = phi_theta
            elif self.margin_type == 'adacos':
                # compute outpute for adacos margin
                zero = torch.tensor(0.).to(cos_theta.device)
                phi_theta = torch.where(one_hot_target.bool(), torch.acos(cos_theta), zero)
                with torch.no_grad():
                    # compute adaptive rescaling parameter
                    B_avg = torch.where(one_hot_target < 1, torch.exp(self.s * cos_theta), zero)
                    B_avg = torch.sum(B_avg) / cos_theta.size(0)
                    theta_med = torch.median(torch.sum(phi_theta, dim=1)).item()
                    theta_med = min(math.pi / 4., theta_med)
                    self.s = torch.log(B_avg) / math.cos(theta_med)
                    output = cos_theta
        else:
            assert self.margin_type == 'cross_entropy'
            output = cos_theta

        if self.gamma == 0 and self.t == 1.:
            pred = F.log_softmax(self.s*output, dim=-1)
            return torch.mean(torch.sum(-target * pred, dim=-1))

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

        pred = F.log_softmax(self.s*output, dim=-1)
        return focal_loss(torch.sum(-target * pred, dim=-1), self.gamma)
