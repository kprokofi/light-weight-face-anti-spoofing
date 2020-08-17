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
    margin_types = ['cos', 'arc', 'adacos']
    def __init__(self, margin_type='cos', device=0, num_classes=2, label_smooth=False, smoothing=0.1, ratio=[1,2.04], gamma=0., m=0.5, s=30, t=1.):
        ''' label smoothing - flag whether or not to use label smoothing '''
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = torch.Tensor([m/i for i in ratio]).cuda(device)
        assert s > 0
        if self.margin_type in ['arc','cos',]:
            self.s = s
        else:
            assert self.margin_type == 'adacos'
            self.s = math.sqrt(2) * math.log(num_classes - 1)
            if self.s <= 1:
                self.s = 5
        # self.cos_m = math.cos(m)
        # self.sin_m = math.sin(m)
        # self.th = math.cos(math.pi - m)
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
            # phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)
        else:
            assert self.margin_type == 'adacos'
            # compute outpute for adacos margin
            phi_theta = torch.acos(cos_theta)
            phi_theta = torch.cos(phi_theta + self.m)
            # one_hot = torch.zeros_like(cos_theta)
            # one_hot.scatter_(1, fold_target.view(-1, 1).long(), 1)
            output = phi_theta
            with torch.no_grad():
                # compute adaptive rescaling parameter
                B_avg = torch.where(one_hot < 1, torch.exp(self.s * cos_theta), torch.zeros_like(cos_theta))
                B_avg = torch.sum(B_avg) / cos_theta.size(0)
                # print(B_avg)
                theta_med = torch.median(phi_theta[one_hot == 1])
                self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))

        if self.gamma == 0 and self.t == 1.:
            pred = F.log_softmax(self.s*output, dim=-1)
            return torch.mean(torch.sum(-target * pred, dim=-1))

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

        return focal_loss(F.cross_entropy(self.s*output, target, reduction='none'), self.gamma)

def test():
    criterion = AMSoftmaxLoss(margin_type='arc')
    cos_teta = torch.randn(3,2)
    print(cos_teta)
    target = torch.randn(3,2)
    print(target)
    loss = criterion(cos_teta, target)
    print(loss)

if __name__ == '__main__':
    test()