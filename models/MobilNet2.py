''' create MobileNetv2 '''

import torch
import torch.nn as nn
from losses import AngleSimpleLinear

class InvertedResidual(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_channels, out_channels, expansion, stride, use_amsoftmax):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        # expansion
        channels = expansion * in_channels
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        if use_amsoftmax:
            self.relu1 = nn.PReLU()
        else:
            self.relu1 = nn.ReLU()
        # depth wise
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if use_amsoftmax:
            self.relu2 = nn.PReLU()
        else:
            self.relu2 = nn.ReLU()
        # point wise
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # self.dropout = nn.Dropout(p=0.1)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) # linear bottleneck without relu()
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, out_channels, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=2, use_amsoftmax=False):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.use_amsoftmax = use_amsoftmax
        if self.use_amsoftmax:
            self.relu1 = nn.PReLU()
        else:
            self.relu1 = nn.ReLU()
        self.layers = self._make_layers(in_channels=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.relu2 = nn.PReLU()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        if self.use_amsoftmax:
            self.linear = AngleSimpleLinear(1280, num_classes)
        else:
            self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for expansion, out_channels, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(InvertedResidual(in_channels, out_channels, expansion, stride, use_amsoftmax=self.use_amsoftmax))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.GAP(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(10,3,224,224)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
