import torch
from torch import nn
from models.base_structure import ConvBNAct, ResModule
import numpy as np


class BaseLayer(nn.Module):
    def __init__(self, channles, **kwargs):
        super(BaseLayer, self).__init__()
        mid_channels = channles // 2
        self.conv1 = ConvBNAct(3, mid_channels, 3, 1, 1, **kwargs)
        self.conv2 = ConvBNAct(mid_channels, channles, 3, 2, 1, **kwargs)
        self.conv3 = ConvBNAct(channles, mid_channels, 1, 1, 0, **kwargs)
        self.conv4 = ConvBNAct(mid_channels, mid_channels, 3, 1, 1, **kwargs)
        self.conv5 = ConvBNAct(mid_channels, channles, 1, 1, 0, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        y = self.conv3(x)
        y = self.conv4(y)
        y = self.conv5(y)
        return x + y


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, num_res, **kwargs):
        super(DownSample, self).__init__()
        self.extractor = ConvBNAct(in_channels, out_channels, 3, 2, 1, **kwargs)
        self.res = ResModule(out_channels, num_res, **kwargs)
        self.branch_conv = ConvBNAct(out_channels, out_channels, 3, 1, 1, **kwargs)
        self.cat_conv = ConvBNAct(out_channels*2, out_channels, 1, 1, 0, **kwargs)

    def forward(self, x):
        x = self.extractor(x)
        branch1 = self.res(x)
        branch2 = self.branch_conv(x)
        x = torch.cat([branch1, branch2], dim=1)
        x = self.cat_conv(x)
        return x

class Body(nn.Module):
    def __init__(self, **kwargs):
        super(Body, self).__init__()
        self.base = BaseLayer(channles=48, **kwargs)
        self.downsample1 = DownSample(48, 96, num_res=2, **kwargs)
        self.downsample2 = DownSample(96, 96, num_res=3, **kwargs)
        self.downsample3 = DownSample(96, 96, num_res=4, **kwargs)
        self.downsample4 = DownSample(96, 192, num_res=5, **kwargs)

    def forward(self, x):
        y = self.base(x)
        y1 = self.downsample1(y)
        y2 = self.downsample2(y1)
        y3 = self.downsample3(y2)
        y4 = self.downsample4(y3)
        return y2, y3, y4


class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()

    def forward(self, x):
        return x


class Head(nn.Module):
    def __init__(self, out_channels):
        super(Head, self).__init__()
        self.branch1 = nn.Conv2d(96, out_channels, 1, 1)
        self.branch2 = nn.Conv2d(96, out_channels, 1, 1)
        self.branch3 = nn.Conv2d(192, out_channels, 1, 1)

    def forward(self, x1, x2, x3):
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        y3 = self.branch3(x3)

        return y1, y2, y3


class BaseModel(nn.Module):
    def __init__(self, num_classes, is_mobile=True):
        super(BaseModel, self).__init__()

        # box + obj + 类别数
        out_channels = (4 + 1 + num_classes) * 3

        self.body = Body(is_mobile=is_mobile)
        self.head = Head(out_channels)

    def forward(self, x):
        y1, y2, y3 = self.body(x)
        y1, y2, y3 = self.head(y1, y2, y3)
        return y1, y2, y3

if __name__ == '__main__':
    from torchstat import stat
    model = BaseModel(2, True)
    stat(model, (3, 255, 255))