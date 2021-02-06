from torch import nn, tanh
from torch.nn import functional as F


def mish(x):
    return x * tanh(F.softplus(x))


class Mish(nn.Module):
    """
    Mish激活函数
    """
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


class DepthSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, bias=True):
        super(DepthSeparableConv2d, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, k, s, p, bias=bias, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)
        )

    def forward(self, x):
        return self.features(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, using_bn=True, act="mish", is_mobile=True):
        super(ConvBNAct, self).__init__()

        self.features = []

        if is_mobile and k > 1:
            conv = DepthSeparableConv2d(in_channels, out_channels, k, s, p, bias=not using_bn)
        else:
            conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=not using_bn)
        self.features.append(conv)

        if using_bn:
            self.features.append(nn.BatchNorm2d(out_channels))

        if act == "mish":
            self.features.append(Mish())
        elif act == "softplus":
            self.features.append(nn.Softplus())
        elif act == "leaky":
            self.features.append(nn.LeakyReLU(0.1))
        elif act == "relu":
            self.features.append(nn.ReLU())
        elif act == "linear":
            pass
        else:
            raise NotImplementedError("目前没有实现{}激活函数".format(act))

        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)


class ResBlock(nn.Module):
    def __init__(self, num_convs, channels, **kwargs):
        super(ResBlock, self).__init__()
        seq = [ConvBNAct(channels, channels, **kwargs) for _ in range(num_convs)]
        self.features = nn.Sequential(*seq)

    def forward(self, x):
        return self.features(x) + x


class ResModule(nn.Module):
    def __init__(self, channels, num_blocks, **kwargs):
        super(ResModule, self).__init__()
        seq = [ResBlock(2, channels, **kwargs) for _ in range(num_blocks)]
        self.features = nn.Sequential(*seq)

    def forward(self, x):
        return self.features(x)
