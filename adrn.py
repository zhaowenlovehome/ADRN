import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def make_model(args, parent=False):
    return ADRN()

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range=255,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale, upscale_factor):
        super(ResGroup, self).__init__()

        self.gamma1 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.gamma3 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.gamma4 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.r1 = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.r2 = ResBlock(conv, n_feats * 2, kernel_size, act=act, res_scale=res_scale)
        self.r3 = ResBlock(conv, n_feats * 4, kernel_size, act=act, res_scale=res_scale)
        self.compression = nn.Sequential(
            default_conv(n_feats * 8, n_feats, kernel_size),
            nn.ReLU(inplace=True)
        )

        self.attention = NonLocal(n_feats)

    def forward(self, x):
        c0 = x

        r1 = self.r1(c0) * self.gamma1
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.r2(c1) * self.gamma2
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.r3(c2) * self.gamma4
        c3 = torch.cat([c2, r3], dim=1)

        compression = self.compression(c3)
        out = self.attention(compression)
        out = out * 0.2 + x

        return out


class NonLocal(nn.Module):
    def __init__(self, channel, reduction=16):
        super(NonLocal, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.channel = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        kernel_size = 7
        self.height = nn.Sequential(
            nn.Conv2d(1, 1, (kernel_size, 1), 1, padding=((kernel_size-1) // 2, 0), bias=True),
            nn.Sigmoid()
        )
        self.width = nn.Sequential(
            nn.Conv2d(1, 1, (1, kernel_size), 1, padding=(0, (kernel_size-1) // 2), bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        c_pool = self.avg_pool(x)
        c_ = self.channel(c_pool)
        h_pool = x.mean(dim=(1, 3), keepdim=True)
        h_ = self.height(h_pool)
        w_pool = x.mean(dim=(1, 2), keepdim=True)
        w_ = self.width(w_pool)


        out = x * c_ * h_ * w_
        return out


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class ADRN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=25, res_scale=0.2, upscale_factor=4,
                 conv=default_conv):
        super(ADRN, self).__init__()

        n_resblocks = num_blocks
        n_feats = num_features
        kernel_size = 3
        scale = upscale_factor
        act = nn.ReLU(True)
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        # define head module
        self.head = conv(in_channels, n_feats, kernel_size)

        # define body module

        body = [
            ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor) for _ in range(n_resblocks)
        ]

        self.body = nn.Sequential(*body)
        self.upscale = Upsampler(conv, scale, n_feats, act=False)
        # define tail
        self.tail = default_conv(n_feats, out_channels, 3)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.body(x)
        x = self.upscale(x)
        x = self.tail(x)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

