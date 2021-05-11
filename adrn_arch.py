# Enhanced Deep Residual Networks for Single Image Super-Resolution
# https://arxiv.org/abs/1707.02921

import math
import torch
import torch.nn as nn
import numpy as np
from networks.blocks import DeconvBlock


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

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

class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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
                m.append(SELayer(n_feats))

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
        self.r2 = ResBlock(conv, n_feats*2, kernel_size, act=act, res_scale=res_scale)
        self.r3 = ResBlock(conv, n_feats*4, kernel_size, act=act, res_scale=res_scale)
        self.compression = nn.Sequential(
            default_conv(n_feats * 8, n_feats, kernel_size),
            nn.ReLU(inplace=True)
        )
        self.ca = SELayer(n_feats)

        if upscale_factor == 2:
            stride = 2
            padding = 2
            k_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            k_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            k_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            k_size = 12

        self.up = DeconvBlock(n_feats, n_feats,
                    kernel_size=k_size, stride=stride, padding=padding,
                    act_type='relu')

    def forward(self, x):
        c0 = x

        r1 = self.r1(c0)*self.gamma1
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.r2(c1)*self.gamma2
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.r3(c2)*self.gamma4
        c3 = torch.cat([c2, r3], dim=1)

        compression = self.compression(c3)
        out = self.ca(compression)
        out = out + x
        up = self.up(out)
        return out, up

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
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

class GhoustMoudle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dw_size=3, ratio=2):
        super(GhoustMoudle, self).__init__()

        self.out_channels = out_channels
        intrinsic_channels = math.ceil(out_channels/ratio)
        new_channels = intrinsic_channels*(ratio-1)
        self.intrinsic = nn.Sequential(
            nn.Conv2d(in_channels, intrinsic_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(intrinsic_channels, new_channels, dw_size, padding=dw_size//2, groups=intrinsic_channels, bias=False)
        )
        
    def forward(self, x):
        intrinsic = self.intrinsic(x)
        cheap_operation = self.cheap_operation(intrinsic)
        x = torch.cat([intrinsic, cheap_operation], dim=1)
        return x[:, :self.out_channels, :, :]
        
class ADRN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_blocks, res_scale, upscale_factor, conv=default_conv):
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
        self.rg1 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg2 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg3 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg4 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg5 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg6 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg7 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg8 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg9 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg10 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg11 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg12 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg13 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg14 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg15 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg16 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg17 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg18 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg19 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)
        self.rg20 = ResGroup(conv, n_feats, kernel_size, act, res_scale, upscale_factor)

        # define tail
        self.tail = default_conv(n_feats, out_channels, 3)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        up = []
        low1, up1 = self.rg1(x)
        up.append(up1)
        low2, up2 = self.rg2(low1)
        up.append(up2)
        low3, up3 = self.rg3(low2)
        up.append(up3)
        low4, up4 = self.rg4(low3)
        up.append(up4)
        low5, up5 = self.rg5(low4)
        up.append(up5)
        low6, up6 = self.rg6(low5)
        up.append(up6)
        low7, up7 = self.rg7(low6)
        up.append(up7)
        low8, up8 = self.rg8(low7)
        up.append(up2)
        low9, up9 = self.rg9(low8)
        up.append(up9)
        low10, up10 = self.rg10(low9)
        up.append(up10)
        low11, up11 = self.rg11(low10)
        up.append(up11)
        low12, up12 = self.rg12(low11)
        up.append(up12)
        low13, up13 = self.rg13(low12)
        up.append(up13)
        low14, up14 = self.rg14(low13)
        up.append(up14)
        low15, up15 = self.rg15(low14)
        up.append(up15)
        low16, up16 = self.rg16(low15)
        up.append(up16)
        low17, up17 = self.rg17(low16)
        up.append(up17)
        low18, up18 = self.rg18(low17)
        up.append(up18)
        low19, up19 = self.rg19(low18)
        up.append(up19)
        low20, up20 = self.rg20(low19)
        up.append(up20)
        # up = torch.cat(up, dim=1)
        u = up1 + up2 + up3 + up4 + up5 + up6 + up7 + up8 + up9 + up10 + up11 + up12 + up13 + up14 + up15 + up16 + up17 + up18 + up19 + up20
        up = self.tail(u)
        x = self.add_mean(up)

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

