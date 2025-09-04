# -*- coding: utf-8 -*-
import numbers
import numpy as np
import torch.nn.init as init
import math

from torch.nn import LeakyReLU

from visualPrompts import *
from torch import nn


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class BaseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, bias_init=0):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
        else:
            self.bias = None

    def forward(self, input):
        b, c, h, w = input.shape
        if c != self.in_channel:
            raise ValueError('Input channel is not equal with conv in_channel')
        out = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, scale, residual_lr=1, kernel_size=3):
        super(ResBlock, self).__init__()
        if scale == 3:
            self.conv1_1 = BaseConv2d(in_channel=in_channel, out_channel=in_channel // 4, kernel_size=1,
                                      stride=1,
                                      padding=0, bias=True)
            self.conv3_3 = BaseConv2d(in_channel=in_channel // 4, out_channel=in_channel // 4, kernel_size=kernel_size,
                                      stride=1,
                                      padding=kernel_size // 2, bias=True)
            self.conv1_2 = BaseConv2d(in_channel=in_channel // 4, out_channel=in_channel // 2, kernel_size=1,
                                      stride=1,
                                      padding=0, bias=True)
            self.conv1_3 = BaseConv2d(in_channel=in_channel, out_channel=in_channel // 2, kernel_size=1, stride=1,
                                      padding=0, bias=True)
        else:
            self.conv1_1 = BaseConv2d(in_channel=in_channel, out_channel=in_channel // 4, kernel_size=1,
                                      stride=1,
                                      padding=0, bias=True)
            self.conv3_3 = BaseConv2d(in_channel=in_channel // 4, out_channel=in_channel // 4,
                                      kernel_size=kernel_size, stride=1,
                                      padding=kernel_size // 2, bias=True)
            self.conv1_2 = BaseConv2d(in_channel=in_channel // 4, out_channel=in_channel * scale,
                                      kernel_size=1, stride=1,
                                      padding=0, bias=True)
            self.conv1_3 = BaseConv2d(in_channel=in_channel, out_channel=in_channel * scale, kernel_size=1, stride=1,
                                      padding=0, bias=True)
        self.lrelu = LeakyReLU(0.2)
        self.residual_lr = residual_lr

        initialize_weights([self.conv1_1, self.conv3_3, self.conv1_2, self.conv1_3], 0.1)

    def forward(self, input_features):
        res = self.conv1_1(input_features)
        res = self.lrelu(res)
        res = self.conv3_3(res)
        res = self.lrelu(res)
        res = self.conv1_2(res)

        input_features = self.conv1_3(input_features)
        out = res + input_features
        return out

class IdentityBlock(nn.Module):
    def __init__(self, in_channel, scale=1, residual_lr=1, kernel_size=3):
        super(IdentityBlock, self).__init__()
        self.conv1_1 = BaseConv2d(in_channel=in_channel, out_channel=in_channel // 2, kernel_size=1,
                                  stride=1,
                                  padding=0, bias=True)
        self.conv3_3 = BaseConv2d(in_channel=in_channel // 2, out_channel=in_channel // 2,
                                  kernel_size=kernel_size, stride=1,
                                  padding=kernel_size // 2, bias=True)
        self.conv1_2 = BaseConv2d(in_channel=in_channel // 2, out_channel=in_channel * scale,
                                  kernel_size=1, stride=1,
                                  padding=0, bias=True)
        self.lrelu = LeakyReLU(0.2)
        self.residual_lr = residual_lr

        initialize_weights([self.conv1_1, self.conv3_3, self.conv1_2], 0.1)

    def forward(self, input_features):
        res = self.conv1_1(input_features)
        res = self.lrelu(res)
        res = self.conv3_3(res)
        res = self.lrelu(res)
        res = self.conv1_2(res)

        out = res + input_features
        return out


class ResLayer(nn.Module):
    def __init__(self, in_channel, scale):
        super(ResLayer, self).__init__()
        if scale==3:
            out_channel = in_channel//2
        else:
            out_channel = in_channel * scale
        self.conv1 = ResBlock(in_channel=in_channel, scale=scale)
        self.conv2 = IdentityBlock(in_channel=out_channel)

    def forward(self, input_features):
        res = self.conv1(input_features)
        res = self.conv2(res)
        return res

class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class MFF(nn.Module):
    def __init__(self, in_channels, scale):
        super(MFF, self).__init__()
        if scale==3:
            out_channels = in_channels//2
        else:
            out_channels = in_channels*scale
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.conv5_5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        )


    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv3_3(x)
        x3 = self.conv5_5(x)

        res = x1 + x2 + x3
        return res

class CBAM(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=7):
        super(CBAM, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class TridentBlock(nn.Module):
    def __init__(self, c1, c2, stride=1, c=False, e=0.5, padding=[1, 2, 3], dilate=[1, 2, 3], bias=False):
        super(TridentBlock, self).__init__()
        self.stride = stride
        self.c = c
        c_ = int(c2 * e)
        self.padding = padding
        self.dilate = dilate
        self.share_weightconv1 = nn.Parameter(torch.Tensor(c_, c1, 1, 1))
        self.share_weightconv2 = nn.Parameter(torch.Tensor(c2, c_, 3, 3))

        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)

        self.act = nn.SiLU()

        nn.init.kaiming_uniform_(self.share_weightconv1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.share_weightconv2, nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.Tensor(c2))
        else:
            self.bias = None

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward_for_small(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[0],
                                   dilation=self.dilate[0])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_middle(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[1],
                                   dilation=self.dilate[1])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_big(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[2],
                                   dilation=self.dilate[2])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward(self, x):
        xm = x
        base_feat = []
        if self.c is not False:
            x1 = self.forward_for_small(x)
            x2 = self.forward_for_middle(x)
            x3 = self.forward_for_big(x)
        else:
            x1 = self.forward_for_small(xm[0])
            x2 = self.forward_for_middle(xm[1])
            x3 = self.forward_for_big(xm[2])

        base_feat.append(x1)
        base_feat.append(x2)
        base_feat.append(x3)

        return base_feat
class RFEM(nn.Module):
    def __init__(self, channel, n=1, e=0.5, stride=1):
        super(RFEM, self).__init__()
        c = True
        c1 = channel
        layers = []
        layers.append(TridentBlock(c1, c1, stride=stride, c=c, e=e))
        for i in range(1, n):
            layers.append(TridentBlock(c1, c1))
        self.layer = nn.Sequential(*layers)
        # self.cv = Conv(c1, c1)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.layer(x)
        out = out[0] + out[1] + out[2] + x
        out = self.act(self.bn(out))
        return out

class FFM(nn.Module):
    def __init__(self, channel, stride=1, c=False, e=0.5, bias=False):
        super(FFM, self).__init__()
        self.stride = stride
        self.c = c

        self.share_weightconv1 = nn.Conv2d(channel, channel // 2, kernel_size=1, bias=bias)
        self.share_weightconv2 = nn.Conv2d(channel // 2, channel, kernel_size=3, stride=1, padding=1, bias=bias)

        self.pixNorm = PixelwiseNorm()

        initialize_weights([self.share_weightconv1, self.share_weightconv2], 0.1)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(channel))
        else:
            self.bias = None

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward_e(self, x):
        out = self.share_weightconv1(x)

        out = self.share_weightconv2(out)
        out = F.gelu(out)
        return out

    def forward_d(self, x):
        out = self.share_weightconv1(x)
        out = self.share_weightconv2(out)

        return out


    def forward(self, e, d):
        x1 = self.forward_e(self.pixNorm(e))
        x2 = self.forward_d(self.pixNorm(d))
        x3 = x1 * x2 + x1
        return x3

class PIM(nn.Module):
    def __init__(self):
        super(PIM, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x, out, prompt):
        bs, c, h, w = x.shape
        out1 = out.view(1, -1, h, w)
        out1 = F.conv2d(out1, weight=prompt[0], bias=prompt[1], stride=1, padding=1,
                        groups=bs, dilation=1)
        out1 = out1.view(bs, 64, h, w)
        out2 = self.conv0(out1)

        out2 = out2.view(1, -1, h, w)
        out2 = F.conv2d(out2, weight=prompt[0], bias=prompt[1], stride=1, padding=1,
                        groups=bs, dilation=1)
        out2 = out2.view(bs, 64, h, w)
        out = self.conv1(out2)

        return out

class Adapter(nn.Module):
    def __init__(self, in_channel):
        super(Adapter, self).__init__()
        self.conv_down = BaseConv2d(in_channel, 64, 1, 1, 0)
        self.elu = nn.ELU()
        self.conv_up = BaseConv2d(64, in_channel, 1, 1, 0)

        self.PIM = PIM()

    def forward(self, x, prompt):
        out = self.conv_down(x)
        out = self.elu(out)
        out1 = self.PIM(x, out, prompt)

        out = self.conv_up(out1) + x

        return out

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

        self.ResLayer1 = ResLayer(in_channel=64, scale=2)
        self.ResLayer1_1 = ResLayer(in_channel=128, scale=1)

        self.ResLayer2 = ResLayer(in_channel=128, scale=2)
        self.ResLayer2_1 = ResLayer(in_channel=256, scale=1)

        self.ResLayer3 = ResLayer(in_channel=256, scale=2)
        self.ResLayer3_1 = ResLayer(in_channel=512, scale=1)

        self.ResLayer4 = ResLayer(in_channel=512, scale=1)
        self.ResLayer4_1 = ResLayer(in_channel=512, scale=1)

        self.ResLayer6 = ResLayer(in_channel=512, scale=1)
        self.ResLayer6_1 = ResLayer(in_channel=512, scale=1)

        self.ResLayer7 = ResLayer(in_channel=512, scale=3)
        self.ResLayer7_1 = ResLayer(in_channel=256, scale=1)

        self.ResLayer8 = ResLayer(in_channel=256, scale=3)
        self.ResLayer8_1 = ResLayer(in_channel=128, scale=1)

        self.ResLayer9 = ResLayer(in_channel=128, scale=3)
        self.ResLayer9_1 = ResLayer(in_channel=64, scale=1)

        self.down1 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down2 = nn.Conv2d(256, 256, 3, 2, 1)
        self.down3 = nn.Conv2d(512, 512, 3, 2, 1)
        self.down4 = nn.Conv2d(512, 512, 3, 2, 1)

        self.CBA1 = CBAM(channel=128)
        self.CBA2 = CBAM(channel=256)
        self.CBA3 = CBAM(channel=512)
        self.CBA4 = CBAM(channel=512)
        self.CBA6 = CBAM(channel=512)
        self.CBA7 = CBAM(channel=256)
        self.CBA8 = CBAM(channel=128)
        self.CBA9 = CBAM(channel=64)

        self.fem1 = RFEM(channel=128)
        self.fem2 = RFEM(channel=256)
        self.fem3 = RFEM(channel=512)
        self.fem4 = RFEM(channel=512)
        self.fem6 = RFEM(channel=512)
        self.fem7 = RFEM(channel=256)
        self.fem8 = RFEM(channel=128)
        self.fem9 = RFEM(channel=64)

        self.ffm1 = FFM(512)
        self.ffm2 = FFM(512)
        self.ffm3 = FFM(256)
        self.ffm4 = FFM(128)

        self.adapter1 = Adapter(512)
        self.adapter2 = Adapter(256)
        self.adapter3 = Adapter(128)
        self.adapter4 = Adapter(64)

    def forward(self, x, prompt):
        e1 = self.ResLayer1(x)
        e1 = self.ResLayer1_1(e1)
        e1 = self.CBA1(e1)
        e1 = self.fem1(e1)
        e2 = self.down1(e1)

        e2 = self.ResLayer2(e2)
        e2 = self.ResLayer2_1(e2)
        e2 = self.CBA2(e2)
        e2 = self.fem2(e2)
        e3 = self.down2(e2)

        e3 = self.ResLayer3(e3)
        e3 = self.ResLayer3_1(e3)
        e3 = self.CBA3(e3)
        e3 = self.fem3(e3)
        e4 = self.down3(e3)

        e4 = self.ResLayer4(e4)
        e4 = self.ResLayer4_1(e4)
        e4 = self.CBA4(e4)
        e4 = self.fem4(e4)
        e5 = self.down4(e4)

        from torch.nn.functional import interpolate
        d6 = interpolate(e5, scale_factor=2, mode="bilinear")
        d6 = self.ffm1(e4, d6)
        d6 = self.ResLayer6(d6)
        d6 = self.ResLayer6_1(d6)
        d6 = self.CBA6(d6)
        d6 = self.fem6(d6)
        d6 = self.adapter1(d6, prompt[0])

        d7 = interpolate(d6, scale_factor=2, mode="bilinear")
        d7 = self.ffm2(e3, d7)
        d7 = self.ResLayer7(d7)
        d7 = self.ResLayer7_1(d7)
        d7 = self.CBA7(d7)
        d7 = self.fem7(d7)
        d7 = self.adapter2(d7, prompt[1])

        d8 = interpolate(d7, scale_factor=2, mode="bilinear")
        d8 = self.ffm3(e2, d8)
        d8 = self.ResLayer8(d8)
        d8 = self.ResLayer8_1(d8)
        d8 = self.CBA8(d8)
        d8 = self.fem8(d8)
        d8 = self.adapter3(d8, prompt[2])

        d9 = interpolate(d8, scale_factor=2, mode="bilinear")
        d9 = self.ffm4(e1, d9)
        d9 = self.ResLayer9(d9)
        d9 = self.ResLayer9_1(d9)
        d9 = self.CBA9(d9)
        d9 = self.fem9(d9)
        d9 = self.adapter4(d9, prompt[3])

        return d9

class UIE(nn.Module):
    def __init__(self):
        super(UIE, self).__init__()
        self.kernel_size = 3

        self.input_conv = BaseConv2d(in_channel=3, out_channel=64, kernel_size=self.kernel_size, stride=1,
                                     padding=self.kernel_size // 2, bias=True)

        self.output_conv = BaseConv2d(in_channel=64, out_channel=3, kernel_size=self.kernel_size,
                                      stride=1, padding=self.kernel_size // 2, bias=True)

        self.myModel = myModel()
        self.FeatureExtrator = FeatureExtraction()
        self.getViaulPrompt = VisualPrompt()

    def forward(self, x):

        features = self.FeatureExtrator(x)
        prompt = self.getViaulPrompt(features)
        x = self.input_conv(x)
        x = self.myModel(x, prompt)
        degraded = self.output_conv(x)

        return degraded