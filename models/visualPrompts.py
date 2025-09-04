import torch
from torch import nn
from torch.nn import functional as F
from Prompt_UIE import BaseConv2d,initialize_weights

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.conv = BaseConv2d(in_channel=48, out_channel=64, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            BaseConv2d(in_channel=64, out_channel=32, kernel_size=1, stride=1,
                       padding=0, bias=True),
            nn.ReLU(inplace=True),
            BaseConv2d(in_channel=32, out_channel=64, kernel_size=1, stride=1,
                       padding=0, bias=True),
            nn.Sigmoid()
        )

        initialize_weights([self.ka], 0.1)

    def forward(self, x):
        xx = x.repeat(1, 8, 1, 1)  # (1,24,256,256)
        xx = torch.cat([xx, xx], dim=1)  # 1,48,256,256
        x = self.conv(xx)  # 1,64,256,256

        a = self.avg_pool(x)
        a1 = self.ka(a)

        return [a1]
class VisualPrompt(nn.Module):
    def __init__(self):
        super(VisualPrompt, self).__init__()
        self.get1 = getVisualPrompt(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.get2 = getVisualPrompt(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.get3 = getVisualPrompt(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.get4 = getVisualPrompt(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)

    def forward(self, f_list):
        p1 = self.get1(f_list[0])

        p2 = self.get2(f_list[0])

        p3 = self.get3(f_list[0])

        p4 = self.get4(f_list[0])

        return [p1, p2, p3, p4]
class getVisualPrompt(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride,
                 padding=0, dilation=1, groups=1, bias=True, type=4,
                 temprature=30, ratio=2, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.type = type
        self.temprature = temprature
        self.init_weight = init_weight

        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, type, kernel_size=1, bias=False)
        )

        # 权重 & 偏置
        self.weight = nn.Parameter(
            torch.randn(type, out_planes, in_planes // groups, kernel_size, kernel_size),
            requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(type, out_planes), requires_grad=True)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        # 初始化 Attention 的卷积
        for m in self.attention_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 初始化权重
        for i in range(self.type):
            nn.init.constant_(self.weight[i], 0)

    def _get_attention(self, x):
        att = self.attention_net(x).view(x.shape[0], -1)
        return F.softmax(att / self.temprature, dim=-1)

    def forward(self, x):
        bs, _, _, _ = x.shape
        softmax_att = self._get_attention(x)

        # 根据注意力分布加权卷积权重
        weight = self.weight.view(self.type, -1)
        visual_promptW = torch.mm(softmax_att, weight).view(
            bs * self.out_planes, self.in_planes // self.groups,
            self.kernel_size, self.kernel_size
        )

        # 根据注意力分布加权偏置
        bias = self.bias.view(self.type, -1)
        visual_promptB = torch.mm(softmax_att, bias).view(-1)

        return [visual_promptW, visual_promptB]
