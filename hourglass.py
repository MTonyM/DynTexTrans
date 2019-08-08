import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Sequential

from SameConv2d import Conv2d as SConv2d


class ResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.input_dim = in_channels
        self.output_dim = out_channels
        self.conv_skip = SConv2d(self.input_dim, self.output_dim, kernel_size=1)
        self.bottle_neck = Sequential(*[
            nn.BatchNorm2d(self.input_dim, momentum=0.9),
            nn.ReLU(inplace=True),
            SConv2d(in_channels=self.input_dim, out_channels=self.output_dim // 2, kernel_size=1, bias=False),
            # => bottle-neck
            nn.BatchNorm2d(self.output_dim // 2, momentum=0.9),
            nn.ReLU(inplace=True),
            SConv2d(in_channels=self.output_dim // 2, out_channels=self.output_dim // 2, kernel_size=3, bias=False,
                    padding=1),
            # End of bottle-neck
            nn.BatchNorm2d(self.output_dim // 2, momentum=0.9),
            nn.ReLU(inplace=True),
            SConv2d(in_channels=self.output_dim // 2, out_channels=self.output_dim, kernel_size=1, bias=False),
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.input_dim == self.output_dim:
            skip = x
        else:
            skip = self.conv_skip(x)
        return self.bottle_neck(x) + skip


class HourglassModule(nn.Module):
    def __init__(self, in_channels=256, out_channels=None, inter_channels=256, depth=5, num_modules=3):
        super().__init__()
        if out_channels is None:
            out_channels = inter_channels
        self.channel_list1 = [in_channels] + [inter_channels] * (num_modules - 1) + [out_channels]
        self.channel_list2 = [in_channels] + [inter_channels] * (num_modules - 1) + [inter_channels]
        assert len(self.channel_list1) == num_modules + 1
        self.depth = depth
        self.num_modules = num_modules
        self.res_list1 = Sequential(
            *[ResidualModule(self.channel_list1[i], self.channel_list1[i + 1]) for i in range(self.num_modules)])
        self.res_list2 = Sequential(
            *[ResidualModule(self.channel_list2[i], self.channel_list2[i + 1]) for i in range(self.num_modules)])
        if self.depth > 1:
            self.sub_hourglass = HourglassModule(inter_channels, out_channels, inter_channels, depth - 1, num_modules)
        else:
            self.res_waist = ResidualModule(inter_channels, out_channels)
        self.out_branch = ResidualModule(out_channels, out_channels)

    def forward(self, x, **kwargs):
        up = self.res_list1(x)
        shape = up.shape
        x = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)(x)
        x = self.res_list2(x)
        if self.depth != 1:
            x = self.sub_hourglass(x)
        else:
            x = self.res_waist(x)
        # x = self.res_list3(x) # Maybe it is not necessary?
        x = self.out_branch(x)
        x = F.interpolate(x, size=shape[2:], mode='nearest')
        return x + up


class StackedHourglass(nn.Module):
    """
    KP detector without gaussian.
    reference: 1. https://blog.csdn.net/qq_38522972/article/details/82958077
               2. http://web.stanford.edu/class/archive/cs/cs221/cs221.1192/2018/restricted/posters/xiaozhg/poster.pdf
               3. `Stacked Hourglass Networks for Human Pose Estimation` https://arxiv.org/pdf/1603.06937.pdf
    """

    def __init__(self, in_channels, out_channels, inter_channels, stacked_num=2, dropout_rate=0.2, refine=False):
        """
        :param in_channels: 3
        :param out_channels: K
        :param inter_channels: 256
        :param stacked_num: 2
        """
        super().__init__()
        self.stacked_num = stacked_num
        self.refine = refine
        self.in_branch = Sequential(*[
            SConv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.post_in_branch = Sequential(*[
            ResidualModule(in_channels=64, out_channels=128),
            ResidualModule(in_channels=128, out_channels=128),
            ResidualModule(in_channels=128, out_channels=inter_channels)
        ])
        self.hg_list = nn.ModuleList(
            [HourglassModule(inter_channels, inter_channels, inter_channels) for _ in range(stacked_num)])
        self.drop_list = nn.ModuleList([nn.Dropout2d(dropout_rate, inplace=True) for _ in range(stacked_num)])
        self.inter_heatmap = nn.ModuleList(
            [SConv2d(in_channels=inter_channels, out_channels=out_channels, kernel_size=1, stride=1) for _ in
             range(stacked_num)])
        self.inter_rechannel = nn.ModuleList([
            SConv2d(in_channels=out_channels, out_channels=inter_channels, kernel_size=1, stride=1) for _ in
            range(stacked_num - 1)])
        self.linear_module = nn.ModuleList([Sequential(*[
            SConv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(inter_channels, momentum=0.9),
            nn.ReLU(inplace=True)
        ]) for _ in range(stacked_num)])
        self.post_linear_module = nn.ModuleList([Sequential(*[
            SConv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1, stride=1),
            # BatchNormalization(momentum=0.9, epsilon=1e-5),
            # Activation('relu')
        ]) for _ in range(stacked_num - 1)])

        self.upsample_r = ResidualModule(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x, **kwargs):
        x_downsample = self.in_branch(x)
        x = self.post_in_branch(x_downsample)
        heat_map_list = []
        for i in range(self.stacked_num):
            hg = self.hg_list[i](x)
            hg = self.drop_list[i](hg)
            hg = self.linear_module[i](hg)
            heat_map = self.inter_heatmap[i](hg)
            heat_map_list.append(heat_map)
            if i != self.stacked_num - 1:
                x_rechanneled = self.inter_rechannel[i](heat_map)
                post_linear = self.post_linear_module[i](hg)
                x = x + post_linear + x_rechanneled
        if self.refine:
            heat_map_list[-1] = F.interpolate(heat_map_list[-1], scale_factor=(2, 2), mode='bilinear',
                                              align_corners=True)
            heat_map_list[-1] = self.upsample_r(heat_map_list[-1])
        return heat_map_list


def test_StackedHourglass():
    import torch
    batch = 10
    input_Spec = torch.randn((batch, 3, 64, 64))
    # f = ResidualModule(3, 64)
    # x = f(input_Spec)
    # f = HourglassModule(in_channels=3, out_channels=10, inter_channels=32, depth=4, num_modules=3)
    # x = f(input_Spec)
    # print(f)
    # print(x.shape)
    f = StackedHourglass(3, 10, 64, 4, 0.2)
    x = f(input_Spec)


if __name__ == '__main__':
    test_StackedHourglass()
