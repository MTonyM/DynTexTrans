import torch
import torch.nn as nn
import torch.nn.functional as F

from SameConv2d import Conv2d as SConv2d
from dense_motion import DenseMotionEstimator

class SConv_BN_Relu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SConv_BN_Relu2d, self).__init__()
        self.conv = SConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class DownSampleSConv_BN_Relu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(DownSampleSConv_BN_Relu2d, self).__init__()
        self.conv = SConv_BN_Relu2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Encoder(nn.Module):
    """
    2btest
    """
    def __init__(self, in_channels, inter_channels, num_blocks=3):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        channel_list = [in_channels] + [inter_channels] * num_blocks
        self.blocks = [DownSampleSConv_BN_Relu2d(in_channels=channel_list[i], out_channels=channel_list[i+1])
                       for i in range(num_blocks)]

    def forward(self, x):
        out_list = [x]
        for i in range(self.num_blocks):
            out_list.append(self.blocks[i](out_list[-1]))
        return out_list


class Decoder(nn.Module):
    """
    2btest
    """
    def __init__(self, inter_channels, out_channels, num_blocks=3):
        super(Decoder, self).__init__()
        self.num_blocks = num_blocks
        channel_list = [inter_channels] * num_blocks + [out_channels]
        self.blocks = [SConv_BN_Relu2d(in_channels=channel_list[i], out_channels=channel_list[i+1])
                     for i in range(num_blocks)]

    def deform_feature(self, feature, flow):
        flow = F.interpolate(flow, size=feature.shape, mode='nearest')
        feature = F.grid_sample(feature, flow)
        return feature

    def forward(self, skips, guidance):
        """
        :param skips: [B C H W]
        :param guidance: B H W 2
        :return:
        """
        inter = self.deform_feature(skips[-1], guidance)
        for i in range(self.num_blocks):
            inp_tmp = torch.cat([inter, skips[-1-i]], dim=1)
            inter = self.blocks[i](inp_tmp)
        return inter


class Generator(nn.Module):
    """
    2btest
    """
    def __init__(self, opt):
        super(Generator, self).__init__()
        # data preparation
        self.dense_motion_estimator = DenseMotionEstimator(opt)
        # encoder
        self.decoder = Decoder(256, 3)
        self.encoder = Encoder(3, 256)
        assert self.decoder.num_blocks == self.encoder.num_blocks, 'The number of blocks incompliant'
        # decoder

    def forward(self, source_image, kp_driving, kp_source):
        # encode
        dense_flow = self.dense_motion_estimator(source_image, kp_driving, kp_source)
        skips = self.encoder(source_image)
        result = self.decoder(skips, dense_flow)
        return result


class Discriminator(nn.Module):
    """
    2btest
    """
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(in_channels=3, inter_channels=256)

    def forward(self, x):
        """
        :param x:
        :return: feature maps.
        """
        features = self.encoder(x)
        return features

