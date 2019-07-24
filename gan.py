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
        self.blocks = [DownSampleSConv_BN_Relu2d(in_channels=channel_list[i], out_channels=channel_list[i + 1])
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
        self.blocks = []
        for i in range(num_blocks):
            in_chn = inter_channels
            out_chn = inter_channels
            if i != 0:
                in_chn *= 2
            if i == num_blocks-1:
                out_chn = out_channels
            self.blocks.append(SConv_BN_Relu2d(in_chn, out_chn))

    def deform_feature(self, feature, flow):
        flow = F.interpolate(flow.permute(0, 3, 1, 2), size=feature.shape[2:], mode='nearest')
        feature = F.grid_sample(feature, flow.permute(0, 2, 3, 1), padding_mode='border')
        return feature

    def forward(self, skips, guidance):
        """
        :param skips: [B C H W]
        :param guidance: B H W 2
        :return:
        """
        inter = self.deform_feature(skips[-1], guidance)
        # print(inter.shape, '------')
        for i in range(self.num_blocks):
            # print(i, inter.shape)
            if i == 0:
                inter = self.blocks[i](inter)
                continue
            skip = F.interpolate(skips[-1 - i], size=inter.shape[2:], mode='nearest')
            skip = self.deform_feature(skip, guidance)
            inp_tmp = torch.cat([inter, skip], dim=1)
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


def done():
    from kpdetector import KeyPointDetector
    from options import TrainOptions
    opt = TrainOptions().parse()
    kpd = KeyPointDetector(opt)
    gen = Generator(opt)
    dis_ = Discriminator(opt)

    batch = 10
    source_tensor = torch.randn((batch, 3, 64, 64))
    drivin_tensor = torch.randn((batch, 3, 64, 64))
    print(source_tensor.shape)
    print(drivin_tensor.shape)
    source_kp = kpd(source_tensor)
    drivin_kp = kpd(drivin_tensor)
    gened = gen(source_tensor, drivin_kp, source_kp)
    dis_feat = dis_(gened)
    print('-> disc feature.!')
    for i in range(len(dis_feat)):
        print(dis_feat[i].shape)

if __name__ == '__main__':
    done()
