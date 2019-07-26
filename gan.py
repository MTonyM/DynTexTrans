import torch
import torch.nn as nn
import torch.nn.functional as F

from SameConv2d import Conv2d as SConv2d
from dense_motion import DenseMotionEstimator
from hourglass import ResidualModule


class SConv_BN_Relu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SConv_BN_Relu2d, self).__init__()
        self.conv = SConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.out_channels = out_channels

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

    def __init__(self, in_channels, inter_channels, block_expasion=4, max_channels=256, num_blocks=3):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        channel_list = [in_channels] + [inter_channels] * num_blocks
        self.blocks = nn.ModuleList(
            [DownSampleSConv_BN_Relu2d(in_channels=channel_list[i], out_channels=channel_list[i + 1]) for i in
             range(num_blocks)])

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
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_chn = inter_channels
            out_chn = inter_channels
            if i != 0:
                in_chn *= 2
            if i == num_blocks - 1:
                out_chn = out_channels
            self.blocks.append(SConv_BN_Relu2d(in_chn, out_chn))
        self.refine = SConv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1, stride=1,
                              padding=0)

    def deform_feature(self, feature, flow):
        # print(feature.shape, flow.shape)
        flow = F.interpolate(flow.permute(0, 3, 1, 2), size=feature.shape[2:], mode='nearest')
        feature = F.grid_sample(feature, flow.permute(0, 2, 3, 1), padding_mode='reflection')
        return feature

    def forward(self, skips, guidance):
        """
        :param skips: [B C H W]
        :param guidance: B H W 2
        :return:
        """
        inter = self.deform_feature(skips[-1], guidance)
        for i in range(self.num_blocks):
            if i == 0:
                inter = self.blocks[i](inter)
                continue
            inter = F.interpolate(inter, size=skips[-1 - i].shape[2:], mode='nearest')
            skip = self.deform_feature(skips[-1 - i], guidance)
            inp_tmp = torch.cat([inter, skip], dim=1)
            inter = self.blocks[i](inp_tmp)
            # print(self.blocks[i].out_channels)
        res = F.interpolate(inter, skips[0].shape[2:], mode='nearest')
        target_def = self.deform_feature(skips[0], guidance)
        # print(res.shape, target_def.shape)
        res = torch.cat([res, target_def], dim=1)
        self.refine(res)
        return target_def


class Generator(nn.Module):
    """
    2btest
    """

    def __init__(self, opt):
        super(Generator, self).__init__()
        # data preparation
        self.dense_motion_estimator = DenseMotionEstimator(opt)
        # encoder
        self.pre_decoder = SConv2d(in_channels=opt.input_dim, out_channels=opt.inter_channels // 2, kernel_size=1,
                                   stride=1, padding=0)
        self.decoder = Decoder(opt.inter_channels, opt.inter_channels // 2)
        self.encoder = Encoder(opt.inter_channels // 2, opt.inter_channels)
        assert self.decoder.num_blocks == self.encoder.num_blocks, 'The number of blocks incompliant'
        self.refinement = nn.Sequential()
        self.refinement.add_module('res_last', ResidualModule(opt.inter_channels // 2, opt.inter_channels // 4))
        self.refinement.add_module('linear_last',
                                   SConv2d(opt.inter_channels // 4, opt.input_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, source_image, kp_driving, kp_source):
        # encode
        dense_flow = self.dense_motion_estimator(source_image, kp_driving, kp_source)
        skips = self.encoder(self.pre_decoder(source_image))
        result = self.decoder(skips, dense_flow)
        result = self.refinement(result)
        return result


class Discriminator(nn.Module):
    """
    2btest
    """

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(in_channels=3, inter_channels=opt.inter_channels)

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
    from losses import generator_loss, discriminator_loss
    from dataloader import DynTexTrainDataset
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import MultiStepLR
    from tqdm import tqdm
    from pprint import pprint
    import cv2
    opt = TrainOptions().parse()
    print('---- training params. ----')
    train_params = {'lr': 0.00001, 'epoch_milestones': (100, 500)}
    loss_weight = {'reconstruction_def': 1.0, 'reconstruction': 10.0, 'generator_gan': 1.0, 'discriminator_gan': 1.0}
    pprint(train_params)
    pprint(loss_weight)

    kpd = KeyPointDetector(opt).train()
    gen = Generator(opt).train()
    dis_ = Discriminator(opt).train()
    start_epoch = 0

    optimizer_generator = torch.optim.Adam(gen.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(dis_.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kpd.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=start_epoch - 1)

    data_root = '/Users/tony/PycharmProjects/DynTexTrans/data/processed'
    dataset = DynTexTrainDataset(data_root, 'flame')
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True)

    for epoch in range(start_epoch, start_epoch + 700):
        pbar = tqdm(total=len(dataloader), desc="=> training epoch # {}".format(epoch), ascii=True, ncols=120)
        pbar.set_postfix({'L_G': 'N/A', 'L_D': 'N/A'})
        for batch, (source_tensor, drivin_tensor) in enumerate(dataloader):
            source_tensor = torch.Tensor.float(source_tensor).requires_grad_()
            drivin_tensor = torch.Tensor.float(drivin_tensor).requires_grad_()

            source_kp, drivin_kp = kpd(torch.cat([source_tensor, drivin_tensor], dim=1))
            gened = gen(source_tensor, drivin_kp, source_kp)
            dis_feat = dis_(gened)
            dis_real = dis_(drivin_tensor)

            loss_g = generator_loss(dis_feat, dis_real, loss_weights=loss_weight)
            loss_g = torch.mean(torch.cat(list(loss_g.values())), dim=0)
            loss_g.backward(retain_graph=True)

            optimizer_generator.step()
            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()

            loss_d = discriminator_loss(dis_feat, dis_real, weight=loss_weight['discriminator_gan'])
            loss_d = torch.mean(torch.cat(list(loss_d.values())), dim=0)
            loss_d.backward()

            optimizer_discriminator.step()
            optimizer_discriminator.zero_grad()

            optimizer_kp_detector.step()
            optimizer_kp_detector.zero_grad()
            # vis
            ge = (gened.detach().numpy()[0].transpose([1, 2, 0]) * 255).astype('uint8')
            # gt = (drivin_tensor.detach().numpy()[0].transpose([1, 2, 0]) * 255).astype('uint8')
            if batch == len(dataloader) - 2:
                root = '/Users/tony/PycharmProjects/DynTexTrans/data/result/'
                cv2.imwrite(root + '{}_{}_ge.png'.format(epoch, batch), ge)
                # cv2.imwrite(root + '{}_{}_gt.png'.format(epoch, batch), gt)
            # vis

            loss_dict = {'L_G': '{:.5f}'.format(float(loss_g)), 'L_D': '{:.5f}'.format(float(loss_d))}
            pbar.set_postfix(loss_dict)
            pbar.update(1)
        scheduler_discriminator.step(epoch)
        scheduler_generator.step(epoch)
        scheduler_kp_detector.step(epoch)
        pbar.close()


if __name__ == '__main__':
    done()
