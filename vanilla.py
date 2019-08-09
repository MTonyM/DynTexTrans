# -*- coding: utf-8 -*-
"""
@Project:   DynTexTrans
@File   :   vanilla
@Author :   TonyMao@AILab
@Date   :   2019-08-08
@Desc   :   None
"""

import os

import cv2
import torch
import torch.nn.functional as tnf
from torch.autograd.variable import Variable
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataloader import DynTexFigureTrainDataset
from nnf import NNFPredictor, Synthesiser
from options import TrainOptions
from vis import Table


def train_simple_trans():
    opt = TrainOptions().parse()
    data_root = 'data/processed'
    train_params = {'lr': 0.01, 'epoch_milestones': (100, 500)}
    # dataset = DynTexNNFTrainDataset(data_root, 'flame')
    dataset = DynTexFigureTrainDataset(data_root, 'flame')
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True)
    nnf_conf = 3
    syner = Synthesiser()
    nnfer = NNFPredictor(out_channel=nnf_conf)
    if torch.cuda.is_available():
        syner = syner.cuda()
        nnfer = nnfer.cuda()
    optimizer_nnfer = Adam(nnfer.parameters(), lr=train_params['lr'])
    table = Table()
    for epoch in range(opt.epoch):
        pbar = tqdm(total=len(dataloader), desc='epoch#{}'.format(epoch))
        pbar.set_postfix({'loss': 'N/A'})
        loss_tot = 0.0
        gamma = epoch / opt.epoch

        for i, (source_t, target_t, source_t1, target_t1) in enumerate(dataloader):

            if torch.cuda.is_available():
                source_t = Variable(source_t, requires_grad=True).cuda()
                target_t = Variable(target_t, requires_grad=True).cuda()
                source_t1 = Variable(source_t1, requires_grad=True).cuda()
                target_t1 = Variable(target_t1, requires_grad=True).cuda()
            nnf = nnfer(source_t, target_t)
            if nnf_conf == 3:
                nnf = nnf[:, :2, :, :] * nnf[:, 2:, :, :]  # mask via the confidence
            # --- synthesis ---
            target_predict = syner(source_t, nnf)
            target_t1_predict = syner(source_t1, nnf)
            loss_t = tnf.mse_loss(target_predict, target_t)
            loss_t1 = tnf.mse_loss(target_t1_predict, target_t1)
            loss = loss_t + loss_t1

            optimizer_nnfer.zero_grad()
            loss.backward()
            optimizer_nnfer.step()
            loss_tot += float(loss_t1)

            # ---   vis    ---
            name = os.path.join(data_root, '../result/', str(epoch), '{}.png'.format(str(i)))
            index = str(epoch) + '({})'.format(i)
            if not os.path.exists('/'.join(name.split('/')[:-1])):
                os.makedirs('/'.join(name.split('/')[:-1]))

            cv2.imwrite(name.replace('.png', '_s.png'),
                        (source_t.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            cv2.imwrite(name.replace('.png', '_s1.png'),
                        (source_t1.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            cv2.imwrite(name.replace('.png', '_t.png'),
                        (target_t.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            cv2.imwrite(name.replace('.png', '_p.png'),
                        (target_predict.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            cv2.imwrite(name.replace('.png', '_t1.png'),
                        (target_t1.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            cv2.imwrite(name.replace('.png', '_p1.png'),
                        (target_t1_predict.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            # vis in table
            table.add(index, os.path.abspath(name.replace('.png', '_s.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_s1.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_t.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_t1.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_p.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_p1.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            pbar.set_postfix({'loss': str(loss_tot / (i + 1))})
            pbar.update(1)
        table.build_html('data/')
        pbar.close()


def train_complex_trans():
    opt = TrainOptions().parse()
    data_root = 'data/processed'
    train_params = {'lr': 0.001, 'epoch_milestones': (100, 500)}
    # dataset = DynTexNNFTrainDataset(data_root, 'flame')
    dataset = DynTexFigureTrainDataset(data_root, 'flame')
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True)
    nnf_conf = 3
    syner = Synthesiser()
    nnfer = NNFPredictor(out_channel=nnf_conf)
    flownet = NNFPredictor(out_channel=nnf_conf)
    if torch.cuda.is_available():
        syner = syner.cuda()
        nnfer = nnfer.cuda()
        flownet = flownet.cuda()
    optimizer_nnfer = Adam(nnfer.parameters(), lr=train_params['lr'])
    table = Table()
    for epoch in range(opt.epoch):
        pbar = tqdm(total=len(dataloader), desc='epoch#{}'.format(epoch))
        pbar.set_postfix({'loss': 'N/A'})
        loss_tot = 0.0
        for i, (source_t, target_t, source_t1, target_t1) in enumerate(dataloader):

            if torch.cuda.is_available():
                source_t = Variable(source_t, requires_grad=True).cuda()
                target_t = Variable(target_t, requires_grad=True).cuda()
                source_t1 = Variable(source_t1, requires_grad=True).cuda()
                target_t1 = Variable(target_t1, requires_grad=True).cuda()

            nnf = nnfer(source_t, target_t)
            flow = flownet(source_t, source_t1)
            # mask...
            if nnf_conf == 3:
                nnf = nnf[:, :2, :, :] * nnf[:, 2:, :, :]  # mask via the confidence
                flow = flow[:, :2, :, :] * flow[:, 2:, :, :]
            # --- synthesis ---
            source_t1_predict = syner(source_t, flow)  # flow penalty
            # target_flow = syner(flow, nnf)  # predict flow
            # target_t1_predict = syner(target_t, target_flow)

            loss_t1_f = tnf.mse_loss(source_t1, source_t1_predict)  # flow penalty
            # loss_t1 = tnf.mse_loss(target_t1_predict, target_t1)  # total penalty
            loss = loss_t1_f

            optimizer_nnfer.zero_grad()
            loss.backward()
            optimizer_nnfer.step()
            loss_tot += float(loss)

            # ---   vis    ---
            name = os.path.join(data_root, '../result/', str(epoch), '{}.png'.format(str(i)))
            index = str(epoch) + '({})'.format(i)
            if not os.path.exists('/'.join(name.split('/')[:-1])):
                os.makedirs('/'.join(name.split('/')[:-1]))

            cv2.imwrite(name.replace('.png', '_s.png'),
                        (source_t.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            cv2.imwrite(name.replace('.png', '_s1.png'),
                        (source_t1.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            cv2.imwrite(name.replace('.png', '_t.png'),
                        (target_t.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            cv2.imwrite(name.replace('.png', '_p.png'),
                        (source_t1_predict.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            cv2.imwrite(name.replace('.png', '_t1.png'),
                        (target_t1.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            # cv2.imwrite(name.replace('.png', '_p1.png'),
            #             (target_t1_predict.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))

            # vis in table
            table.add(index, os.path.abspath(name.replace('.png', '_s.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_s1.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_t.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_t1.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_p.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            table.add(index, os.path.abspath(name.replace('.png', '_p1.png')).replace(
                '/mnt/cephfs_hl/lab_ad_idea/maoyiming', ''))
            pbar.set_postfix({'loss': str(loss_tot / (i + 1))})
            pbar.update(1)
        table.build_html('data/')
        pbar.close()


if __name__ == '__main__':
    # train_simple_trans()
    train_complex_trans()
