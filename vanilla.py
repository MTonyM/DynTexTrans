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
import torch.nn.functional as tnf
from torch.autograd.variable import Variable
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataloader import DynTexNNFTrainDataset
from nnf import NNFPredictor, Synthesiser
from options import TrainOptions


def train():
    opt = TrainOptions().parse()
    data_root = '/Users/tony/PycharmProjects/DynTexTrans/data/processed'
    train_params = {'lr': 0.01, 'epoch_milestones': (100, 500)}
    dataset = DynTexNNFTrainDataset(data_root, 'flame')
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True)
    nnf_conf = 3
    syner = Synthesiser()
    nnfer = NNFPredictor(out_channel=nnf_conf)
    optimizer_nnfer = Adam(nnfer.parameters(), lr=train_params['lr'])
    for epoch in range(opt.epoch):
        pbar = tqdm(total=len(dataloader), desc='epoch#{}'.format(epoch))
        pbar.set_postfix({'loss': 'N/A'})
        loss_tot = 0.0
        gamma = epoch / opt.epoch

        for i, (source_t, target_t, source_t1, target_t1) in enumerate(dataloader):
            source_t = Variable(source_t, requires_grad=True)
            target_t = Variable(target_t, requires_grad=True)
            source_t1 = Variable(source_t1, requires_grad=True)
            target_t1 = Variable(target_t1, requires_grad=True)
            nnf = nnfer(source_t, target_t)
            if nnf_conf == 3:
                nnf = nnf[:, :2, :, :] * nnf[:, 2:, :, :]  # mask via the confidence
            # --- synthesis ---
            target_predict = syner(source_t, nnf)
            target_t1_predict = syner(source_t1, nnf)

            loss = tnf.mse_loss(target_predict, target_t) + gamma * tnf.mse_loss(target_t1_predict, target_t1)

            optimizer_nnfer.zero_grad()
            loss.backward()
            optimizer_nnfer.step()
            loss_tot += float(tnf.mse_loss(target_t1_predict, target_t1))

            # ---   vis    ---
            name = os.path.join(data_root, '../result', '{}.png'.format(str(i)))
            cv2.imwrite(name.replace('.png', '_t1P.png'),
                        (target_t1_predict.detach().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))
            cv2.imwrite(name.replace('.png', '_t1T.png'),
                        (target_t1.detach().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))
            cv2.imwrite(name.replace('.png', '_t0T.png'),
                        (target_t.detach().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))
            pbar.set_postfix({'loss': str(loss_tot / (i + 1))})
            pbar.update(1)

        pbar.close()


if __name__ == '__main__':
    train()
