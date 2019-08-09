# -*- coding: utf-8 -*-
"""
@Project:   DynTexTrans
@File   :   nnf
@Author :   TonyMao@AILab
@Date   :   2019-08-07
@Desc   :   None
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from hourglass import StackedHourglass


class NNFPredictor(nn.Module):
    def __init__(self, out_channel=3):
        super(NNFPredictor, self).__init__()
        self.nnf_predictor = StackedHourglass(in_channels=6, out_channels=out_channel, inter_channels=256,
                                              stacked_num=3, dropout_rate=0.1, refine=True)
        # NNF predicts. coordinates + /angle1 [:2] [2]

    def forward(self, source, target):
        inputs = torch.cat([source, target], dim=1)
        nnfs = self.nnf_predictor(inputs)
        nnf = nnfs[-1]
        return nnf


class Synthesiser(nn.Module):
    def __init__(self):
        super(Synthesiser, self).__init__()

    def forward(self, source, nnf):
        bs, c, h, w = source.shape
        patch_coordinates = nnf  # B, 2, h, w
        patch_index_i, patch_index_j = torch.split(patch_coordinates, [1, 1], dim=1)  # B, 1, h, w
        grid = torch_make_grid((h, w), center=True, normalized=True)  # B, h, w, 2
        if torch.cuda.is_available():
            grid = grid.cuda()
        grid = torch.repeat_interleave(grid.unsqueeze(0), bs, dim=0)
        grid[:, :, :, 0] += patch_index_i.squeeze()
        grid[:, :, :, 1] += patch_index_j.squeeze()
        grid = torch.clamp(grid, -1, 1)
        return tnf.grid_sample(source, grid, padding_mode='zeros')


class Synthesiser3D(nn.Module):
    # deprecated but still important.
    def __init__(self):
        super(Synthesiser3D, self).__init__()
        self.ps = 5

    def forward(self, source, nnf):
        bs, c, h, w = source.shape
        coordinates, angle = torch.split(nnf, [2, 1], dim=1)  # B, h, w, x
        coordinates = torch.reshape(coordinates.permute(0, 2, 3, 1), (bs, h, w, 1, 1, 2))  # B, h, w, 1, 1, 2
        angle = torch.reshape(angle.permute(0, 2, 3, 1), (bs, h, w, 1, 1, 1))
        patch_index = torch_make_grid((self.ps, self.ps), True).reshape((1, 1, 1, self.ps, self.ps, 2))
        patch_index_i, patch_index_j = torch.split(patch_index, [1, 1], dim=-1)  # 1, 1, 1, ps, ps, 1
        patch_index_iR = patch_index_i * torch.sin(angle * np.pi) - patch_index_j * torch.cos(angle * np.pi)
        patch_index_jR = patch_index_i * torch.cos(angle * np.pi) - patch_index_j * torch.sin(angle * np.pi)
        patch_index_R = torch.cat([patch_index_iR, patch_index_jR], -1)  # B, h, w, ps, ps, 2
        patch_coordinates = coordinates + patch_index_R  # B, h, w, ps, ps,
        patch_index_i, patch_index_j = torch.split(patch_coordinates, [1, 1], dim=-1)  # B, h, w, ps, ps, 1
        patch_index_i = torch.clamp(patch_index_i, 0, h - 1).squeeze(-1)
        patch_index_j = torch.clamp(patch_index_j, 0, w - 1).squeeze(-1)
        source = source.permute(0, 2, 3, 1)
        img_style = []
        for i in range(bs):
            image_style_patches = source[i, patch_index_i.long()[i], patch_index_j.long()[i]]
            img_style.append(image_style_patches)
        img_style = torch.stack(img_style, dim=0)
        synthesis = torch.sum(img_style, dim=[-2, -3], keepdim=False).permute(0, -1, 1, 2)
        return synthesis


def torch_make_grid(shape, center=False, normalized=False):
    x = torch.linspace(0, shape[0] - 1, steps=shape[0]).view((shape[0], 1))
    y = torch.linspace(0, shape[1] - 1, steps=shape[1]).view((1, shape[1]))
    xx = torch.repeat_interleave(x, shape[1], dim=1)
    yy = torch.repeat_interleave(y, shape[0], dim=0)

    if center:
        xx -= shape[0] // 2
        yy -= shape[1] // 2
    if normalized:
        xx /= shape[0]
        yy /= shape[1]
    grid = torch.stack([yy, xx], dim=-1)
    return grid


def main():
    syner = Synthesiser()
    nnfer = NNFPredictor()
    source_t = torch.zeros((10, 3, 64, 64))
    source_t1 = torch.zeros((10, 3, 64, 64))
    nnf = nnfer(source_t, source_t1)
    print(nnf.shape)
    syned = syner(source_t, nnf)
    print(syned.shape)


if __name__ == '__main__':
    main()
