import torch.nn as nn
import torch.nn.functional as F

from hourglass import StackedHourglass
from deprecated.statutils import gaussian2kp


class KeyPointDetector(nn.Module):
    """
    Chap3.2 Unsupervised Keypoint Detection
    forward :return {'key_point': kp {'mean': B H W 2, 'var': B H W 2x2 }, 'heatmap': heatmap}
    """

    def __init__(self, opt):
        super(KeyPointDetector, self).__init__()
        self.heatmap_predictor = StackedHourglass(opt.input_dim * 2, opt.num_keypoints * 2, opt.inter_channels,
                                                  opt.num_stack,
                                                  opt.ratio_drop)
        self.temperature = opt.temperature
        self.opt = opt

    def kp_gen(self, heatmap):
        heatmap = F.softmax(heatmap.reshape((*heatmap.shape[:2], -1)), dim=-1).reshape(heatmap.shape)
        heatmap = heatmap / self.temperature
        kp = gaussian2kp(heatmap)
        return {'key_point': kp,
                'heatmap': heatmap}

    def forward(self, x):
        heatmap = self.heatmap_predictor(x)[-1]
        source_heatmap = heatmap[:, :self.opt.num_keypoints, :, :]
        target_heatmap = heatmap[:, self.opt.num_keypoints:, :, :]
        source_kp = self.kp_gen(source_heatmap)
        target_kp = self.kp_gen(target_heatmap)
        return source_kp, target_kp


if __name__ == '__main__':
    import torch
    from options import TrainOptions

    opt = TrainOptions().parse()
    kpd = KeyPointDetector(opt)
    x = torch.randn((4, 6, 64, 64))
    kp_s, kp_t = kpd(x)
