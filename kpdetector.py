import torch.nn as nn
import torch.nn.functional as F

from hourglass import StackedHourglass
from statutils import gaussian2kp


class KeyPointDetector(nn.Module):
    """
    Chap3.2 Unsupervised Keypoint Detection
    """

    def __init__(self, opt):
        super(KeyPointDetector, self).__init__()
        self.heatmap_predictor = StackedHourglass(opt.input_dim, opt.num_keypoints, opt.inter_channels, opt.num_stack,
                                                  opt.ratio_drop)
        self.temperature = opt.temperature

    def forward(self, x):
        heatmap = self.heatmap_predictor(x)[-1]
        heatmap /= self.temperature
        heatmap = F.softmax(heatmap.view((*heatmap.shape[:2], -1)), dim=-1).view(heatmap.shape)
        kp = gaussian2kp(heatmap)
        return {'key_point': kp,
                'heatmap': heatmap}


if __name__ == '__main__':
    import torch
    from options import TrainOptions

    opt = TrainOptions().parse()
    kpd = KeyPointDetector(opt)
    x = torch.randn((4, 3, 64, 64))
    kp, heatmap = kpd(x)
