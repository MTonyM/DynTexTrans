import torch.nn as nn
import torch.nn.functional as F

from hourglass import HourglassModule
from matutils import make_coordinate_grid


class PrepareForDenseMotion(nn.Module):
    def __init__(self):
        super(PrepareForDenseMotion, self).__init__()

    def forward(self, source_image, kp_driving, kp_source):
        """
        Prepare for the data to input to DenseMotionEstimator.
        :param source_image: [B C H W]
        :param kp_driving: {key_point: {mean: [B K 2], var: [B K 2*2/1*1]}, heatmap: [B K H W]}
        :param kp_source: {key_point: {mean: [B K 2], var: [B K 2*2/1*1]}, heatmap: [B K H W]}
        :return:
        """
        heatmap_drivin = kp_driving['heatmap']  # B K H W
        heatmap_source = kp_source['heatmap']
        batch_size, channel, h, w = heatmap_source.shape
        keypoint_drivin = kp_driving['key_point']['mean']  # B K 2
        keypoint_source = kp_source['key_point']['mean']  # B K 2
        _, num_keypoint, _ = keypoint_source.shape
        # --- normal heatmap (k) ---

        # --- background heatmap ---

        # ------ diff of kp --------
        keypoint_diff = keypoint_drivin - keypoint_source
        # bias of grid.
        keypoint_diff_ext = keypoint_diff.view((batch_size, num_keypoint, 1, 1, 2)).repeat(1, 1, h, w, 1)  # B K H W 2
        origin_grid = make_coordinate_grid((h, w), keypoint_diff_ext.type()).view((1, 1, h, w, 2)).repeat(
            (batch_size, num_keypoint, 1, 1, 1))
        grid = origin_grid + keypoint_diff_ext
        # ---- warped feature   ----
        warped_image = F.grid_sample(source_image, grid, mode='nearest')
        # TODO:...
        inputs = [warped_image, heatmap_drivin, heatmap_source, '...']


class DenseMotionEstimator(nn.Module):
    def __init__(self):
        super(DenseMotionEstimator, self).__init__()
        self.hourglass = HourglassModule(out_channels=1)

    def forward(self, *input):
        pass
