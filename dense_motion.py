import torch
import torch.nn as nn
import torch.nn.functional as F

from hourglass import HourglassModule
from matutils import make_coordinate_grid


class PrepareForDenseMotion(nn.Module):
    """
        Prepare for the data to input to DenseMotionEstimator.
    """
    def __init__(self, channel, num_kp):
        super(PrepareForDenseMotion, self).__init__()
        self.out_channel = (channel + 3) * (num_kp + 1)

    def forward(self, source_image, kp_driving, kp_source):
        """
        Prepare for the data to input to DenseMotionEstimator.
        :param source_image: [B C H W]
        :param kp_driving: {key_point: {mean: [B K 2], var: [B K 2*2/1*1]}, heatmap: [B K H W]}
        :param kp_source: {key_point: {mean: [B K 2], var: [B K 2*2/1*1]}, heatmap: [B K H W]}
        :return:
        """
        heatmap_drivin = kp_driving['heatmap']                                                  # B K H W
        heatmap_source = kp_source['heatmap']
        batch_size, channel, h, w = heatmap_source.shape
        # make source image the same shape with heatmap/flow.
        source_image = F.interpolate(source_image, size=(h, w), mode='nearest')                 # B c h w
        keypoint_drivin = kp_driving['key_point']['mean']                                       # B K 2
        keypoint_source = kp_source['key_point']['mean']                                        # B K 2
        _, num_keypoint, _ = keypoint_source.shape
        # --- normal heatmap (k) ---
        normalized_heatmap_driving = heatmap_drivin / heatmap_drivin.sum(dim=(2, 3), keepdim=True)  # B K H w
        normalized_heatmap_source = heatmap_source / heatmap_source.sum(dim=(2, 3), keepdim=True)
        diff_heatmap = normalized_heatmap_driving - normalized_heatmap_source
        diff_heatmap = diff_heatmap.reshape((batch_size, num_keypoint, h, w, 1))
        # --- background heatmap ---
        bg_heatmap = torch.zeros((batch_size, 1, h, w, 1))
        diff_heatmap = torch.cat([diff_heatmap, bg_heatmap], dim=1)                             # B K+1 h w 1
        # ------ diff of kp --------
        keypoint_diff = keypoint_drivin - keypoint_source
        # bias of grid.
        keypoint_diff_ext = keypoint_diff.reshape((batch_size, num_keypoint, 1, 1, 2)           # B K 1 1 2
                                                  ).repeat(1, 1, h, w, 1)                       # B K H W 2
        bg_keypoint_ext = torch.zeros((batch_size, 1, h, w, 2))
        num_keypoint += 1  # added bg feature.
        keypoint_diff_ext = torch.cat([keypoint_diff_ext, bg_keypoint_ext], dim=1)              # B K+1 H W 2
        origin_grid = make_coordinate_grid((h, w), keypoint_diff_ext.type()
                                           ).reshape((1, 1, h, w, 2)).repeat(batch_size, num_keypoint, 1, 1, 1)
        # ---- deformations... -----
        deformation_approx = origin_grid + keypoint_diff_ext
        # ---- warped feature   ----(deformed source image.)
        appearance_repeat = source_image.reshape((batch_size, 1, -1, h, w)).repeat(             # B 1 C h w
            1, num_keypoint, 1, 1, 1)                                                           # B K+1 C h w
        appearance_repeat = appearance_repeat.reshape((batch_size * num_keypoint, -1, h, w))
        warped_image = F.grid_sample(appearance_repeat, deformation_approx.reshape((batch_size*num_keypoint, h, w, 2))
                                     , padding_mode='border') # B K+1 H W C
        warped_image = warped_image.reshape((batch_size, num_keypoint, h, w, -1))
        inputs = torch.cat([warped_image, diff_heatmap, keypoint_diff_ext], dim=-1)             # B K+1 H W C+3
        inputs = inputs.permute(0, 1, 4, 2, 3).reshape((batch_size, -1, h, w))                  # B (k+1)*(C+3) H W
        # channel = [channel(warped image) + 1(heatmap) + 2(deformation grid)]* (num_keypoint+1)
        return inputs, keypoint_diff_ext


class DenseMotionEstimator(nn.Module):
    """
        Fine Motion Estimator.
    """
    def __init__(self, opt):
        super(DenseMotionEstimator, self).__init__()
        self.data_generator = PrepareForDenseMotion(channel=opt.input_dim, num_kp=opt.num_keypoints)
        self.hourglass = HourglassModule(in_channels=self.data_generator.out_channel,
                                         out_channels=(opt.num_keypoints + 3),
                                         depth=4, num_modules=3)
        self.num_keypoints = opt.num_keypoints

    def forward(self, source_image, kp_driving, kp_source):
        data, keypoint_diff = self.data_generator(source_image, kp_driving, kp_source)          # [] [B K+1 H W 2]
        batch_size, _, h, w = data.shape
        prediction = self.hourglass(data)
        attention = prediction[:, :self.num_keypoints + 1, :, :].reshape(
            (batch_size, self.num_keypoints + 1, h, w, 1))                                      # B K+1 H W 1
        attention = F.softmax(attention, dim=1)
        masked_flow = attention * keypoint_diff                                                 # B K+1 H W 2
        # F_coarse
        masked_flow_sythesised = masked_flow.sum(dim=1)                                         # B H W 2
        # F_residual
        flow_residual = prediction[:, -2:, :, :].permute(0, 2, 3, 1)                            # B H W 2
        # F = F_coarse + F_residual
        flow_final = masked_flow_sythesised + flow_residual                                     # B H W 2
        coordinate_grid = make_coordinate_grid((h, w), type=data.type()).reshape((1, h, w, 2))  # 1 H W 2
        deformation = coordinate_grid + flow_final
        return deformation                                                                      # B H W 2
