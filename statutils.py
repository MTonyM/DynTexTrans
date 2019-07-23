import torch

from matutils import make_coordinate_grid


def gaussian2kp(heatmap, kp_variance='matrix', clip_variance=None):
    """
    Extract the mean and the variance from a heatmap
    """
    shape = heatmap.shape  # (B C H W)
    heatmap = heatmap.view((*shape, 1)) + 1e-7
    grid = make_coordinate_grid(shape[2:], heatmap.type()).view((1, 1, *shape[2:], 2))
    # grid is the bias for heatmap direction.
    mean = (heatmap * grid).sum(dim=(2, 3))
    kp = {'mean': mean}
    if kp_variance == 'matrix':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        var = var * heatmap.view((*shape, 1, 1))
        var = var.sum(dim=(2, 3))
        if clip_variance:
            # TODO.
            min_norm = torch.tensor(clip_variance).type(var.type())
            sg = smallest_singular(var).unsqueeze(-1)
            var = torch.max(min_norm, sg) * var / sg
        kp['var'] = var
    elif kp_variance == 'single':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = mean_sub ** 2
        var = var * heatmap
        var = var.sum(dim=(2, 3))
        var = var.mean(dim=-1, keepdim=True)
        var = var.unsqueeze(-1)
        kp['var'] = var
    return kp


if __name__ == '__main__':
    data = torch.randn((10, 5, 64, 64))
    kp = gaussian2kp(data, kp_variance='single')
    print(kp['var'].shape, kp['mean'].shape)
    kp = gaussian2kp(data, kp_variance='matrix')
    print(kp['var'].shape, kp['mean'].shape)
