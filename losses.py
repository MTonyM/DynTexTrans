import torch


def mean_batch(val):
    return val.view(val.shape[0], -1).mean(-1)


def reconstruction_loss(prediction, target, weight, prefix=''):
    """
    Chap3.5.3
    :param prediction: torch tensor [B ? H W]
    :param target:
    :param weight: scalar
    :return:
    """
    if weight == 0:
        return {'_'.join(['L_rec', prefix]): 0}
    return {'_'.join(['L_rec', prefix]): weight * mean_batch(torch.abs(prediction - target))}


def discriminator_loss(generated_map, real_map, weight):
    #  least-square GAN
    predict = generated_map[-1]
    gt = real_map[-1]
    print(predict.shape, gt.shape)
    score = (1 - gt) ** 2 + predict ** 2
    return {'L_D': weight * mean_batch(score)}


def generator_gan_loss(discriminator_maps_generated, weight):
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_generated) ** 2
    return {'L_G_gan': weight * mean_batch(score)}


def generator_loss(discriminator_maps_generated, discriminator_maps_real, loss_weights):
    loss_values = {}
    # Layer 0/ input of target frame.
    loss_values.update(
        reconstruction_loss(discriminator_maps_real[0], discriminator_maps_generated[0],
                            loss_weights['reconstruction_def'], prefix='0'))

    # lambda_rec = 10 as common.
    # Layer ith's feature in discriminator.
    for i, (a, b) in enumerate(zip(discriminator_maps_real[1:-1], discriminator_maps_generated[1:-1])):
        loss_values.update(reconstruction_loss(b, a, weight=loss_weights['reconstruction'], prefix=str(i+1)))

    # print(generator_gan_loss(discriminator_maps_generated, weight=loss_weights['generator_gan']))
    loss_values.update(generator_gan_loss(discriminator_maps_generated, weight=loss_weights['generator_gan']))
    return loss_values
