import torch
import torch.nn as nn


def gan_loss(prediction, target_is_real, device):
    target_real_label = 1.0
    target_fake_label = 0.0
    gan_mode = "lsgan"
    if target_is_real:
        target = torch.tensor(target_real_label)
    else:
        target = torch.tensor(target_fake_label)
    target = target.expand_as(prediction).to(device)
    if gan_mode == "lsgan":
        loss = nn.MSELoss()

    return loss(prediction, target)
