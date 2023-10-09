import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce

class LossForGenerator(nn.Module):
    """
    L = -log(softmax(D(G(z))))
    """
    def __init__(self):
        super().__init__()

    def forward(self, probs_fake):
        return bce(probs_fake, torch.ones_like(probs_fake))

class LossForDiscriminator(nn.Module):
    """
    L = - log(softmax(D(x))) - log(1 - softmax(D(G(z))))
    """
    def __init__(self):
        super().__init__()

    def forward(self, probs_fake, probs_real):
        loss_fake = bce(probs_fake.detach(), torch.zeros_like(probs_fake))
        loss_real = bce(probs_real, torch.ones_like(probs_fake))

        loss = torch.stack([loss_fake, loss_real]).mean()

        return loss