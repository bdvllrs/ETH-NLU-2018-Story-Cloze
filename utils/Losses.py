
import torch.nn as nn
from models import Discriminator


class DiscriminatorLossCompute:
    def __init__(self, discriminator: Discriminator):
        self.discriminator = discriminator
        self.criterion = nn.BCELoss(size_average=False)

    def compute(self, encoder_output, target):
        log_prob = self.discriminator.forward(encoder_output)
        log_prob = log_prob.view(-1)
        adv_loss = self.criterion(log_prob, target)
        return adv_loss


