import torch
import numpy as np
import torch.nn as nn


def WeightBCE(domain_out, weight_ad, tag, device, num_class=10):
    domain_size = domain_out.size()[0]
    weight_ad = weight_ad.float().to(device)
    all_loss = 0
    for i in range(domain_size):
        dc_target = torch.from_numpy(np.array([[tag]] * num_class)).float().to(device)
        loss = nn.BCELoss(weight=weight_ad.view(-1))(domain_out[i].view(-1), dc_target.view(-1))
        all_loss += loss
    return all_loss / domain_size


class binary_cross_entropy_loss(nn.Module):
    def __init__(self, device='cpu'):
        super(binary_cross_entropy_loss, self).__init__()
        self.one = torch.tensor(1, dtype=torch.float, device=device)
        self._EPSILON = 1e-7

    def forward(self, output, target):
        output = torch.clamp(output, self._EPSILON, 1 - self._EPSILON)
        res = -target * torch.log(output) - (self.one - target) * torch.log(self.one - output)
        return res.mean()


def logits_BCE(domain_outs, target):
    all_loss = 0
    bce = nn.BCELoss()
    for domain_out in domain_outs:
        loss = bce(domain_out.view(-1), target.view(-1))
        all_loss += loss
    return all_loss


def logits_BCE_cat(domain_outs, target):
    bce = nn.BCELoss()
    domain_out_all = None
    target_all = None
    for domain_out in domain_outs:
        if domain_out_all is None:
            domain_out_all = domain_out
            target_all = target
        else:
            domain_out_all = torch.cat([domain_out_all, domain_out], dim=0)
            target_all = torch.cat([target_all, target], dim=0)
    loss = bce(domain_out_all.view(-1), target_all.view(-1))
    return loss
