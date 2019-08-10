from torch.optim import Optimizer
import torch

confidence_thresh = 0.96837722
loss = 'var'
cls_bal_scale = False
n_classes = 10
cls_bal_scale_range = 0.0

cls_balance = 0.005
cls_balance_loss = 'bce'


def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def get_cls_bal_function(name):
    if name == 'bce':
        return robust_binary_crossentropy
    elif name == 'log':
        return log_cls_bal
    elif name == 'bug':
        return bugged_cls_bal_bce


cls_bal_fn = get_cls_bal_function(cls_balance_loss)


def compute_aug_loss(stu_out, tea_out, rampup=0, use_rampup=False, rampup_weight_in_list=[0]):
    # Augmentation loss
    if use_rampup:
        unsup_mask = None
        conf_mask_count = None
        unsup_mask_count = None
    else:
        conf_tea = torch.max(tea_out, 1)[0]
        unsup_mask = conf_mask = (conf_tea > confidence_thresh).float()
        unsup_mask_count = conf_mask_count = conf_mask.sum()

    if loss == 'bce':
        aug_loss = robust_binary_crossentropy(stu_out, tea_out)
    else:
        d_aug_loss = stu_out - tea_out
        aug_loss = d_aug_loss * d_aug_loss

    # Class balance scaling
    if cls_bal_scale:
        if use_rampup:
            n_samples = float(aug_loss.shape[0])
        else:
            n_samples = unsup_mask.sum()
        avg_pred = n_samples / float(n_classes)
        bal_scale = avg_pred / torch.clamp(tea_out.sum(dim=0), min=1.0)
        if cls_bal_scale_range != 0.0:
            bal_scale = torch.clamp(bal_scale, min=1.0 / cls_bal_scale_range, max=cls_bal_scale_range)
        bal_scale = bal_scale.detach()
        aug_loss = aug_loss * bal_scale[None, :]

    aug_loss = aug_loss.mean(dim=1)

    if use_rampup:
        unsup_loss = aug_loss.mean() * rampup_weight_in_list[0]
    else:
        unsup_loss = (aug_loss * unsup_mask).mean()

    # Class balance loss
    if cls_balance > 0.0:
        # Compute per-sample average predicated probability
        # Average over samples to get average class prediction
        avg_cls_prob = stu_out.mean(dim=0)
        # Compute loss
        equalise_cls_loss = cls_bal_fn(avg_cls_prob, float(1.0 / n_classes))

        equalise_cls_loss = equalise_cls_loss.mean() * n_classes

        if use_rampup:
            equalise_cls_loss = equalise_cls_loss * rampup_weight_in_list[0]
        else:
            if rampup == 0:
                equalise_cls_loss = equalise_cls_loss * unsup_mask.mean(dim=0)

        unsup_loss += equalise_cls_loss * cls_balance

    return unsup_loss, conf_mask_count, unsup_mask_count


def bugged_cls_bal_bce(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def log_cls_bal(pred, tgt):
    return -torch.log(pred + 1.0e-6)


class EMAWeightOptimizer (object):
    def __init__(self, target_net, source_net, alpha=0.999):
        self.target_net = target_net
        self.source_net = source_net
        self.ema_alpha = alpha
        self.target_params = list(target_net.state_dict().values())
        self.source_params = list(source_net.state_dict().values())

        target_keys = set(target_net.state_dict().keys())
        source_keys = set(source_net.state_dict().keys())
        if target_keys != source_keys:
            raise ValueError('Source and target networks do not have the same state dict keys; do they have different architectures?')

    def step(self):
        one_minus_alpha = 1.0 - self.ema_alpha
        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p.mul_(self.ema_alpha)
            tgt_p.add_(src_p * one_minus_alpha)

