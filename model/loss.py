import torch.nn as nn
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def multilabel_soft_margin_loss(output, target):
    return F.multilabel_margin_loss(output, target)


def BCEWithLogitsLoss(output, target):
    critirion = nn.BCEWithLogitsLoss()
    return critirion(output, target)