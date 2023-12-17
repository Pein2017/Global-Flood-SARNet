import torch
import torch.nn as nn


class XEDiceLoss(nn.Module):

    def __init__(self, alpha=0.5, EPS=1e-7):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.EPS = EPS

    def forward(self, preds, targets):
        xe_loss = self.cross_entropy(preds, targets)
        dice_loss = self.calculate_dice_loss(preds, targets)
        return self.alpha * xe_loss + (1 - self.alpha) * dice_loss

    def calculate_dice_loss(self, preds, targets):
        targets = targets.float()
        preds = torch.softmax(preds, dim=1)[:, 1]
        intersection = torch.sum(preds * targets)
        union = torch.sum(preds + targets)
        return 1 - (2.0 * intersection) / (union + self.EPS)


# Metric Calculation
def tp_fp_fn(preds, targets):
    tp = torch.sum(preds * targets)
    fp = torch.sum(preds) - tp
    fn = torch.sum(targets) - tp
    return tp.item(), fp.item(), fn.item()
