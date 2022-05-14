import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class RPNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.smooth_l1_loss = SmoothL1Loss()

    def forward(self, pred_cls, pred_loc, target_cls, target_loc):
        rpn_cls_loss = self.cross_entropy_loss(pred_cls, target_cls, ignore_index=-1)
        rpn_loc_loss = self.smooth_l1_loss(pred_loc[target_cls > 0], target_loc[target_cls > 0])

        return rpn_cls_loss, rpn_loc_loss

class FastRCNNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.smooth_l1_loss = SmoothL1Loss()

    def forward(self, pred_cls, pred_loc, target_cls, target_loc):
        fast_rcnn_cls_loss = self.cross_entropy_loss(pred_cls, target_cls)
        fast_rcnn_loc_loss = self.smooth_l1_loss(pred_loc[target_cls > 0], target_loc[target_cls > 0])

        return fast_rcnn_cls_loss, fast_rcnn_loc_loss


class FRCNNLoss(torch.nn.Module):
    def __init__(self, coder):
        super().__init__()
        self.rpn_loss = RPNLoss()
        self.fast_rcnn_loss = FastRCNNLoss()

    def forward(self, pred, target):

        pred_rpn_cls, pred_rpn_loc, pred_fast_rcnn_cls, pred_fast_rcnn_loc = pred
        target_rpn_cls, target_rpn_loc, target_fast_rcnn_cls, target_fast_rcnn_loc = target

        rpn_cls_loss, rpn_loc_loss = self.rpn_loss(pred_rpn_cls, pred_rpn_loc, target_rpn_cls, target_rpn_loc)
        fast_rcnn_cls_loss, fast_rcnn_loc_loss = self.fast_rcnn_loss(pred_fast_rcnn_cls, pred_fast_rcnn_loc,
                                                                     target_fast_rcnn_cls, target_fast_rcnn_loc)
        total_loss = rpn_cls_loss.sum() + rpn_loc_loss.sum() + fast_rcnn_cls_loss.sum() + fast_rcnn_loc_loss.sum()

        return total_loss

