import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import vgg16_bn
from utils import cxcy_to_xy, decode, xy_to_cxcy
from torchvision.ops.boxes import nms


class RegionProposal(nn.Module):
    def __init__(self, min_size=16/1000):
        super().__init__()
        self.min_size = min_size

    def forward(self, pred, anchor, mode='test'):

        pre_nms_top_k = 12000
        post_num_top_k = 2000
        if mode == 'test':
            pre_nms_top_k = 6000
            post_num_top_k = 300

        # 1. setting for make roi
        pred_cls, pred_loc = pred  # [B, anchor, 2] / [B, anchor, 4]

        # 2. pred to roi
        roi = cxcy_to_xy(decode(pred_loc.squeeze(), xy_to_cxcy(anchor))).clamp(0, 1)       # for batch 1, [67995, 4]
        pred_scores = pred_cls.squeeze()                                         # for batch 1, [67995, num_classes]

        # TODO pred scores to softmax or sigmoid -> must softmax
        # pred_scores = torch.sigmoid(pred_scores) .. (X)
        softmax_pred_scores = torch.softmax(pred_scores, dim=-1)[..., 1]        # foreground rpn score, [num_anchors]

        # 3. keep longer than minimum size
        ws = roi[:, 2] - roi[:, 0]
        hs = roi[:, 3] - roi[:, 1]
        keep = (hs >= self.min_size) & (ws >= self.min_size)  # [17173]
        roi = roi[keep, :]
        softmax_pred_scores = softmax_pred_scores[keep]

        # 4. nms
        sorted_scores, sorted_scores_indices = softmax_pred_scores.sort(descending=True)
        if len(sorted_scores_indices) < pre_nms_top_k:
            pre_nms_top_k = len(sorted_scores_indices)
        roi = roi[sorted_scores_indices[:pre_nms_top_k]]      # [12000, 4]
        roi = roi[:pre_nms_top_k]                             # [12000]
        sorted_scores = sorted_scores[:pre_nms_top_k]         # [12000]
        keep_idx = nms(boxes=roi, scores=sorted_scores, iou_threshold=0.7)
        keep = torch.zeros(pre_nms_top_k, dtype=torch.bool)
        keep[keep_idx] = 1                                    # int64 to bool  # [ex)1735 ~ 2000]
        roi = roi[keep][:post_num_top_k]
        return roi


class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        num_anchors = 9
        self.intermediate_layer = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self.loc_layer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)
        self.region_proposal = RegionProposal()
        # self.initialize()
        normal_init(self.intermediate_layer, 0, 0.01)
        normal_init(self.cls_layer, 0, 0.01)
        normal_init(self.loc_layer, 0, 0.01)

    def initialize(self):
        for c in self.intermediate_layer.children():
            if isinstance(c, nn.Conv2d):
                np.random.seed(0)
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

        for c in self.cls_layer.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

        for c in self.loc_layer.children():
            if isinstance(c, nn.Conv2d):
                # TODO compare 3 random lib
                import torch
                torch.manual_seed(111)
                import random
                random.seed(0)
                import numpy as np
                np.random.seed(0)
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

    def forward(self, features, anchor, mode):

        batch_size = features.size(0)
        x = torch.relu(self.intermediate_layer(features))
        pred_cls = self.cls_layer(x)
        pred_loc = self.loc_layer(x)

        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] to [B, H, W, C]
        pred_loc = pred_loc.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] to [B, H, W, C]
        pred_cls = pred_cls.reshape(batch_size, -1, 2)
        pred_loc = pred_loc.reshape(batch_size, -1, 4)

        rois = self.region_proposal((pred_cls, pred_loc), anchor, mode)
        return pred_cls, pred_loc, rois


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # sequential 일때, (fc
    if type(m) == torch.nn.modules.container.Sequential:
        for m_ in m.children():
            if type(m_) == torch.nn.modules.linear.Linear or type(m_) == torch.nn.modules.conv.Conv2d:
                m_.weight.data.normal_(mean, stddev)
                m_.bias.data.zero_()
    else:
        # x is a parameter
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


if __name__ == '__main__':
    img = torch.randn([2, 3, 1000, 600])  # 62, 37
    rpn = RPN()
    cls, reg = rpn(img)

    print(cls.size())
    print(reg.size())
