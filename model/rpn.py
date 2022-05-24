import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import vgg16_bn
from utils import cxcy_to_xy, decode, xy_to_cxcy
from torchvision.ops.boxes import nms


class RegionProposal(nn.Module):
    def __init__(self, min_size=16/800):
        super().__init__()
        self.min_size = min_size

    def forward(self, pred, anchor, mode='test'):

        pre_nms_top_k = 12000
        post_num_top_k = 2000
        if mode == 'test':
            pre_nms_top_k = 6000
            post_num_top_k = 300

        # 1. setting for make roi
        pred_cls, pred_reg = pred  # [B, anchor, 2] - [1, 16650, 2] / [B, anchor, 4] - [1, 16650, 4]

        # 2. pred to roi
        roi = cxcy_to_xy(decode(pred_reg.squeeze(), xy_to_cxcy(anchor))).clamp(0, 1)       # for batch 1, [67995, 4]
        pred_scores = pred_cls  # .squeeze()                                         # for batch 1, [67995, num_classes]

        # TODO pred scores to softmax or sigmoid -> must softmax
        # pred_scores = torch.sigmoid(pred_scores) .. (X)
        # softmax_pred_scores = torch.softmax(pred_scores, dim=-1)[..., 1].contiguous()        # foreground rpn score, [num_anchors]
        softmax_pred_scores = torch.softmax(pred_scores.view(1, pred_scores.size(1), pred_scores.size(2), 9, 2), dim=4)
        softmax_pred_scores = softmax_pred_scores[:, :, :, :, 1].contiguous()
        softmax_pred_scores = softmax_pred_scores.view(1, -1)

        # 3. keep longer than minimum size
        ws = roi[:, 2] - roi[:, 0]
        hs = roi[:, 3] - roi[:, 1]
        keep = (hs >= self.min_size) & (ws >= self.min_size)  # [17173]
        print(keep)
        roi = roi[keep, :]
        softmax_pred_scores = softmax_pred_scores[0][keep]

        # 4. nms
        sorted_scores, sorted_scores_indices = softmax_pred_scores.sort(descending=True)
        if len(sorted_scores_indices) < pre_nms_top_k:
            pre_nms_top_k = len(sorted_scores_indices)
        roi = roi[sorted_scores_indices[:pre_nms_top_k]]      # [12000, 4]
        # roi = roi[:pre_nms_top_k]                             # [12000]
        sorted_scores = sorted_scores[:pre_nms_top_k]         # [12000]

        # nms
        keep_idx = nms(boxes=roi, scores=sorted_scores, iou_threshold=0.7)
        keep_idx = keep_idx[:post_num_top_k]   # tensor([    0,     1,     2,  ..., 11960, 11982, 11997])
        roi = roi[keep_idx]
        return roi


class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        num_anchors = 9
        self.intermediate_layer = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)
        self.region_proposal = RegionProposal()
        # self.initialize()
        normal_init(self.intermediate_layer, 0, 0.01)
        normal_init(self.cls_layer, 0, 0.01)
        normal_init(self.reg_layer, 0, 0.01)

    def forward(self, features, anchor, mode):

        batch_size = features.size(0)
        x = torch.relu(self.intermediate_layer(features))
        pred_cls = self.cls_layer(x)
        pred_reg = self.reg_layer(x)

        pred_cls_ = pred_cls.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] to [B, H, W, C]
        pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] to [B, H, W, C]
        pred_cls = pred_cls_.reshape(batch_size, -1, 2)
        pred_reg = pred_reg.reshape(batch_size, -1, 4)

        rois = self.region_proposal((pred_cls_, pred_reg), anchor, mode)
        return pred_cls, pred_reg, rois


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # sequential 일때, (fc
    if type(m) == torch.nn.modules.container.Sequential:
        for m_ in m.children():
            if type(m_) == torch.nn.modules.linear.Linear or type(m_) == torch.nn.modules.conv.Conv2d:
                torch.manual_seed(111)
                m_.weight.data.normal_(mean, stddev)
                m_.bias.data.zero_()
    else:
        # x is a parameter
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            torch.manual_seed(111)
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()


if __name__ == '__main__':
    img = torch.randn([2, 3, 1000, 600])  # 62, 37
    rpn = RPN()
    cls, reg = rpn(img)

    print(cls.size())
    print(reg.size())
