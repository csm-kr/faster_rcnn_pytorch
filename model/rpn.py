import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import vgg16_bn
from utils import cxcy_to_xy, decode
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
        pred_cls, pred_loc = pred
        batch_size = pred_cls.size(0)
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] to [B, H, W, C]
        pred_loc = pred_loc.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] to [B, H, W, C]
        pred_cls = pred_cls.reshape(batch_size, -1, 2)
        pred_loc = pred_loc.reshape(batch_size, -1, 4)

        # 2. pred to roi
        roi = pred_bboxes = cxcy_to_xy(decode(pred_loc.squeeze(), anchor)).clamp(0, 1)     # for batch 1, [67995, 4]
        pred_scores = pred_cls.squeeze()                                         # for batch 1, [67995, num_classes]

        # TODO pred scores to softmax or sigmoid
        pred_scores = torch.sigmoid(pred_scores)

        # 2. minimum size keep
        # ws = roi[:, 2] - roi[:, 0]
        # hs = roi[:, 3] - roi[:, 1]
        # TODO np to torch
        # keep = np.where((hs >= self.min_size) & (ws >= self.min_size))[0]
        # roi = roi[keep, :]
        # score = score[keep]

        # 3. nms keep
        sorted_scores, sorted_idx_scores = pred_scores[..., 1].squeeze().sort(descending=True)
        if len(sorted_idx_scores) < pre_nms_top_k:
            pre_nms_top_k = len(sorted_idx_scores)
        roi = roi[sorted_idx_scores[:pre_nms_top_k]]  # [12000, 4]
        roi = roi[:pre_nms_top_k]  # [12000]
        sorted_scores = sorted_scores[:pre_nms_top_k]  # [12000]

        keep_idx = nms(boxes=roi, scores=sorted_scores, iou_threshold=0.7)
        keep = torch.zeros(pre_nms_top_k, dtype=torch.bool)
        keep[keep_idx] = 1  # int64 to bool
        roi = roi[keep][:post_num_top_k]

        return roi


class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        num_anchors = 9
        self.intermediate_layer = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)
        self.region_proposal = RegionProposal()
        self.initialize()

    def initialize(self):
        for c in self.intermediate_layer.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)
        for c in self.cls_layer.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)
        for c in self.reg_layer.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

    def forward(self, features, anchor, mode):

        x = self.intermediate_layer(features)
        cls = self.cls_layer(x)
        reg = self.reg_layer(x)
        roi = self.region_proposal((cls, reg), anchor, mode)
        return cls, reg, roi


if __name__ == '__main__':
    img = torch.randn([2, 3, 1000, 600])  # 62, 37
    rpn = RPN()
    cls, reg = rpn(img)

    print(cls.size())
    print(reg.size())
