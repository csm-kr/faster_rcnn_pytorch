import torch
import numpy as np
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.ops import nms
from anchor import FRCNNAnchorMaker
from utils import xy_to_cxcy, decode, cxcy_to_xy


class RegionProposal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reg, cls, anchor, mode):
        # pred reg, pred cls로 nms 해서 2000 / 300 개 고르는 곳
        # ** 1. set nms top k **
        pre_nms_top_k = 12000
        post_num_top_k = 2000
        if mode == 'test':
            pre_nms_top_k = 6000
            post_num_top_k = 300

        # ** 2. make pred reg to bbox coord using tensor anchor **
        anchor_tensor = torch.from_numpy(anchor)
        roi_tensor = decode(reg,
                            xy_to_cxcy(anchor_tensor.to(reg.get_device()))
                            )
        roi_tensor = cxcy_to_xy(roi_tensor).clamp(0, 1)

        # 3. keep longer than minimum size
        ws = roi_tensor[:, 2] - roi_tensor[:, 0]
        hs = roi_tensor[:, 3] - roi_tensor[:, 1]
        keep = (hs >= (self.min_size / 1000)) & (ws >= (self.min_size / 1000))
        roi_tensor = roi_tensor[keep, :]
        softmax_pred_cls_scores = cls[keep]

        # 4. nms
        sorted_scores, sorted_scores_indices = softmax_pred_cls_scores.sort(descending=True)
        pre_nms_top_k = pre_nms_top_k
        if len(sorted_scores_indices) < pre_nms_top_k:
            pre_nms_top_k = len(sorted_scores_indices)
        roi_tensor = roi_tensor[sorted_scores_indices[:pre_nms_top_k]]  # [12000, 4]
        sorted_scores = sorted_scores[:pre_nms_top_k]                   # [12000]

        # conduct pytorch nms
        keep_idx = nms(boxes=roi_tensor, scores=sorted_scores, iou_threshold=0.7)
        keep_idx = keep_idx[:post_num_top_k]  # tensor([    0,     1,     2,  ..., 11960, 11982, 11997])
        roi_tensor = roi_tensor[keep_idx]

        return roi_tensor


class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=512):
        super().__init__()

        num_anchors = 9
        self.inter_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        self.anchor_maker = FRCNNAnchorMaker()
        self.anchor_base = self.anchor_maker.generate_anchor_base()
        self.region_proposal = RegionProposal()

        normal_init(self.inter_layer, 0, 0.01)
        normal_init(self.cls_layer, 0, 0.01)
        normal_init(self.reg_layer, 0, 0.01)

    def forward(self, features, img_size, mode='train'):

        batch_size = features.size(0)

        # ** 1. make anchor **
        # the shape of anchor - [f_h * f_w * 9 , 4]
        anchor = self.anchor_maker._enumerate_shifted_anchor(anchor_base=np.array(self.anchor_base,
                                                                                  dtype=np.float32),
                                                             origin_image_size=img_size)  # H, W

        # ** 2. forward **
        #                                                            if image size is # [1, 3, 600, 800]
        x = torch.relu(self.inter_layer(features))                                    # [1, 512, 37, 50]

        pred_reg = self.reg_layer(x)                                                  # [1, 36, 37, 50]
        pred_cls = self.cls_layer(x)                                                  # [1, 18, 37, 50]

        pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # [1, 16650, 4]
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)  # [1, 16650, 2]

        rpn_softmax_fg_cls = torch.softmax(pred_cls, dim=-1)[..., 1]                  # [1, 16650]

        # ** 3. make roi **
        # the shape of roi_tensor is [num_roi(<=2000), 4]
        roi_tensor = self.region_proposal(reg=pred_reg.squeeze(0),
                                          cls=rpn_softmax_fg_cls.squeeze(0),
                                          anchor=anchor,
                                          mode=mode)

        return pred_reg, pred_cls, roi_tensor, anchor


class FRCNNHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0


class TargetBuilder:
    def __init__(self, module_name):
        super().__init__()
        self.module_name = module_name

    def build_rpn_target(self, bbox, anchor):
        assert self.module_name == 'rpn'
        return 0

    def build_frcnn_target(self, bbox, label, rois):
        assert self.module_name == 'frcnn'
        return 0


class FRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone
        backbone = vgg16(pretrained=True)

        # extractor
        self.extractor = nn.Sequential(
            *list(backbone.features.children())[:-1]
        )

        # classifier
        self.classifier = np.array(list(backbone.classifier.children()))
        self.classifier = nn.Sequential(
            *list(self.classifier[[0, 1, 3, 4]])
        )

        # freeze top 4 conv
        for layer in self.extractor[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        self.rpn = RegionProposalNetwork()
        self.head = FRCNNHead()
        self.rpn_target_builder = TargetBuilder('rpn')
        self.frcnn_target_builder = TargetBuilder('frcnn')

    def forward(self, x):
        return 0

    def predict(self, x):
        return -1


def normal_init(m, mean, stddev):
    import torch
    torch.manual_seed(111)
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()


if __name__ == '__main__':
    model = FRCNN()
    print(model.extractor)
    print(model.classifier)