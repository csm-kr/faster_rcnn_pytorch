import torch
import numpy as np
import torch.nn as nn
from utils.util import xy_to_cxcy, cxcy_to_xy, encode, decode, find_jaccard_overlap

# torchvision
import torchvision
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights



class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.min_size = 10
        self.rpn_head = RPNHead()
        self.anchor_generator = torchvision.models.detection.rpn.\
            AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                            aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    def forward(self, x, features, mode):

        cls = []
        reg = []

        features = list(features.values())
        for feature in features:
            # pred
            cls_, reg_ = self.rpn_head(feature)
            cls.append(cls_)
            reg.append(reg_)
            # anchor

        cls = pred_rpn_cls = torch.cat(cls, dim=1).flatten(0, -2)
        reg = pred_rpn_reg = torch.cat(reg, dim=1).reshape(-1, 4)

        device = cls.get_device()
        # for using anchor generator
        h, w = x.shape[2:]
        anchor = self.anchor_generator(ImageList(x, [(w, h)]), features)[0]
        anchor /= torch.FloatTensor([w, h, w, h]).to(device)

        # 0. make foreground_softmax_pred_rpn_cls
        cls = torch.softmax(cls, dim=-1)[..., 1]  # [16650]

        # pred reg, pred cls로 nms 해서 2000 / 300 개 고르는 곳
        # 1. set nms top k
        pre_nms_top_k = 4000
        post_num_top_k = 1000
        if mode == 'test':
            pre_nms_top_k = 2000
            post_num_top_k = 1000

        # 2. make pred reg to bbox coord using tensor anchor
        roi_tensor = decode(reg,
                            xy_to_cxcy(anchor)
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
        roi_tensor = roi_tensor[keep_idx].detach()  # ** important : detach function makes normalization possible **

        return pred_rpn_cls, pred_rpn_reg, roi_tensor, anchor


class RPNHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256):
        super().__init__()

        num_anchors = 3
        self.inter_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        normal_init(self.inter_layer, 0, 0.01)
        normal_init(self.cls_layer, 0, 0.01)
        normal_init(self.reg_layer, 0, 0.01)

    def forward(self, features):
        # features 가 dict이면,
        # 기존
        batch_size = features.size(0)
        # if image size is # [1, 3, 600, 800]
        x = torch.relu(self.inter_layer(features))                                    # [1, 512, 37, 50]
        pred_cls = self.cls_layer(x)                                                  # [1, 18, 37, 50]
        pred_reg = self.reg_layer(x)                                                  # [1, 36, 37, 50]
        pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # [1, 16650, 4]
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)  # [1, 16650, 2]
        return pred_cls, pred_reg


class FRCNNHead(nn.Module):
    def __init__(self,
                 num_classes,
                 roi_size,
                 classifier
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = nn.Linear(1024, num_classes)  # roi 에 대하여 클래스를 만들어야 하므로
        self.reg_head = nn.Linear(1024, num_classes * 4)  # 각 클래스별로 coord 를 만들어야 하므로
        self.roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        self.classifier = classifier

        # initialization
        normal_init(self.cls_head, 0, 0.01)
        normal_init(self.reg_head, 0, 0.001)

    def forward(self, features, roi, img_shape):
        # ** roi 가 0 ~ 1 사이의 값으로 들어온다. --> scale roi **
        device = roi.get_device()
        h, w = img_shape  # torch.Size([800, 800])
        scale_from_roi_to_feature = torch.FloatTensor([w, h, w, h]).to(device)
        sclaed_roi = roi * scale_from_roi_to_feature
        scaled_roi_list = [sclaed_roi]  # for make it input of roi pool - list tensor [[512,4]]

        # ** roi pool **
        pool = self.roi_pool(features, scaled_roi_list, [(w, h)])  # [512, 256, 7, 7]
        x = pool.view(pool.size(0), -1)  # 512, 12544

        # ** fast rcnn forward head ** #
        x = self.classifier(x)  # 512, 1024
        pred_fast_rcnn_cls = self.cls_head(x)  # 512, 91
        pred_fast_rcnn_reg = self.reg_head(x)  # 512, 91 * 4
        return pred_fast_rcnn_cls, pred_fast_rcnn_reg


class FRCNNTargetMaker(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bbox, label, rois):
        # 1. concatenate bbox and roi -
        # remove the list for batch
        bbox = bbox[0]
        label = label[0]

        # 무조건 나오는 roi를 만들기 위해서 와 bbox를 concat 한다.
        rois = torch.cat([rois, bbox], dim=0)
        # print(rois.size())
        iou = find_jaccard_overlap(rois, bbox)  # [2000 + num_obj, num objects]
        IoU_max, IoU_argmax = iou.max(dim=1)

        # set background label 0
        fast_rcnn_tg_cls = label[IoU_argmax] + 1

        # n_pos = 32 or IoU 0.5 이상
        n_pos = int(min((IoU_max >= 0.5).sum(), 128))

        # random select pos and neg indices
        device = bbox.get_device()
        pos_index = torch.arange(IoU_max.size(0), device=device)[IoU_max >= 0.5]
        perm = torch.randperm(pos_index.size(0))
        pos_index = pos_index[perm[:n_pos]]
        n_neg = 512 - n_pos

        neg_index = torch.arange(IoU_max.size(0), device=device)[(IoU_max < 0.5) & (IoU_max >= 0.0)]
        perm = torch.randperm(neg_index.size(0))
        neg_index = neg_index[perm[:n_neg]]

        assert n_neg + n_pos == 512

        # print("pos neg : ", n_pos, n_neg)
        keep_index = torch.cat([pos_index, neg_index], dim=-1)

        # if len(keep_index) != 512:
        #     print(keep_index.shape)

        # make CLS target
        fast_rcnn_tg_cls = fast_rcnn_tg_cls[keep_index]
        # set negative indices background label
        fast_rcnn_tg_cls[n_pos:] = 0
        fast_rcnn_tg_cls = fast_rcnn_tg_cls.type(torch.long)

        # make roi
        sample_rois = rois[keep_index, :]
        # make REG target
        fast_rcnn_tg_reg = encode(xy_to_cxcy(bbox[IoU_argmax][keep_index]), xy_to_cxcy(sample_rois))

        # normalization bbox
        mean = torch.FloatTensor([0., 0., 0., 0.]).to(device)
        std = torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(device)
        fast_rcnn_tg_reg = (fast_rcnn_tg_reg - mean) / std

        return fast_rcnn_tg_cls, fast_rcnn_tg_reg, sample_rois


# def assign_rpn_target(anchors, targets):
#     labels = []
#     matched_gt_boxes = []
#
#     # batch에 따른 target을 위해서
#     for anchors_per_image, targets_per_image in zip(anchors, targets):
#         # 1. get gt boxes
#         gt_boxes = targets_per_image["boxes"]
#
#         # 2. get iou between gt_boxes and anchors_p_i
#         iou = find_jaccard_overlap(gt_boxes, anchors_per_image)  # num_boxes(objects), num_anchors
#         iou_valmax, iou_argmax = iou.max(dim=0)
#         all_matches = iou_argmax.clone()
#
#         # Assign candidate matches with low quality to negative (unassigned) values
#         below_low_threshold = iou_valmax < 0.3
#         between_thresholds = (iou_valmax >= 0.3) & (iou_valmax < 0.7)
#         iou_argmax[below_low_threshold] = -1
#         iou_argmax[between_thresholds] = -2
#
#         # For each gt, find the prediction with which it has the highest quality
#         highest_iou_vals_per_gt, _ = iou.max(dim=1)
#         # Find the highest quality match available, even if it is low, including ties
#
#         # 이 방식
#         gt_pred_pairs_of_highest_quality = torch.where(iou == highest_iou_vals_per_gt[:, None])
#         # Example gt_pred_pairs_of_highest_quality:
#         #   tensor([[    0, 39796],
#         #           [    1, 32055],
#         #           [    1, 32070],
#         #           [    2, 39190],
#         #           [    2, 40255],
#         #           [    3, 40390],
#         #           [    3, 41455],
#         #           [    4, 45470],
#         #           [    5, 45325],
#         #           [    5, 46390]])
#         # Each row is a (gt index, prediction index)
#         # Note how gt items 1, 2, 3, and 5 each have two ties
#
#         pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
#         iou_argmax[pred_inds_to_update] = all_matches[pred_inds_to_update]
#
#
#         matched_idxs = iou_argmax
#         matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
#
#         labels_per_image = matched_idxs >= 0
#         labels_per_image = labels_per_image.to(dtype=torch.float32)
#
#         # Background (negative examples)
#         bg_indices = matched_idxs == -1
#         labels_per_image[bg_indices] = 0.0
#
#         # discard indices that are between thresholds
#         inds_to_discard = matched_idxs == -2
#         labels_per_image[inds_to_discard] = -1.0
#
#
#     pos_idx = []
#     neg_idx = []
#     for matched_idxs_per_image in matched_idxs:
#         positive = torch.where(matched_idxs_per_image >= 1)[0]
#         negative = torch.where(matched_idxs_per_image == 0)[0]
#
#         num_pos = int(256 * 0.5)
#         # protect against not enough positive examples
#         num_pos = min(positive.numel(), num_pos)
#         num_neg = 256 - num_pos
#         # protect against not enough negative examples
#         num_neg = min(negative.numel(), num_neg)
#
#         # randomly select positive and negative examples
#         perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
#         perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
#
#         pos_idx_per_image = positive[perm1]
#         neg_idx_per_image = negative[perm2]
#
#         # create binary mask from indices
#         pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
#         neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
#
#         pos_idx_per_image_mask[pos_idx_per_image] = 1
#         neg_idx_per_image_mask[neg_idx_per_image] = 1
#
#         pos_idx.append(pos_idx_per_image_mask)
#         neg_idx.append(neg_idx_per_image_mask)


class RPNTargetMaker(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bbox, anchor):

        # 1. anchor cross boundary 만 걸러내기
        bbox = bbox[0]  # remove the list for batch : shape [num_obj, 4]
        anchor_keep = ((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) & (anchor[:, 2] <= 1) & (anchor[:, 3] <= 1))
        anchor = anchor[anchor_keep]
        num_anchors = anchor.size(0)

        # 2. iou 따라 label 만들기
        # if label is 1 (positive), 0 (negative), -1 (ignore)
        device = bbox.get_device()
        label = -1 * torch.ones(num_anchors, dtype=torch.float32, device=device)

        iou = find_jaccard_overlap(anchor, bbox)  # [num anchors, num objects]
        IoU_max, IoU_argmax = iou.max(dim=1)

        # 2-1. set negative label
        label[IoU_max < 0.3] = 0

        # 2-2. set positive label that have highest iou.
        _, IoU_argmax_per_object = iou.max(dim=0)

        # FIXME
        # ** max 값이 여러개 있다면(동일하게), 그것을 가져오는 부분.   **
        # IoU_argmax_per_object update (max 값 포함하는 index 찾기)

        # 검증
        # print((iou == IoU_max_per_object).sum() == iou.size(1))
        # print((iou == IoU_max_per_object).sum())
        # print(iou.size(1))

        # ** 2020/08/17 n_neg = 0 인 error -> 무시 **
        # IoU_argmax_per_object = torch.nonzero(input=(iou == IoU_max_per_object))[:, 0]  # 2차원이라 앞의 column 가져오기
        # FIXME

        label[IoU_argmax_per_object] = 1
        # 2-3. set positive label
        label[IoU_max >= 0.7] = 1
        # 2-4 sample target
        n_pos = (label == 1).sum()
        n_neg = (label == 0).sum()

        # print(n_pos)
        # print(n_neg)

        # num_pos = min(positive.numel(), num_pos)

        if n_pos > 128:

            pos_indices = torch.arange(label.size(0), device=device)[label == 1]
            perm = torch.randperm(pos_indices.size(0))
            label[pos_indices[perm[128:]]] = -1  # convert pos label to ignore label

        if n_neg > 256 - n_pos:
            if n_pos > 128:
                n_pos = 128
            neg_indices = torch.arange(label.size(0), device=device)[label == 0]
            perm = torch.randperm(neg_indices.size(0))
            label[neg_indices[perm[(256 - n_pos):]]] = -1  # convert neg label to ignore label

        # assert (label == 1).sum() + (label == 0).sum() > 200, \
        #     'less than 200 addition? pos : {} vs neg : {}'.format((label == 1).sum(), (label == 0).sum())
        if (label == 1).sum() + (label == 0).sum() < 200:
            print('less than 200 addition? pos : {} vs neg : {}'.format((label == 1).sum(), (label == 0).sum()))

        # 3. bbox encoding
        tg_cxywh = encode(xy_to_cxcy(bbox[IoU_argmax]), xy_to_cxcy(anchor))

        # 4. pad label and bbox for ignore label
        pad_label = -1 * torch.ones(len(anchor_keep), dtype=torch.float32, device=device)
        keep_indices = torch.arange(len(anchor_keep), device=device)[anchor_keep]
        pad_label[keep_indices] = label
        rpn_tg_cls = pad_label.type(torch.long)

        pad_bbox = torch.zeros([len(anchor_keep), 4], dtype=torch.float32, device=device)
        pad_bbox[keep_indices] = tg_cxywh
        rpn_tg_reg = pad_bbox

        # The size of rpn_tg_cls / rpn_tg_reg : [16650] / [16650, 4]
        return rpn_tg_cls, rpn_tg_reg


class FRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        # backbone
        self.backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.IMAGENET1K_V1, trainable_layers=3)
        self.classifier = nn.Sequential(nn.Linear(in_features=12544, out_features=1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=1024, out_features=1024),
                                        nn.ReLU(inplace=True))

        # region proposal network
        self.rpn = RegionProposalNetwork()
        # rpn target
        self.rpn_target_maker = RPNTargetMaker()
        # fast rcnn target
        self.frcnn_target_maker = FRCNNTargetMaker()
        # fast rcnn head
        self.frcnn_head = FRCNNHead(num_classes=num_classes, roi_size=7, classifier=self.classifier)
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, bbox, label):

        # 1. extract features
        features = self.backbone(x)    # dict

        # 2. forward rpn
        pred_rpn_cls, pred_rpn_reg, rois, anchor = self.rpn(x, features, 'train')
        # print("rois", rois.shape)

        # 3. make target for rpn
        target_rpn_cls, target_rpn_reg = self.rpn_target_maker(bbox=bbox,
                                                               anchor=anchor)

        # 4. make target for fast rcnn
        target_fast_rcnn_cls, target_fast_rcnn_reg, sample_rois = self.frcnn_target_maker(bbox=bbox,
                                                                                          label=label,
                                                                                          rois=rois)
        # print(sample_rois.shape)
        # 5. forward fast rcnn head
        pred_fast_rcnn_cls, pred_fast_rcnn_reg = self.frcnn_head(features, sample_rois, x.shape[2:])
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(512, -1, 4)
        pred_fast_rcnn_reg = pred_fast_rcnn_reg[torch.arange(0, 512).long(), target_fast_rcnn_cls.long()]

        pred_rpn_cls = pred_rpn_cls.unsqueeze(0)
        pred_rpn_reg = pred_rpn_reg.unsqueeze(0)

        return (pred_rpn_cls, pred_rpn_reg, pred_fast_rcnn_cls, pred_fast_rcnn_reg), \
               (target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg)

    def predict(self, x, opts):

        # 1. extract features
        features = self.backbone(x)  # dict

        # 3. forward rpn
        pred_rpn_cls, pred_rpn_reg, rois, anchor = self.rpn(x, features, 'test')

        # 5. forward fast_rcnn_head
        pred_fast_rcnn_cls, pred_fast_rcnn_reg = self.frcnn_head(features, rois, x.shape[2:])

        # make pred prob and bbox(post process)
        pred_cls = (torch.softmax(pred_fast_rcnn_cls, dim=-1))      # batch 없애는 부분
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(-1, self.num_classes, 4)  # ex) [184, 21, 4]
        # un-normalization
        pred_fast_rcnn_reg = pred_fast_rcnn_reg * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(torch.get_device(pred_fast_rcnn_reg))
        rois = rois.reshape(-1, 1, 4).expand_as(pred_fast_rcnn_reg)
        pred_bbox = decode(pred_fast_rcnn_reg.reshape(-1, 4), xy_to_cxcy(rois.reshape(-1, 4)))
        pred_bbox = cxcy_to_xy(pred_bbox)

        pred_bbox = pred_bbox.reshape(-1, self.num_classes * 4)
        pred_bbox = pred_bbox.clamp(min=0, max=1)
        bbox, label, score = self._suppress(pred_bbox, pred_cls, opts)
        return bbox, label, score

    def _suppress(self, raw_cls_bbox, raw_prob, opts):
        bbox = list()
        label = list()
        score = list()

        # skip cls_id = 0 because it is the background class
        for l in range(1, self.num_classes):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.num_classes, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > opts.thres
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, iou_threshold=0.3)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


def normal_init(m, mean, stddev):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()


############ backbone ################

from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from typing import Callable, Dict, List, Optional, Union
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
# from .._utils import handle_legacy_interface, IntermediateLayerGetter


# Old weights:
# model = resnet50(weights=ResNet50_Weights.ImageNet1K_V1)

if __name__ == '__main__':

    boxes_tensor = [torch.FloatTensor([[79.8867, 286.8000, 329.7450, 444.0000],
                                       [11.8980, 13.2000, 596.6006, 596.4000]])]
    boxes_tensor_scale_1 = [(box_tensor/800).cuda() for box_tensor in boxes_tensor]
    label_tensor = [torch.Tensor([11, 14])]
    label_tensor = [label.cuda() for label in label_tensor]
    bbox = boxes_tensor_scale_1
    label = label_tensor

    # train
    img = torch.randn([1, 3, 800, 800]).cuda()
    model = FRCNN(num_classes=91).cuda()
    outputs = model(img, bbox, label)

    # test
    with torch.no_grad():
        import argparse
        img = torch.randn([1, 3, 800, 800]).cuda()
        parser = argparse.ArgumentParser()
        parser.add_argument('--thres', type=float, default=0.7)
        opts = parser.parse_args()
        model = FRCNN(num_classes=91).cuda()
        outputs = model.predict(img, opts)

        bbox, label, score = outputs
        print(bbox.shape)
        print(label)
        print(score)

    # for output in outputs:
    #     for out in output:
    #         print(out.size())

    # from torch import nn, Tensor
    # from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    # from torchvision.models import ResNet50_Weights
    #
    # backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.IMAGENET1K_V1, trainable_layers=3)
    # img = torch.randn([1, 3, 800, 800])
    #
    # model = FRCNN(num_classes=91)
    # print(model.predict(img, opts=None).shape)
    # # print(model.classifier)