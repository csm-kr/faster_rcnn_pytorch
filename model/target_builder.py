import torch
import torch.nn as nn
from utils import find_jaccard_overlap, xy_to_cxcy, encode


class FastRCNNTargetBuilder(nn.Module):
    def __init__(self):
        super().__init__()

    def build_fast_rcnn_target(self, bbox, label, rois):
        # 1. concatenate bbox and roi -
        # remove the list for batch
        bbox = bbox[0]
        label = label[0]

        # 무조건 나오는 roi를 만들기 위해서 와 bbox를 concat 한다.
        rois = torch.cat([rois, bbox], dim=0)
        iou = find_jaccard_overlap(rois, bbox)  # [2000 + num_obj, num objects]
        IoU_max, IoU_argmax = iou.max(dim=1)

        # set background label 0
        fast_rcnn_tg_cls = label[IoU_argmax] + 1

        # n_pos = 32 or IoU 0.5 이상
        n_pos = int(min((IoU_max >= 0.5).sum(), 32))

        # random select pos and neg indices
        # pos_indices = torch.arange(IoU_max.size(0))[IoU_max >= 0.5]
        # perm = torch.randperm(pos_indices.size(0))
        # pos_indices = pos_indices[perm[:n_pos]]
        import numpy as np
        # np.random.seed(111)
        pos_index = torch.arange(IoU_max.size(0))[IoU_max >= 0.5].cpu().numpy()
        if pos_index.size > 0:
            np.random.seed(111)
            pos_index = np.random.choice(pos_index, size=n_pos, replace=False)

        n_neg = 128 - n_pos
        # neg_indices = torch.arange(IoU_max.size(0))[(IoU_max < 0.5) & (IoU_max >= 0.0)]
        # perm = torch.randperm(neg_indices.size(0))
        # neg_indices = neg_indices[perm[:n_neg]]

        neg_index = torch.arange(IoU_max.size(0))[(IoU_max < 0.5) & (IoU_max >= 0.0)].cpu().numpy()
        # n_remnant_length = int(min(128 - n_pos, neg_index.size))
        if neg_index.size > 0:
            # print(neg_index.size)
            np.random.seed(111)
            neg_index = np.random.choice(neg_index, size=128 - n_pos, replace=False)

        assert n_neg + n_pos == 128

        # # keep indices
        # keep_indices = torch.cat([pos_indices, neg_indices], dim=-1)
        # fast_rcnn_tg_cls = fast_rcnn_tg_cls[keep_indices]
        # # set negative indices background label
        # fast_rcnn_tg_cls[n_pos:] = 0
        # fast_rcnn_tg_cls = fast_rcnn_tg_cls.type(torch.long)
        #
        # # make roi
        # sample_rois = rois[keep_indices, :]
        # # make fast rcnn reg
        # fast_rcnn_tg_reg = encode(xy_to_cxcy(bbox[IoU_argmax][keep_indices]), xy_to_cxcy(sample_rois))

        keep_index = np.concatenate([pos_index, neg_index], axis=-1)

        # make CLS target
        fast_rcnn_tg_cls = fast_rcnn_tg_cls[keep_index]
        # set negative indices background label
        fast_rcnn_tg_cls[n_pos:] = 0
        fast_rcnn_tg_cls = fast_rcnn_tg_cls.type(torch.long)

        # make roi
        sample_rois = rois[keep_index, :]
        # make REG target
        fast_rcnn_tg_reg = encode(xy_to_cxcy(bbox[IoU_argmax][keep_index]), xy_to_cxcy(sample_rois))

        return fast_rcnn_tg_cls, fast_rcnn_tg_reg, sample_rois


class RPNTargetBuilder(nn.Module):
    def __init__(self):
        super().__init__()

    def build_rpn_target(self, bbox, anchor):
        '''
        bbox : list of tensor [tensor]
        '''
        # 1. anchor cross boundary 만 걸러내기
        bbox = bbox[0]  # remove the list for batch : shape [num_obj, 4]
        anchor_keep = ((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) & (anchor[:, 2] <= 1) & (anchor[:, 3] <= 1))
        anchor = anchor[anchor_keep]
        num_anchors = anchor.size(0)

        # 2. iou 따라 label 만들기
        # if label is 1 (positive), 0 (negative), -1 (ignore)
        label = -1 * torch.ones(num_anchors, dtype=torch.float32, device=bbox.get_device())

        iou = find_jaccard_overlap(anchor, bbox)  # [num anchors, num objects]
        IoU_max, IoU_argmax = iou.max(dim=1)

        # 2-1. set negative label
        label[IoU_max < 0.3] = 0
        # 2-2. set positive label that have highest iou.
        IoU_max_per_object, IoU_argmax_per_object = iou.max(dim=0)
        # ** max 값이 여러개 있다면(동일하게), 그것을 가져오는 부분. **
        # IoU_argmax_per_object update (max 값 포함하는 index 찾기)
        IoU_argmax_per_object = torch.nonzero(input=(iou == IoU_max_per_object))[:, 0]  # 2차원이라 앞의 column 가져오기
        label[IoU_argmax_per_object] = 1
        # 2-3. set positive label
        label[IoU_max >= 0.7] = 1
        # 2-4 sample target
        n_pos = (label == 1).sum()
        n_neg = (label == 0).sum()

        if n_pos > 128:
            # pos_indices = torch.arange(label.size(0))[label == 1]
            # perm = torch.randperm(pos_indices.size(0))
            # label[pos_indices[perm[128:]]] = -1  # convert pos label to ignore label

            import numpy as np
            np.random.seed(111)
            pos_index = torch.arange(label.size(0))[label == 0].numpy()
            disable_index = np.random.choice(pos_index, size=int(n_pos - 128), replace=False)
            label[disable_index] = -1

        if n_neg > 256 - n_pos:

            # neg_indices = torch.arange(label.size(0))[label == 0]
            # perm = torch.randperm(neg_indices.size(0))
            # label[neg_indices[perm[(256 - n_pos):]]] = -1  # convert neg label to ignore label

            import numpy as np
            np.random.seed(111)
            neg_index = torch.arange(label.size(0))[label == 0].numpy()
            disable_index = np.random.choice(neg_index, size=int(len(neg_index) - (256 - n_pos)), replace=False)
            label[disable_index] = -1  # tensor 의 index 로 numpy 가 된다??

            # perm = torch.randperm(neg_indices.size(0))
            # label[neg_indices[perm[(256 - n_pos):]]] = -1  # convert neg label to ignore label

        assert (label == 1).sum() + (label == 0).sum() == 256, \
            '더해서 256이 아니라고? pos : {} vs neg : {}'.format((label == 1).sum(), (label == 0).sum())

        # 3. bbox encoding
        tg_cxywh = encode(xy_to_cxcy(bbox[IoU_argmax]), xy_to_cxcy(anchor))

        # 4. pad label and bbox for ignore label
        pad_label = -1 * torch.ones(len(anchor_keep), dtype=torch.float32, device=bbox.get_device())
        keep_indices = torch.arange(len(anchor_keep))[anchor_keep]
        pad_label[keep_indices] = label
        rpn_tg_cls = pad_label.type(torch.long)

        pad_bbox = torch.zeros([len(anchor_keep), 4], dtype=torch.float32, device=bbox.get_device())
        pad_bbox[keep_indices] = tg_cxywh
        rpn_tg_reg = pad_bbox

        # The size of rpn_tg_cls / rpn_tg_reg : [16650] / [16650, 4]
        return rpn_tg_cls, rpn_tg_reg
