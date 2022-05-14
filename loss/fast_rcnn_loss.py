import torch
import torch.nn as nn
from utils import find_jaccard_overlap, encode, xy_to_cxcy


class FastRCNNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def build_fast_rcnn_target(self, bbox, label, roi):
        # 1. concatenate bbox and roi -

        # remove the list for batch
        bbox = bbox[0]
        label = label[0]

        # 무조건 나오는 roi를 만들기 위해서 와 bbox를 concat 한다.
        roi = torch.cat([roi, bbox], dim=0)
        iou = find_jaccard_overlap(roi, bbox)  # [2000 + num_obj, num objects]
        IoU_max, IoU_argmax = iou.max(dim=1)

        # set background label 0
        fast_rcnn_tg_cls = label[IoU_argmax] + 1

        # n_pos = 32 or IoU 0.5 이상
        n_pos = int(min((IoU_max > 0.5).sum(), 32))

        # random select pos and neg indices
        pos_indices = torch.arange(IoU_max.size(0))[IoU_max >= 0.5]
        perm = torch.randperm(pos_indices.size(0))
        pos_indices = pos_indices[perm[:n_pos]]

        n_neg = 128 - n_pos
        neg_indices = torch.arange(IoU_max.size(0))[(IoU_max < 0.5) & (IoU_max >= 0.0)]
        perm = torch.randperm(neg_indices.size(0))
        neg_indices = neg_indices[perm[:n_neg]]
        assert n_neg + n_pos == 128

        # keep indices
        keep_indices = torch.cat([pos_indices, neg_indices], dim=-1)

        # make fast rcnn cls
        fast_rcnn_tg_cls = fast_rcnn_tg_cls[keep_indices]

        # set negative indices background label
        fast_rcnn_tg_cls[n_pos:] = 0

        # make roi
        sample_roi = roi[keep_indices, :]

        # make fast rcnn loc
        fast_rcnn_tg_loc = encode(xy_to_cxcy(bbox[keep_indices]), xy_to_cxcy(sample_roi))
        return fast_rcnn_tg_cls, fast_rcnn_tg_loc, sample_roi

    def forward(self, x):
        return x


if __name__ == '__main__':
    import time
    from PIL import Image
    from model.faster_rcnn import FRCNN
    import torchvision.transforms as tfs

    # 1. load image
    image = Image.open('../figures/000001.jpg').convert('RGB')
    # 2. transform image
    transforms = tfs.Compose([tfs.Resize((600, 600)),
                              tfs.ToTensor(),
                              tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])

    # 3. set image and boxes tensor
    image_tensor = transforms(image).unsqueeze(0)
    boxes_tensor = [torch.FloatTensor([[79.8867, 286.8000, 329.7450, 444.0000],
                                       [11.8980, 13.2000, 596.6006, 596.4000]])]

    boxes_tensor_scale_1 = [(box_tensor/600).cuda() for box_tensor in boxes_tensor]

    label_tensor = [torch.Tensor([11, 14])]
    label_tensor = [label.cuda() for label in label_tensor]

    # boxes_tensor_scale_1 = [b.cuda() for b in boxes_tensor_scale_1]
    img1 = image_tensor.cuda()

    tic = time.time()
    frcnn = FRCNN().cuda()
    rpn_cls, rpn_reg, frcnn_cls, frcnn_reg, anchor, rois = frcnn(img1)

    print(rpn_cls.size())     # torch.Size([1, 18, 37, 62])
    print(rpn_reg.size())     # torch.Size([1, 36, 37, 62])
    print(frcnn_cls.size())   # torch.Size([1, 1988, 21])
    print(frcnn_reg.size())   # torch.Size([1, 1988, 4])
    print(anchor.size())      # torch.Size([20646, 4])

    from loss.rpn_loss import RPNLoss
    from loss.fast_rcnn_loss import FastRCNNLoss

    rpn_loss = RPNLoss()
    rpn_tg_cls, rpn_tg_loc = rpn_loss.build_rpn_target(bbox=boxes_tensor_scale_1, anchor=anchor)
    print(rpn_tg_cls.size())
    print(rpn_tg_loc.size())

    fast_rcnn_loss = FastRCNNLoss()
    fast_rcnn_tg_cls, fast_rcnn_tg_loc, roi = fast_rcnn_loss.build_fast_rcnn_target(bbox=boxes_tensor_scale_1, label=label_tensor, roi=rois)
    print(fast_rcnn_tg_cls.size())
    print(fast_rcnn_tg_loc.size())
    print(roi.size())




