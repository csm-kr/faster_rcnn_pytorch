import torch
import torch.nn as nn
from utils import decode, xy_to_cxcy, cxcy_to_xy

# from torchvision.ops.boxes import box_area


# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)
#
#
# def box_xyxy_to_cxcywh(x):
#     x0, y0, x1, y1 = x.unbind(-1)
#     b = [(x0 + x1) / 2, (y0 + y1) / 2,
#          (x1 - x0), (y1 - y0)]
#     return torch.stack(b, dim=-1)
#
#
# # modified from torchvision to also return the union
# def box_iou(boxes1, boxes2):
#     area1 = box_area(boxes1)
#     area2 = box_area(boxes2)
#
#     lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#     rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
#
#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
#
#     union = area1[:, None] + area2 - inter
#
#     iou = inter / union
#     return iou, union
#
#
# def generalized_box_iou(boxes1, boxes2):
#     """
#     Generalized IoU from https://giou.stanford.edu/
#     The boxes should be in [x0, y0, x1, y1] format
#     Returns a [N, M] pairwise matrix, where N = len(boxes1)
#     and M = len(boxes2)
#     """
#     # degenerate boxes gives inf / nan results
#     # so do an early check
#     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
#     assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
#     iou, union = box_iou(boxes1, boxes2)
#
#     lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
#     rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
#
#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     area = wh[:, :, 0] * wh[:, :, 1]
#
#     return iou - (area - union) / area


class GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, anchor, mask):
        """
        pred : [1, 12996, 4]
        """

        center_anchor = xy_to_cxcy(anchor)
        boxes1 = cxcy_to_xy(decode(pred.squeeze(0), center_anchor))
        boxes2 = cxcy_to_xy(decode(target, center_anchor))

        # giou_loss = generalized_box_iou(boxes1, boxes2)

        # iou loss
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [2, s, s, 3]
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [2, s, s, 3]

        inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])                          # [B, s, s, 3, 2]
        inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])                       # [B, s, s, 3, 2]

        inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))  # [B, s, s, 3, 2]
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area                                  # [B, s, s, 3]
        ious = 1.0 * inter_area / union_area                                                 # [B, s, s, 3]

        # iou_loss = 1 - ious
        # return iou_loss

        outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])                          # [B, s, s, 3, 2]
        outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])                       # [B, s, s, 3, 2]
        outer_section = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
        outer_area = outer_section[..., 0] * outer_section[..., 1]                           # [B, s, s, 3]

        giou = ious - (outer_area - union_area) / (outer_area)
        giou_loss = 1 - giou

        giou_loss = giou_loss[mask[0]].sum()
        return giou_loss


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
    def __init__(self, loss_type=None):
        super().__init__()
        self.loss_type = loss_type
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.loss_type == 'giou':
            self.giou_loss = GIoULoss()
        else:
            self.smooth_l1_loss = SmoothL1Loss(beta=1/9)
        self.rpn_lambda = 10

    def forward(self, pred_cls, pred_reg, target_cls, target_reg, anchor=None):

        # pred_cls : [1, 12321, 2] - batch 없애야함
        # pred_reg : [1, 12321, 4]

        # target_cls : [12321] - must be torch.long
        # target_reg : [12321, 4]

        rpn_cls_loss = self.cross_entropy_loss(pred_cls.squeeze(0), target_cls)
        rpn_reg_loss = None
        if self.loss_type == 'giou':
            rpn_reg_loss = self.giou_loss(pred_reg, target_reg, anchor, [target_cls > 0])
        else:
            rpn_reg_loss = self.smooth_l1_loss(pred_reg.squeeze(0)[target_cls > 0], target_reg[target_cls > 0])

        # # FIXME : follow the paper
        # N_reg = target_cls.size(0) // 9
        # rpn_reg_loss = self.rpn_lambda * (rpn_reg_loss.sum() / N_reg)
        rpn_reg_loss = rpn_reg_loss.sum() / (target_cls >= 0).sum()

        return rpn_cls_loss, rpn_reg_loss


class FastRCNNLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        if self.loss_type == 'giou':
            self.giou_loss = GIoULoss()
        else:
            self.smooth_l1_loss = SmoothL1Loss(beta=1)
        # self.smooth_l1_loss = SmoothL1Loss(1)

    def forward(self, pred_cls, pred_reg, target_cls, target_reg, sample_roi=None):
        # pred_cls : [1, 128, 21]
        # pred_reg : [1, 128, 4]
        # target_cls : [128] - must be torch.long
        # target_reg : [128, 4]
        fast_rcnn_cls_loss = self.cross_entropy_loss(pred_cls.squeeze(0), target_cls)
        if self.loss_type == 'giou':
            fast_rcnn_reg_loss = self.giou_loss(pred_reg, target_reg, sample_roi, [target_cls > 0])
        else:
            fast_rcnn_reg_loss = self.smooth_l1_loss(pred_reg.squeeze(0)[target_cls > 0], target_reg[target_cls > 0])

        # FIXME
        fast_rcnn_reg_loss = fast_rcnn_reg_loss.sum() / (target_cls >= 0).sum()

        return fast_rcnn_cls_loss, fast_rcnn_reg_loss


class FRCNNLoss(torch.nn.Module):
    def __init__(self, loss_type=None):
        super().__init__()
        self.loss_type = loss_type
        self.rpn_loss = RPNLoss(loss_type)
        self.fast_rcnn_loss = FastRCNNLoss(loss_type)

    def forward(self, pred, target, anchor=None, sample_roi=None):

        pred_rpn_cls, pred_rpn_reg, pred_fast_rcnn_cls, pred_fast_rcnn_reg = pred
        target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg = target

        # bbox normalization
        # target_fast_rcnn_reg /= torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(torch.get_device(target_fast_rcnn_reg))
        # target_rpn_reg *= torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(torch.get_device(target_fast_rcnn_reg))

        rpn_cls_loss, rpn_reg_loss = self.rpn_loss(pred_rpn_cls, pred_rpn_reg,
                                                   target_rpn_cls, target_rpn_reg,
                                                   anchor)
        fast_rcnn_cls_loss, fast_rcnn_reg_loss = self.fast_rcnn_loss(pred_fast_rcnn_cls, pred_fast_rcnn_reg,
                                                                     target_fast_rcnn_cls, target_fast_rcnn_reg,
                                                                     sample_roi)

        total_loss = rpn_cls_loss + rpn_reg_loss + fast_rcnn_cls_loss + fast_rcnn_reg_loss
        return total_loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss


if __name__ == '__main__':
    import time
    from PIL import Image
    import torchvision.transforms as tfs
    from model import FRCNN

    # 1. load image
    image = Image.open('./figures/000001.jpg').convert('RGB')
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
    img = image_tensor.cuda()
    bbox = boxes_tensor_scale_1
    label = label_tensor

    # label =
    tic = time.time()
    from model_dc5 import FRCNN

    loss_type = 'giou'

    frcnn = FRCNN(num_classes=21, model_type='vgg_origin').cuda()
    frcnn = FRCNN(num_classes=21, model_type='resnet_dc5', loss_type=loss_type).cuda()
    # frcnn = FRCNN(num_classes=21).cuda()
    pred, target, anchor, sample_roi = frcnn(img, bbox, label)

    pred_rpn_cls, pred_rpn_reg, pred_fast_cls, pred_fast_rcnn_reg = pred
    target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg = target

    print(pred_rpn_cls.size())     # torch.Size([1, 18, 37, 62])
    print(pred_rpn_reg.size())     # torch.Size([1, 18, 37, 62])
    print(pred_fast_cls.size())     # torch.Size([1, 18, 37, 62])
    print(pred_fast_rcnn_reg.size())     # torch.Size([1, 18, 37, 62])

    print((target_rpn_cls >= 0).sum())     # torch.Size([1, 18, 37, 62])
    # print(target_rpn_cls[target_rpn_cls >= 0].size())     # torch.Size([1, 18, 37, 62])
    # print(target_rpn_reg[target_rpn_cls >= 0].size())     # torch.Size([1, 36, 37, 62])
    print(target_rpn_cls.size())     # torch.Size([1, 18, 37, 62])
    print(target_rpn_reg.size())     # torch.Size([1, 36, 37, 62])
    print(target_fast_rcnn_cls.size())   # torch.Size([1, 1988, 21])
    print(target_fast_rcnn_reg.size())   # torch.Size([1, 1988, 4])

    # if anchor is not None:
    #     criterion = FRCNNLoss(loss_type='giou')
    #     loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss = criterion(pred, target, anchor, sampled_roi)
    # else:
    criterion = FRCNNLoss(loss_type=loss_type)
    loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss = criterion(pred, target, anchor, sample_roi)
    print("total_loss :", loss)
    print("rpn_cls_loss :", rpn_cls_loss)
    print("rpn_reg_loss :", rpn_reg_loss)
    print("fast_rcnn_cls_loss :", fast_rcnn_cls_loss)
    print("fast_rcnn_reg_loss :", fast_rcnn_reg_loss)

