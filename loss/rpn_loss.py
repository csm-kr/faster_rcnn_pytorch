import torch
import torch.nn as nn
from utils import find_jaccard_overlap, encode, xy_to_cxcy


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class RPNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.coder = coder
        # self.num_classes = self.coder.num_classes
        self.bce = nn.BCELoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        # self.smooth_l1_loss = SmoothL1Loss()

    def build_rpn_target(self, bbox, anchor):
        '''
        bbox : list of tensor [tensor]
        '''

        # 1. anchor cross boundary 만 걸러내기
        bbox = bbox[0]  # remove the list for batch : shape [num_obj, 4]
        anchor_keep = ((anchor[:, 0] >= 0) & (anchor[:, 1] >= 0) & (anchor[:, 2] < 1) & (anchor[:, 3] < 1))
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
        _, IoU_argmax_per_object = iou.max(dim=0)
        label[IoU_argmax_per_object] = 1
        # 2-3. set positive label
        label[IoU_max > 0.7] = 1
        # 2-4 sample target
        n_pos = (label == 1).sum()
        n_neg = (label == 0).sum()
        if n_pos > 128:
            pos_indices = torch.arange(label.size(0))[label == 1]
            perm = torch.randperm(pos_indices.size(0))
            label[pos_indices[perm[128 - n_pos:]]] = -1  # convert pos label to ignore label
        if n_neg > 128:
            if n_pos >= 128:
                neg_indices = torch.arange(label.size(0))[label == 0]
                perm = torch.randperm(neg_indices.size(0))
                label[neg_indices[perm[128 - n_neg:]]] = -1  # convert neg label to ignore label
            # fewer that 128 positive samples ,pad negative ones
            if n_pos < 128:
                neg_indices = torch.arange(label.size(0))[label == 0]
                perm = torch.randperm(neg_indices.size(0))
                label[neg_indices[perm[(256 - n_pos) - n_neg:]]] = -1  # convert neg label to ignore label
        assert (label == 1).sum() + (label == 0).sum() == 256

        # 3. bbox encoding
        tg_cxywh = encode(xy_to_cxcy(bbox[IoU_argmax]), xy_to_cxcy(anchor))

        rpn_tg_cls = label
        rpn_tg_loc = tg_cxywh
        return rpn_tg_cls, rpn_tg_loc

    def forward(self, pred, boxes, labels, anchors):
        pred_cls, pred_reg = pred

        batch_size = pred_cls.size(0)
        pred_cls = pred_cls.permute(0, 2, 3, 1)  # [B, C, H, W] to [B, H, W, C]
        pred_reg = pred_reg.permute(0, 2, 3, 1)  # [B, C, H, W] to [B, H, W, C]
        pred_cls = pred_cls.reshape(batch_size, -1, 2)
        pred_reg = pred_reg.reshape(batch_size, -1, 4)

        # build rpn target
        rpn_tg_cls, rpn_tg_loc = self.build_rpn_target(boxes, anchors)

        self.coder.set_anchors(size=size)
        gt_t_cls, gt_t_reg, anchor_identifier = self.coder.build_target(boxes, labels)

        cls_mask = (anchor_identifier >= 0).unsqueeze(-1).expand_as(gt_t_cls)       # [B, num_anchors, 2]
        N_cls = (anchor_identifier >= 0).sum()
        cls_loss = self.bce(torch.sigmoid(pred_cls), gt_t_cls)                      # [B, num_anchors, 2]
        cls_loss = (cls_loss * cls_mask).sum() / N_cls

        reg_mask = (anchor_identifier == 1).unsqueeze(-1).expand_as(gt_t_reg)       # [B, num_anchors, 4]
        N_reg = gt_t_reg.size(1) // 9   # number of anchor location - H' x W' == num_anchors // 9 - divided into (anchor scales times aspect ratio).
        lambda_ = 10
        # loc loss
        reg_loss = self.smooth_l1_loss(pred_reg, gt_t_reg)
        reg_loss = lambda_ * (reg_mask * reg_loss).sum() / N_reg
        total_loss = cls_loss + reg_loss

        # gt_t_classes      - [B, num_anchors, 2]
        # gt_t_boxes        - [B, num_anchors, 4]
        # anchor_identifier - [B, num_anchors   ] each value \in {-1, 0, 1} which -1 ignore, 0 negative, 1 positive
        return total_loss, cls_loss, reg_loss


if __name__ == '__main__':
    import dataset.detection_transforms as det_transforms
    from dataset.voc_dataset import VOC_Dataset
    from torch.utils.data import Dataset, DataLoader
    from coder import FasterRCNN_Coder
    from model.model import RPN

    # train_transform
    window_root = 'D:\data\\voc'
    root = window_root

    transform_train = det_transforms.DetCompose([
        # ------------- for Tensor augmentation -------------
        # det_transforms.DetRandomPhotoDistortion(),
        det_transforms.DetRandomHorizontalFlip(),
        det_transforms.DetToTensor(),
        # ------------- for Tensor augmentation -------------
        # det_transforms.DetRandomZoomOut(max_scale=3),
        # det_transforms.DetRandomZoomIn(),
        det_transforms.DetResize(size=600, max_size=1000, box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    transform_test = det_transforms.DetCompose([
        det_transforms.DetToTensor(),
        det_transforms.DetResize(size=600, max_size=1000, box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    train_set = VOC_Dataset(root,
                            split='train',
                            download=True,
                            transform=transform_train,
                            visualization=True)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=1,
                              shuffle=True,
                              collate_fn=train_set.collate_fn)

    model = RPN().to('cuda')
    coder = FasterRCNN_Coder()
    criterion = RPNLoss(coder)

    for images, boxes, labels in train_loader:
        images = images.to('cuda')
        boxes = [b.to('cuda') for b in boxes]
        labels = [l.to('cuda') for l in labels]

        height, width = images.size()[2:]  # height, width
        size = (height, width)
        pred = model(images)   # [cls, reg] - [B, 18, H', W'], [B, 36, H', W']
        loss = criterion(pred, boxes, labels, size)
        print(loss)