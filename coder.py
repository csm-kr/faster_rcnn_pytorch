import numpy as np
import torch
from math import sqrt
from abc import ABCMeta, abstractmethod
# from util.utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap
from utils import find_jaccard_overlap, xy_to_cxcy, xy_to_cxcy2
from anchor import FasterRCNN_Anchor
from utils import cxcy_to_xy


class Coder(metaclass=ABCMeta):

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass


class FasterRCNN_Coder(Coder):

    """
    class of target encode/decode
    when gt box center-coordinates (x, y, w, h) and anchor box center-coordinates (x_a, y_a, w_a, h_a),
    encode:
    t_x = (x - x_a) / w_a
    t_x = (y - y_a) / h_a
    t_w = log(w / w_a)
    t_h = log(h / h_a)
    decode:
    x = t_x * w_a + x_a
    y = t_y * h_a + y_a
    w = e^(t_w) * w_a
    h = e^(t_h) * w_h
    """

    def __init__(self):
        super().__init__()
        self.data_type = 'voc'
        self.anchor_obj = FasterRCNN_Anchor()
        self.num_classes = 20
        self.anchor_dic = {}

    def set_anchors(self, size):
        if self.anchor_dic.get(size) is None:
            self.anchor_dic[size] = self.anchor_obj.create_anchors(image_size=size)
        self.center_anchor, self.keep = self.anchor_dic[size]

    # def assign_anchors_to_device(self):
    #     self.center_anchor = self.center_anchor.to(device)
    #
    # def assign_anchors_to_cpu(self):
    #     self.center_anchor = self.center_anchor.to('cpu')

    def encode(self, cxcy):
        """
        for loss, gt(cxcy) to target tcxcy with anchors
        """
        tcxcy = (cxcy[:, :2] - self.center_anchor[:, :2]) / self.center_anchor[:, 2:]       # (box cxy-anc cxy)/anc wh
        twh = torch.log(cxcy[:, 2:] / self.center_anchor[:, 2:])                            # log(box wh / anc wh)
        return torch.cat([tcxcy, twh], dim=1)

    def decode(self, tcxcy):
        """
        for test and demo, tcxcy to gt
        """
        cxcy = tcxcy[:, :2] * self.center_anchor[:, 2:] + self.center_anchor[:, :2]
        wh = torch.exp(tcxcy[:, 2:]) * self.center_anchor[:, 2:]
        return torch.cat([cxcy, wh], dim=1)

    def sample_anchors(self, anchor_identifier, num_samples=256):

        positive_indices_bool = anchor_identifier == 1
        negative_indices_bool = anchor_identifier == 0
        num_positive_anchors = positive_indices_bool.sum()
        num_negative_anchors = negative_indices_bool.sum()

        if num_positive_anchors < num_samples//2:   # if pos anchors are smaller than 128,
            # zero of anchor_identifier convert to -1 except 256 - num_pos_anchors
            num_neg_sample = num_samples - num_positive_anchors

            # sample 256 - num_pos_anchors from negative anchors
            negative_indices = torch.arange(anchor_identifier.size(0))[negative_indices_bool]
            perm = torch.randperm(negative_indices.size(0))
            anchor_identifier[negative_indices[perm[num_neg_sample:]]] = -1
            # sanity check
            assert (anchor_identifier == 0).sum() == num_neg_sample

        else:
            # if pos anchors are larger than 128, sample 128 and background 128 patches.

            # pos sample 을 128 로 맞춤
            num_pos_sample = num_samples//2
            num_neg_sample = num_samples//2

            positive_indices = torch.arange(anchor_identifier.size(0))[positive_indices_bool]
            perm_pos = torch.randperm(positive_indices.size(0))
            anchor_identifier[positive_indices[perm_pos[num_pos_sample:]]] = -1

            negative_indices = torch.arange(anchor_identifier.size(0))[negative_indices_bool]
            perm_neg = torch.randperm(negative_indices.size(0))
            anchor_identifier[negative_indices[perm_neg[num_neg_sample:]]] = -1

            # sanity check
            assert (anchor_identifier == 0).sum() == (anchor_identifier == 1).sum()

        return anchor_identifier

    def build_target(self, gt_boxes, gt_labels):
        """
        gt_boxes : [B, ]
        gt_labels : [B, ]
        """

        batch_size = len(gt_labels)
        num_anchors = self.center_anchor.size(0)
        device_ = gt_labels[0].get_device()
        self.center_anchor = self.center_anchor.to(device_)
        self.keep = self.keep.to(device_)

        # ----- 1. make container
        # gt_boxes  - [B, anchors, 4]
        gt_locations = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float, device=device_)

        # foreground vs background - [B, anchors, 2]
        gt_classes = torch.zeros((batch_size, num_anchors, 2), dtype=torch.float, device=device_)
        # gt_classes = -1 * torch.ones((batch_size, n_priors, self.num_classes), dtype=torch.float, device=device_)

        anchor_identifier = -1 * torch.ones((batch_size, num_anchors), dtype=torch.float32, device=device_)

        # if anchor is positive -> 1,
        #              negative -> 0,
        #              ignore   -> -1

        # ----- 2. make corner anchors
        self.center_anchor = self.center_anchor.to(device_)
        corner_anchor = cxcy_to_xy(self.center_anchor)

        for i in range(batch_size):
            boxes = gt_boxes[i]     # xy coord
            labels = gt_labels[i]

            # ----- 3. find iou between anchors and boxes
            iou = find_jaccard_overlap(corner_anchor, boxes)
            IoU_max, IoU_argmax = iou.max(dim=1)

            # ----- 4. build gt_classes
            # [1] third condition - negative anchors
            negative_indices = IoU_max < 0.3
            gt_classes[i][negative_indices, 0] = 1
            anchor_identifier[i][negative_indices] = 0

            # [2] second condition - positive anchors (iou > 0.7)
            positive_indices = IoU_max >= 0.7

            # [3] second condition - (maximum iou)
            _, IoU_argmax_per_object = iou.max(dim=0)
            positive_indices[IoU_argmax_per_object] = 1
            positive_indices = positive_indices.type(torch.bool)

            # assigning label
            # argmax_labels = labels[IoU_argmax]
            gt_classes[i][positive_indices, 1] = 1
            # gt_classes[i][positive_indices, argmax_labels[positive_indices].long()] = 1. # objects

            anchor_identifier[i][positive_indices] = 1                                     # original masking \in {0, 1}

            # ----- 4. build gt_locations
            argmax_locations = boxes[IoU_argmax]
            center_locations = xy_to_cxcy(argmax_locations)  # [67995, 4] 0 ~ 1 사이이다. boxes 가
            gt_tcxcywh = self.encode(center_locations)
            gt_locations[i] = gt_tcxcywh

            negative_ones = -1 * torch.ones((batch_size, num_anchors), dtype=torch.float32, device=device_)
            # remove border-sides anchors. keep : cross-boundary anchors

            if self.keep.sum() < 256:
                keep_indices = torch.arange(self.keep.size(0))[self.keep == 0]
                perm = torch.randperm(keep_indices.size(0))
                self.keep[keep_indices[perm[:256]]] = True

            self.keep = torch.logical_or(self.keep, anchor_identifier[i] == 1)

            anchor_identifier[i] = torch.where(self.keep, anchor_identifier[i], negative_ones[i])
            # sample 256 anchors which ratio is 1:1
            anchor_identifier[i] = self.sample_anchors(anchor_identifier[i])

        return gt_classes, gt_locations, anchor_identifier

    def post_processing(self, pred, is_demo=False):

        if is_demo:
            self.assign_anchors_to_cpu()
            pred_cls = pred[0].to('cpu')
            pred_loc = pred[1].to('cpu')
        else:
            pred_cls = pred[0]
            pred_loc = pred[1]

        n_priors = self.center_anchor.size(0)
        assert n_priors == pred_loc.size(1) == pred_cls.size(1)

        # decode 에서 나온 bbox 는 center coord
        pred_bboxes = cxcy_to_xy(self.decode(pred_loc.squeeze())).clamp(0, 1)        # for batch 1, [67995, 4]
        pred_scores = pred_cls.squeeze()                                             # for batch 1, [67995, num_classes]

        # corner coordinates 를 x1y1x2y2 를 0 ~ 1 로 scaling 해줌
        # 0.3109697496017331 -> 0.3115717185294685 로 오름

        return pred_bboxes, pred_scores


if __name__ == '__main__':
    import dataset.detection_transforms as det_transforms
    from dataset.voc_dataset import VOC_Dataset
    from torch.utils.data import Dataset, DataLoader

    # train_transform
    window_root = 'D:\data\\voc'
    root = window_root

    transform_train = det_transforms.DetCompose([
        # ------------- for Tensor augmentation -------------
        det_transforms.DetRandomPhotoDistortion(),
        det_transforms.DetRandomHorizontalFlip(),
        det_transforms.DetToTensor(),
        # ------------- for Tensor augmentation -------------
        det_transforms.DetRandomZoomOut(max_scale=3),
        det_transforms.DetRandomZoomIn(),
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

    coder = FasterRCNN_Coder()

    for images, boxes, labels in train_loader:
        images = images.to('cuda')
        boxes = [b.to('cuda') for b in boxes]
        labels = [l.to('cuda') for l in labels]

        height, width = images.size()[2:]  # height, width
        size = (height, width)
        coder.set_anchors(size=size)
        gt_t_classes, gt_t_boxes, anchor_identifier = coder.build_target(boxes, labels)


    # image, box, label = train_set.__getitem__(10)
    # height, width = image.size()[1:]     # height, width
    # size = (height, width)
    # coder = FasterRCNN_Coder()
    # coder.set_anchors(size=size)
    # coder.encode(box)
