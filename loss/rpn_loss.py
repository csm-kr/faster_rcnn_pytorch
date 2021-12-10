import torch
import torch.nn as nn


class RPNLoss(torch.nn.Module):
    def __init__(self, coder):
        super().__init__()
        self.coder = coder
        self.num_classes = self.coder.num_classes
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, boxes, labels, size):
        pred_cls, pred_reg = pred

        batch_size = pred_cls.size(0)
        pred_cls = pred_cls.permute(0, 2, 3, 1)  # [B, C, H, W] to [B, H, W, C]
        pred_reg = pred_reg.permute(0, 2, 3, 1)  # [B, C, H, W] to [B, H, W, C]
        pred_cls = pred_cls.reshape(batch_size, -1, 2)
        pred_reg = pred_reg.reshape(batch_size, -1, 4)

        # Q ) view 가 되는 자연스러운

        print(pred_cls.size())
        print(pred_reg.size())
        self.coder.set_anchors(size=size)
        gt_t_cls, gt_t_reg, anchor_identifier = self.coder.build_target(boxes, labels)

        cls_mask = (anchor_identifier >= 0).unsqueeze(-1).expand_as(gt_t_cls)       # [B, num_anchors, 2]
        N_cls = (anchor_identifier >= 0).sum()
        cls_loss = self.bce(torch.sigmoid(pred_cls), gt_t_cls)                      # [B, num_anchors, 2]
        cls_loss = (cls_loss * cls_mask).sum() / N_cls

        reg_mask = (anchor_identifier == 1).unsqueeze(-1).expand_as(gt_t_reg)       # [B, num_anchors, 4]
        N_reg = gt_t_reg.size(1) // 9     # number of anchor location - H' x W' == num_anchors // 9
        lambda_ = 10

        # gt_t_classes      - [B, num_anchors, 2]
        # gt_t_boxes        - [B, num_anchors, 4]
        # anchor_identifier - [B, num_anchors   ] each value \in {-1, 0, 1} which -1 ignore, 0 negative, 1 positive

        print(anchor_identifier.size())
        print(gt_t_cls.size())
        print(pred_cls.size())


        return 0


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
