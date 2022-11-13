import torch
import torch.nn as nn


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
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.smooth_l1_loss = SmoothL1Loss(beta=1/9)
        self.rpn_lambda = 10

    def forward(self, pred_cls, pred_reg, target_cls, target_reg):

        # pred_cls : [1, 12321, 2] - batch 없애야함
        # pred_reg : [1, 12321, 4]

        # target_cls : [12321] - must be torch.long
        # target_reg : [12321, 4]

        rpn_cls_loss = self.cross_entropy_loss(pred_cls.squeeze(0), target_cls)
        rpn_reg_loss = self.smooth_l1_loss(pred_reg.squeeze(0)[target_cls > 0], target_reg[target_cls > 0])

        # # FIXME : follow the paper
        # N_reg = target_cls.size(0) // 9
        # rpn_reg_loss = self.rpn_lambda * (rpn_reg_loss.sum() / N_reg)
        rpn_reg_loss = rpn_reg_loss.sum() / (target_cls >= 0).sum()

        return rpn_cls_loss, rpn_reg_loss


class FastRCNNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.smooth_l1_loss = SmoothL1Loss(1)

    def forward(self, pred_cls, pred_reg, target_cls, target_reg):
        # pred_cls : [1, 128, 21]
        # pred_reg : [1, 128, 4]
        # target_cls : [128] - must be torch.long
        # target_reg : [128, 4]

        fast_rcnn_cls_loss = self.cross_entropy_loss(pred_cls.squeeze(0), target_cls)
        fast_rcnn_reg_loss = self.smooth_l1_loss(pred_reg.squeeze(0)[target_cls > 0], target_reg[target_cls > 0])

        # FIXME
        fast_rcnn_reg_loss = fast_rcnn_reg_loss.sum() / (target_cls >= 0).sum()

        return fast_rcnn_cls_loss, fast_rcnn_reg_loss


class FRCNNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rpn_loss = RPNLoss()
        self.fast_rcnn_loss = FastRCNNLoss()

    def forward(self, pred, target):

        pred_rpn_cls, pred_rpn_reg, pred_fast_rcnn_cls, pred_fast_rcnn_reg = pred
        target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg = target

        # bbox normalization
        # target_fast_rcnn_reg /= torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(torch.get_device(target_fast_rcnn_reg))
        # target_rpn_reg *= torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(torch.get_device(target_fast_rcnn_reg))

        rpn_cls_loss, rpn_reg_loss = self.rpn_loss(pred_rpn_cls, pred_rpn_reg, target_rpn_cls, target_rpn_reg)
        fast_rcnn_cls_loss, fast_rcnn_reg_loss = self.fast_rcnn_loss(pred_fast_rcnn_cls, pred_fast_rcnn_reg,
                                                                     target_fast_rcnn_cls, target_fast_rcnn_reg)

        total_loss = rpn_cls_loss + rpn_reg_loss + fast_rcnn_cls_loss + fast_rcnn_reg_loss
        return total_loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss


if __name__ == '__main__':
    import time
    from PIL import Image
    import torchvision.transforms as tfs
    from models.model import FRCNN

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
    frcnn = FRCNN().cuda()
    pred, target = frcnn(img, bbox, label)
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

    criterion = FRCNNLoss()
    loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss = criterion(pred, target)
    print("total_loss :", loss)
    print("rpn_cls_loss :", rpn_cls_loss)
    print("rpn_reg_loss :", rpn_reg_loss)
    print("fast_rcnn_cls_loss :", fast_rcnn_cls_loss)
    print("fast_rcnn_reg_loss :", fast_rcnn_reg_loss)

