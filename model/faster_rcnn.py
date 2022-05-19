import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool
import numpy as np
from utils import propose_region
from model.rpn import RPN
from anchor import FRCNNAnchorMaker
from model.rpn import normal_init
from model.target_builder import FastRCNNTargetBuilder
from model.target_builder import RPNTargetBuilder
import time
from utils import decode, xy_to_cxcy, cxcy_to_xy


class FRCNNHead(nn.Module):
    def __init__(self, num_classes, roi_output_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = nn.Linear(4096, num_classes)
        self.reg_head = nn.Linear(4096, num_classes * 4)
        self.roi_pool = RoIPool(output_size=(roi_output_size, roi_output_size), spatial_scale=1.)

        import torchvision
        self.fc = torchvision.models.vgg16(True).classifier
        del self.fc[6]
        del self.fc[5]
        del self.fc[2]

        # self.fc = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(4096, 4096),
        #                         nn.ReLU(inplace=True)
        #                         )

        normal_init(self.cls_head, 0, 0.001)
        normal_init(self.reg_head, 0, 0.01)
        # normal_init(self.fc, 0, 0.01)

    def initialize(self):

        for c in self.cls_head.children():
            if isinstance(c, nn.Linear):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

        for c in self.reg_head.children():
            if isinstance(c, nn.Linear):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)
        for c in self.fc.children():
            if isinstance(c, nn.Linear):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

    def forward(self, features, rois):

        device = rois.get_device()
        filtered_rois_numpy = np.array(rois.detach().cpu()).astype(np.float32)

        # 3. scale original box w, h to feature w, h
        filtered_rois_numpy[:, ::2] = filtered_rois_numpy[:, ::2] * features.size(3)
        filtered_rois_numpy[:, 1::2] = filtered_rois_numpy[:, 1::2] * features.size(2)

        # 4. convert numpy boxes to list of tensors
        filtered_boxes_tensor = [torch.FloatTensor(filtered_rois_numpy).to(device)]

        # --------------- RoI Pooling --------------- #
        x = self.roi_pool(features, filtered_boxes_tensor)             # [2000, 512, 7, 7]
        x = x.view(x.size(0), -1)                                      # 2000, 512 * 7 * 7

        # --------------- forward head --------------- #
        x = self.fc(x)                                                 # 2000, 4096
        cls = self.cls_head(x)                                         # 2000, 21
        reg = self.reg_head(x)                                         # 2000, 21 * 4
        frcnn_cls = cls.view(1, -1, self.num_classes)                  # [1, 2000, 21]
        frcnn_reg = reg.view(1, -1, 4)                                 # [1, 2000, 21 * 4]
        return frcnn_cls, frcnn_reg


class FRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ** for forward
        self.extractor = nn.Sequential(*list(vgg16(pretrained=True).features.children())[:-1])
        self.rpn = RPN()
        self.rpn_target_builder = RPNTargetBuilder()
        self.fast_rcnn_target_builder = FastRCNNTargetBuilder()
        self.head = FRCNNHead(num_classes=21)

        # ** for anchor
        self.anchor_maker = FRCNNAnchorMaker()
        self.anchor_base = self.anchor_maker.generate_anchor_base()

    def forward(self, x, bbox, label):
        # x : image [B, 3, H, W]
        features = self.extractor(x)

        # each image has different anchor
        anchor = self.anchor_maker._enumerate_shifted_anchor(self.anchor_base, origin_image_size=x.size()[2:])
        anchor = torch.from_numpy(anchor).to(x.get_device())  # assign device

        # make target for rpn
        target_rpn_cls, target_rpn_reg = self.rpn_target_builder.build_rpn_target(bbox=bbox,
                                                                                  anchor=anchor)
        # forward rpn
        pred_rpn_cls, pred_rpn_reg, rois = self.rpn(features, anchor, mode='train')


        # make target for fast rcnn
        target_fast_rcnn_cls, target_fast_rcnn_reg, sample_rois = self.fast_rcnn_target_builder.build_fast_rcnn_target(bbox=bbox,
                                                                                                                       label=label,
                                                                                                                       rois=rois)
        # forward frcnn (frcnn을 forward 하려면 sampled roi (bbox가 필요하기에 여기서 만듦)
        pred_fast_rcnn_cls, pred_fast_rcnn_reg = self.head(features, sample_rois)

        # 각 class 에 대한 값 박스좌표를 모두 예측합
        # print(pred_fast_rcnn_reg.size())  # [1, 2688, 4]
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(128, -1, 4)
        pred_fast_rcnn_reg = pred_fast_rcnn_reg[torch.arange(0, 128), target_fast_rcnn_cls.long()]

        return (pred_rpn_cls, pred_rpn_reg, pred_fast_rcnn_cls, pred_fast_rcnn_reg), \
               (target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg)

    def predict(self, x, visualization=False):
        # feature extractor
        features = self.extractor(x)
        # each image has different anchor
        anchor = self.anchor_maker._enumerate_shifted_anchor(self.anchor_base, origin_image_size=x.size()[2:])
        anchor = torch.from_numpy(anchor).to(x.get_device())  # assign device
        # forward rpn
        pred_rpn_cls, pred_rpn_reg, rois = self.rpn(features, anchor, mode='test')
        # forward frcnn (frcnn을 forward 하려면 sampled roi (bbox가 필요하기에 여기서 만듦)
        pred_fast_rcnn_cls, pred_fast_rcnn_reg = self.head(features, rois)

        # pred_fast_rcnn_cls : [128, 21]
        # pred_fast_rcnn_reg : [128, 84] - [128, 21, 4]

        # make pred prob and bbox
        pred_cls = (torch.softmax(pred_fast_rcnn_cls[0], dim=-1))

        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(-1, 21, 4)  # ex) [184, 21, 4]
        rois = rois.reshape(-1, 1, 4).expand_as(pred_fast_rcnn_reg)
        pred_bbox = decode(pred_fast_rcnn_reg.reshape(-1, 4), xy_to_cxcy(rois.reshape(-1, 4)))
        pred_bbox = cxcy_to_xy(pred_bbox)

        pred_bbox = pred_bbox.reshape(-1, 21 * 4)
        pred_bbox = pred_bbox.clamp(min=0, max=1)
        bbox, label, score = self._suppress(pred_bbox, pred_cls)

        cv2_vis = visualization
        if cv2_vis:
            import cv2
            img_height, img_width = x.size()[2:]
            multiplier = np.array([img_width, img_height, img_width, img_height])
            bbox *= multiplier
            # print(bbox)

            # 0. permute
            images = x.cpu()
            images = images.squeeze(0).permute(1, 2, 0)  # B, C, H, W --> H, W, C

            # 1. un normalization
            images *= torch.Tensor([0.229, 0.224, 0.225])
            images += torch.Tensor([0.485, 0.456, 0.406])

            # 2. RGB to BGR
            image_np = images.numpy()

            x_img = image_np
            im_show = cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR)
            for j in range(len(bbox)):
                cv2.rectangle(im_show,
                              (int(bbox[j][0]), int(bbox[j][1])),
                              (int(bbox[j][2]), int(bbox[j][3])),
                              (0, 0, 255),
                              1)

            cv2.imshow('result', im_show)
            cv2.waitKey(0)
        return bbox, label, score

    def _suppress(self, raw_cls_bbox, raw_prob):
        from torchvision.ops import nms

        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, 21):
            cls_bbox_l = raw_cls_bbox.reshape((-1, 21, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > 0.05
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


if __name__ == '__main__':

    tic = time.time()
    img1 = torch.randn([1, 3, 600, 1000]).cuda()  # 37, 62
    print("image cuda time :", time.time() - tic)
    #
    # # load image and check frcnn shape
    # # 1. load image using PIL library
    # from PIL import Image
    # img1 = Image.open('../figures/000001.jpg').convert('RGB')
    #
    # # 2. transform resize and tensor
    # from dataset.detection_transforms import DetResize, DetToTensor, DetRandomHorizontalFlip, DetNormalize
    #
    # img1 = DetRandomHorizontalFlip()(img1)
    # img1 = DetToTensor()(img1)
    # img1 = DetResize(size=600, max_size=1000)(img1)
    # img1 = DetNormalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])(img1)


    #  **** real image 1 make
    from PIL import Image
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
    img = image_tensor.cuda()
    bbox = boxes_tensor_scale_1
    label = label_tensor

    # label =
    tic = time.time()
    frcnn = FRCNN().cuda()
    (pred_rpn_cls, pred_rpn_reg, pred_fast_cls, pred_fast_rcnn_reg), \
    (target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg) = frcnn(img, bbox, label)

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

