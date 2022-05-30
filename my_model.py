import torch
import numpy as np
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.ops import nms
from anchor import FRCNNAnchorMaker
from utils import xy_to_cxcy, cxcy_to_xy, encode, decode, find_jaccard_overlap
from torchvision.ops import RoIPool


class RegionProposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.min_size = 16

    def forward(self, reg, cls, anchor, mode):
        # pred reg, pred cls로 nms 해서 2000 / 300 개 고르는 곳
        # ** 1. set nms top k **
        pre_nms_top_k = 12000
        post_num_top_k = 2000
        if mode == 'test':
            pre_nms_top_k = 6000
            post_num_top_k = 300

        # ** 2. make pred reg to bbox coord using tensor anchor **
        anchor_tensor = anchor
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
        roi_tensor = roi_tensor[keep_idx].detach()  # ** important : detach function makes normalization possible **

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
        device = features.get_device()

        # ** 1. make anchor **
        # the shape of anchor - [f_h * f_w * 9 , 4]
        anchor = self.anchor_maker._enumerate_shifted_anchor(anchor_base=np.array(self.anchor_base,
                                                                                  dtype=np.float32),
                                                             origin_image_size=img_size)  # H, W
        anchor = torch.from_numpy(anchor).to(device)

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
    def __init__(self,
                 num_classes,
                 roi_size,
                 classifier
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = nn.Linear(4096, num_classes)      # roi 에 대하여 클래스를 만들어야 하므로
        self.reg_head = nn.Linear(4096, num_classes * 4)  # 각 클래스별로 coord 를 만들어야 하므로
        self.roi_pool = RoIPool(output_size=(roi_size, roi_size), spatial_scale=1.)
        self.classifier = classifier

        # initialization
        normal_init(self.cls_head, 0, 0.01)
        normal_init(self.reg_head, 0, 0.001)

    def forward(self, features, roi):
        
        # ** roi 가 0 ~ 1 사이의 값으로 들어온다. --> scale roi **
        device = features.get_device()
        f_height, f_width = features.size()[2:]
        scale_from_roi_to_feature = torch.FloatTensor([f_width, f_height, f_width, f_height]).to(device)
        sclaed_roi = roi * scale_from_roi_to_feature
        scaled_roi_list = [sclaed_roi]  # for make it input of roi pool
        
        pool = self.roi_pool(features, scaled_roi_list)                # [128, 512, 7, 7]
        x = pool.view(pool.size(0), -1)                                # 128, 512 * 7 * 7

        # ** fast rcnn forward head ** #
        x = self.classifier(x)                                         # 1, 128, 4096
        pred_fast_rcnn_reg = self.reg_head(x)                          # 1, 128, 21 * 4
        pred_fast_rcnn_cls = self.cls_head(x)                          # 1, 128, 21
        return pred_fast_rcnn_reg, pred_fast_rcnn_cls


class TargetBuilder:
    def __init__(self, module_name):
        super().__init__()
        self.module_name = module_name

    def build_rpn_target(self, bbox, anchor):
        assert self.module_name == 'rpn'
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
        return rpn_tg_reg, rpn_tg_cls

    def build_fast_rcnn_target(self, bbox, label, rois):

        assert self.module_name == 'fast_rcnn'
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
        n_pos = int(min((IoU_max >= 0.5).sum(), 32))

        # random select pos and neg indices
        # pos_index = torch.arange(IoU_max.size(0))[IoU_max >= 0.5]
        # perm = torch.randperm(pos_index.size(0))
        # pos_index = pos_index[perm[:n_pos]]

        import numpy as np
        pos_index = torch.arange(IoU_max.size(0))[IoU_max >= 0.5].cpu().numpy()
        if pos_index.size > 0:
            np.random.seed(111)
            pos_index = np.random.choice(pos_index, size=n_pos, replace=False)

        n_neg = 128 - n_pos

        neg_index = torch.arange(IoU_max.size(0))[(IoU_max < 0.5) & (IoU_max >= 0.0)].cpu().numpy()
        if neg_index.size > 0:
            # print(neg_index.size)
            np.random.seed(111)
            neg_index = np.random.choice(neg_index, size=128 - n_pos, replace=False)

        # neg_index = torch.arange(IoU_max.size(0))[(IoU_max < 0.5) & (IoU_max >= 0.0)]
        # perm = torch.randperm(neg_index.size(0))
        # neg_index = neg_index[perm[:n_neg]]

        assert n_neg + n_pos == 128

        import numpy as np
        keep_index = np.concatenate([pos_index, neg_index], axis=-1)

        # keep_index = torch.cat([pos_index, neg_index], dim=-1)

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
        device = torch.get_device(fast_rcnn_tg_reg)
        mean = torch.FloatTensor([0., 0., 0., 0.]).to(device)
        std = torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(device)
        fast_rcnn_tg_reg = (fast_rcnn_tg_reg - mean) / std

        return fast_rcnn_tg_reg, fast_rcnn_tg_cls, sample_rois


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
        self.head = FRCNNHead(num_classes=21, roi_size=7, classifier=self.classifier)
        self.rpn_target_builder = TargetBuilder('rpn')
        self.fast_rcnn_target_builder = TargetBuilder('fast_rcnn')

    def forward(self, x, bbox, label):

        features = self.extractor(x)
        # forward rpn
        pred_rpn_reg, pred_rpn_cls, rois, anchor = self.rpn(features, x.size()[2:])

        # make target for rpn
        target_rpn_reg, target_rpn_cls = self.rpn_target_builder.build_rpn_target(bbox=bbox,
                                                                                  anchor=anchor)
        # print(rois.shape)

        # make target for fast rcnn
        target_fast_rcnn_reg, target_fast_rcnn_cls, sample_rois = self.fast_rcnn_target_builder.build_fast_rcnn_target(
            bbox=bbox,
            label=label,
            rois=rois)
        pred_fast_rcnn_reg, pred_fast_rcnn_cls = self.head(features, sample_rois)
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(128, -1, 4)
        pred_fast_rcnn_reg = pred_fast_rcnn_reg[torch.arange(0, 128).long(), target_fast_rcnn_cls.long()]

        return (pred_rpn_cls, pred_rpn_reg, pred_fast_rcnn_cls, pred_fast_rcnn_reg), \
               (target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg)

    def predict(self, x, visualization=False):
        # feature extractor
        features = self.extractor(x)
        # each image has different anchor

        # forward rpn
        pred_rpn_reg, pred_rpn_cls, rois, _ = self.rpn(features, x.size()[2:], mode='test')
        pred_fast_rcnn_reg, pred_fast_rcnn_cls = self.head(features, rois)
        # pred_fast_rcnn_cls : [128, 21]
        # pred_fast_rcnn_reg : [128, 84] - [128, 21, 4]

        # make pred prob and bbox
        pred_cls = (torch.softmax(pred_fast_rcnn_cls, dim=-1))  # batch 없애는 부분
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(-1, 21, 4)  # ex) [184, 21, 4]
        pred_fast_rcnn_reg = pred_fast_rcnn_reg * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(
            torch.get_device(pred_fast_rcnn_reg))
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


def normal_init(m, mean, stddev):
    import torch
    torch.manual_seed(111)
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()


if __name__ == '__main__':
    model = FRCNN()
    print(model.extractor)
    print(model.classifier)