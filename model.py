import cv2
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

    def forward(self, cls, reg, anchor, mode):

        # 0. make foreground_softmax_pred_rpn_cls
        cls = torch.softmax(cls, dim=-1)[..., 1]  # [16650]

        # pred reg, pred cls로 nms 해서 2000 / 300 개 고르는 곳
        # 1. set nms top k
        pre_nms_top_k = 12000
        post_num_top_k = 2000
        if mode == 'test':
            pre_nms_top_k = 6000
            post_num_top_k = 300

        # 2. make pred reg to bbox coord using tensor anchor
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
        sorted_scores = sorted_scores[:pre_nms_top_k]  # [12000]

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

        normal_init(self.inter_layer, 0, 0.01)
        normal_init(self.cls_layer, 0, 0.01)
        normal_init(self.reg_layer, 0, 0.01)

    def forward(self, features):
        batch_size = features.size(0)
        # if image size is # [1, 3, 600, 800]
        x = torch.relu(self.inter_layer(features))                                    # [1, 512, 37, 50]
        pred_cls = self.cls_layer(x)                                                  # [1, 18, 37, 50]
        pred_reg = self.reg_layer(x)                                                  # [1, 36, 37, 50]
        pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # [1, 16650, 4]
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)  # [1, 16650, 2]
        return pred_cls, pred_reg


class FastRCNNHead(nn.Module):
    def __init__(self,
                 num_classes,
                 roi_size,
                 classifier
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = nn.Linear(4096, num_classes)  # roi 에 대하여 클래스를 만들어야 하므로
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

        # ** roi pool **
        pool = self.roi_pool(features, scaled_roi_list)  # [128, 512, 7, 7]
        x = pool.view(pool.size(0), -1)  # 128, 512 * 7 * 7

        # ** fast rcnn forward head ** #
        x = self.classifier(x)  # 1, 128, 4096
        pred_fast_rcnn_cls = self.cls_head(x)  # 1, 128, 21
        pred_fast_rcnn_reg = self.reg_head(x)  # 1, 128, 21 * 4
        return pred_fast_rcnn_cls, pred_fast_rcnn_reg


class FastRcnnTargetMaker(nn.Module):
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
        n_pos = int(min((IoU_max >= 0.5).sum(), 32))

        # random select pos and neg indices
        pos_index = torch.arange(IoU_max.size(0))[IoU_max >= 0.5]
        perm = torch.randperm(pos_index.size(0))
        pos_index = pos_index[perm[:n_pos]]

        # import numpy as np
        # pos_index = torch.arange(IoU_max.size(0))[IoU_max >= 0.5].cpu().numpy()
        # if pos_index.size > 0:
        #     np.random.seed(111)
        #     pos_index = np.random.choice(pos_index, size=n_pos, replace=False)

        n_neg = 128 - n_pos

        # neg_index = torch.arange(IoU_max.size(0))[(IoU_max < 0.5) & (IoU_max >= 0.0)].cpu().numpy()
        # if neg_index.size > 0:
        #     # print(neg_index.size)
        #     np.random.seed(111)
        #     neg_index = np.random.choice(neg_index, size=128 - n_pos, replace=False)

        neg_index = torch.arange(IoU_max.size(0))[(IoU_max < 0.5) & (IoU_max >= 0.0)]
        perm = torch.randperm(neg_index.size(0))
        neg_index = neg_index[perm[:n_neg]]

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

        return fast_rcnn_tg_cls, fast_rcnn_tg_reg, sample_rois


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
        label = -1 * torch.ones(num_anchors, dtype=torch.float32, device=bbox.get_device())

        iou = find_jaccard_overlap(anchor, bbox)  # [num anchors, num objects]
        IoU_max, IoU_argmax = iou.max(dim=1)

        # 2-1. set negative label
        label[IoU_max < 0.3] = 0
        n_neg = (label == 0).sum()

        if n_neg <= 200:
            print(n_neg)

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

        # print(n_pos)
        # print(n_neg)

        if n_pos > 128:

            pos_indices = torch.arange(label.size(0))[label == 1]
            perm = torch.randperm(pos_indices.size(0))
            label[pos_indices[perm[128:]]] = -1  # convert pos label to ignore label

        if n_neg > 256 - n_pos:
            if n_pos > 128:
                n_pos = 128
            neg_indices = torch.arange(label.size(0))[label == 0]
            perm = torch.randperm(neg_indices.size(0))
            label[neg_indices[perm[(256 - n_pos):]]] = -1  # convert neg label to ignore label

        assert (label == 1).sum() + (label == 0).sum() > 200, \
            'less than 200 addition? pos : {} vs neg : {}'.format((label == 1).sum(), (label == 0).sum())

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


class FRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # backbone
        backbone = vgg16(pretrained=True)

        # extractor
        self.extractor = nn.Sequential(
            *list(backbone.features.children())[:-1]
        )
        self.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=4096),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=4096, out_features=4096),
                                        nn.ReLU(inplace=True))

        # region proposal network
        self.rpn = RegionProposalNetwork()
        # region proposal
        self.rp = RegionProposal()
        # anchor
        self.anchor_maker = FRCNNAnchorMaker()
        # rpn target
        self.rpn_target_maker = RPNTargetMaker()
        # fast rcnn target
        self.fast_rcnn_target_maker = FastRcnnTargetMaker()
        # fast rcnn head
        self.fast_rcnn_head = FastRCNNHead(num_classes=21, roi_size=7, classifier=self.classifier)
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, bbox, label):

        # 1. extract features
        features = self.extractor(x)

        # 2. make anchor
        anchor = self.anchor_maker._enumerate_shifted_anchor(origin_image_size=x.size()[2:])
        device = features.get_device()
        anchor = torch.from_numpy(anchor).to(device)

        # 3. forward rpn
        pred_rpn_cls, pred_rpn_reg = self.rpn(features)

        # 4. propose region -> roi
        rois = self.rp(cls=pred_rpn_cls.squeeze(0),
                       reg=pred_rpn_reg.squeeze(0),
                       anchor=anchor,
                       mode='train')

        # 5. make target for rpn
        target_rpn_cls, target_rpn_reg = self.rpn_target_maker(bbox=bbox,
                                                               anchor=anchor)

        # 6. make target for fast rcnn
        target_fast_rcnn_cls, target_fast_rcnn_reg, sample_rois = self.fast_rcnn_target_maker(bbox=bbox,
                                                                                              label=label,
                                                                                              rois=rois)

        # 7. forward fast rcnn head
        pred_fast_rcnn_cls, pred_fast_rcnn_reg = self.fast_rcnn_head(features, sample_rois)
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(128, -1, 4)
        pred_fast_rcnn_reg = pred_fast_rcnn_reg[torch.arange(0, 128).long(), target_fast_rcnn_cls.long()]

        return (pred_rpn_cls, pred_rpn_reg, pred_fast_rcnn_cls, pred_fast_rcnn_reg), \
               (target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg)

    def predict(self, x, visualization=False):

        # 1. extract features
        features = self.extractor(x)

        # 2. make anchor
        anchor = self.anchor_maker._enumerate_shifted_anchor(origin_image_size=x.size()[2:])
        device = features.get_device()
        anchor = torch.from_numpy(anchor).to(device)

        # 3. forward rpn
        pred_rpn_cls, pred_rpn_reg = self.rpn(features)

        # 4. propose region -> roi
        rois = self.rp(cls=pred_rpn_cls.squeeze(0),
                       reg=pred_rpn_reg.squeeze(0),
                       anchor=anchor,
                       mode='test')

        # 5. forward fast_rcnn_head
        pred_fast_rcnn_cls, pred_fast_rcnn_reg = self.fast_rcnn_head(features, rois)

        # make pred prob and bbox(post process)
        pred_cls = (torch.softmax(pred_fast_rcnn_cls, dim=-1))      # batch 없애는 부분
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(-1, 21, 4)  # ex) [184, 21, 4]
        pred_fast_rcnn_reg = pred_fast_rcnn_reg * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(torch.get_device(pred_fast_rcnn_reg))
        rois = rois.reshape(-1, 1, 4).expand_as(pred_fast_rcnn_reg)
        pred_bbox = decode(pred_fast_rcnn_reg.reshape(-1, 4), xy_to_cxcy(rois.reshape(-1, 4)))
        pred_bbox = cxcy_to_xy(pred_bbox)

        pred_bbox = pred_bbox.reshape(-1, 21 * 4)
        pred_bbox = pred_bbox.clamp(min=0, max=1)
        bbox, label, score = self._suppress(pred_bbox, pred_cls)

        if visualization:

            img_height, img_width = x.size()[2:]
            multiplier = np.array([img_width, img_height, img_width, img_height])
            bbox *= multiplier

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

                from utils import voc_color_array, voc_label_map
                voc_label_list = list(voc_label_map.keys())

                x_min = int(bbox[j][0])
                y_min = int(bbox[j][1])
                x_max = int(bbox[j][2])
                y_max = int(bbox[j][3])

                cv2.rectangle(im_show,
                              pt1=(x_min, y_min),
                              pt2=(x_max, y_max),
                              color=voc_color_array[label[j]],
                              thickness=2)

                # text_size
                text_size = cv2.getTextSize(text=voc_label_list[label[j]] + ' {:.2f}'.format(score[j].item()),
                                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                                            fontScale=1,
                                            thickness=1)[0]

                # text_rec
                cv2.rectangle(im_show,
                              pt1=(x_min, y_min),
                              pt2=(x_min + text_size[0] + 3, y_min + text_size[1] + 4),
                              color=voc_color_array[label[j]],
                              thickness=-1)

                # put text
                cv2.putText(im_show,
                            text=voc_label_list[label[j]] + ' {:.2f}'.format(score[j].item()),
                            org=(x_min + 10, y_min + 10),   # must be int
                            fontFace=0,
                            fontScale=0.4,
                            color=(0, 0, 0))

            return bbox, label, score, im_show
        return bbox, label, score

    def _suppress(self, raw_cls_bbox, raw_prob):
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
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()


if __name__ == '__main__':
    model = FRCNN()
    print(model.extractor)
    print(model.classifier)