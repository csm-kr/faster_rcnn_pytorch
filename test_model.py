from __future__ import absolute_import
import torch
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool, nms

from model.target_builder import RPNTargetBuilder, FastRCNNTargetBuilder
from anchor import FRCNNAnchorMaker
import numpy as np
from utils import decode, cxcy_to_xy, xy_to_cxcy


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch.Tensor):
        return data.item()

def decom_vgg16():

    model = vgg16(pretrained=True)
    features = list(model.features)[:30]
    classifier = model.classifier
    classifier = list(classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]
    classifier = nn.Sequential(*classifier)
    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    return nn.Sequential(*features), classifier


class RegionProposalNetwork(nn.Module):
    def __init__(
            self,
            in_channels=512,
            mid_channels=512,
            ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32],
            feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_maker = FRCNNAnchorMaker()
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, 9 * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, 9 * 4, 1, 1, 0)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, mode='train'):
        n, _, hh, ww = x.shape
        # 1. base를 이용해서 anchor 만들기
        anchor = self.anchor_maker._enumerate_shifted_anchor(anchor_base=np.array(self.anchor_maker.generate_anchor_base(), dtype=np.float32),
                                                             origin_image_size=img_size)  # H, W
        # anchor = np.round(anchor, 4)
        n_anchor = anchor.shape[0] // (hh * ww)

        # 2. rpn forward
        h = torch.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()

        rpn_softmax_scores = torch.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        # rpn_softmax_scores = torch.softmax(rpn_scores.view(n, -1, 2), dim=-1)[..., 1]


        rpn_scores = rpn_scores.view(n, -1, 2)

        # ------------------------------

        rois = list()
        roi_indices = list()

        roi = self.proposal_layer(
            rpn_locs[0],
            rpn_fg_scores[0],
            anchor,
            img_size,
            mode=mode)
        batch_index = 0 * np.ones((len(roi),), dtype=np.int32)
        rois.append(roi)
        roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


class FasterRCNNVGG16(nn.Module):
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        super().__init__()
        self.extractor, self.classifier = decom_vgg16()

        self.rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        self.head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=1.,
            classifier=self.classifier
        )
        self.rpn_target_builder = RPNTargetBuilder()
        self.fast_rcnn_target_builder = FastRCNNTargetBuilder()

    def predict(self, x, visualization=False):
        # feature extractor
        features = self.extractor(x)
        # each image has different anchor

        # forward rpn
        pred_rpn_cls, pred_rpn_reg, rois, _, _ = self.rpn(features, x.size()[2:], mode='test')
        roi_index = torch.zeros(len(rois))
        pred_fast_rcnn_reg, pred_fast_rcnn_cls = self.head(features, rois, roi_index)
        # pred_fast_rcnn_cls : [128, 21]
        # pred_fast_rcnn_reg : [128, 84] - [128, 21, 4]

        rois = torch.from_numpy(rois).to(pred_fast_rcnn_cls.get_device())

        # make pred prob and bbox
        pred_cls = (torch.softmax(pred_fast_rcnn_cls, dim=-1))  # batch 없애는 부분
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(-1, 21, 4)  # ex) [184, 21, 4]
        pred_fast_rcnn_reg = pred_fast_rcnn_reg * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(torch.get_device(pred_fast_rcnn_reg))
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

    def forward(self, x, bbox, label):

        # x : image [B, 3, H, W]
        features = self.extractor(x)

        # forward rpn
        pred_rpn_reg, pred_rpn_cls, rois, roi_indices, anchor = self.rpn(features, x.size()[2:])

        # make target for rpn
        target_rpn_cls, target_rpn_reg = self.rpn_target_builder.build_rpn_target(bbox=bbox,
                                                                                  anchor=anchor)
        print(rois.shape)

        # make target for fast rcnn
        target_fast_rcnn_cls, target_fast_rcnn_reg, sample_rois = self.fast_rcnn_target_builder.build_fast_rcnn_target(bbox=bbox,
                                                                                                                       label=label,
                                                                                                                       rois=rois)
        # forward frcnn (frcnn을 forward 하려면 sampled roi (bbox가 필요하기에 여기서 만듦)
        sample_roi_index = torch.zeros(len(sample_rois))
        pred_fast_rcnn_reg, pred_fast_rcnn_cls  = self.head(features, sample_rois, sample_roi_index)

        # 각 class 에 대한 값 박스좌표를 모두 예측합
        # print(pred_fast_rcnn_reg.size())  # [1, 2688, 4] = [1, 128 * 21, 4]
        pred_fast_rcnn_reg = pred_fast_rcnn_reg.reshape(128, -1, 4)
        pred_fast_rcnn_reg = pred_fast_rcnn_reg[torch.arange(0, 128).long(), target_fast_rcnn_cls.long()]

        return (pred_rpn_cls, pred_rpn_reg, pred_fast_rcnn_cls, pred_fast_rcnn_reg), \
               (target_rpn_cls, target_rpn_reg, target_fast_rcnn_cls, target_fast_rcnn_reg)


class ProposalCreator:
    def __init__(self,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):

        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self,
                 loc,
                 score,
                 anchor,
                 img_size,
                 mode='train'):

        pre_nms_top_k = 12000
        post_num_top_k = 2000
        if mode == 'test':
            pre_nms_top_k = 6000
            post_num_top_k = 300

        anchor_tensor = torch.from_numpy(anchor)
        from utils import xy_to_cxcy, decode, encode, cxcy_to_xy
        roi_tensor = cxcy_to_xy(decode(loc, xy_to_cxcy(anchor_tensor.to(loc.get_device())))).clamp(0, 1)

        # 3. keep longer than minimum size
        ws = roi_tensor[:, 2] - roi_tensor[:, 0]
        hs = roi_tensor[:, 3] - roi_tensor[:, 1]
        keep = (hs >= (self.min_size / 1000)) & (ws >= (self.min_size / 1000))  # [17173]
        print(keep)
        roi_tensor = roi_tensor[keep, :]
        softmax_pred_scores = score[keep]

        # 4. nms
        sorted_scores, sorted_scores_indices = softmax_pred_scores.sort(descending=True)
        pre_nms_top_k = pre_nms_top_k
        if len(sorted_scores_indices) < pre_nms_top_k:
            pre_nms_top_k = len(sorted_scores_indices)
        roi_tensor = roi_tensor[sorted_scores_indices[:pre_nms_top_k]]      # [12000, 4]
        sorted_scores = sorted_scores[:pre_nms_top_k]         # [12000]

        # nms
        keep_idx = nms(boxes=roi_tensor, scores=sorted_scores, iou_threshold=0.7)
        keep_idx = keep_idx[:post_num_top_k]   # tensor([    0,     1,     2,  ..., 11960, 11982, 11997])
        roi_tensor = roi_tensor[keep_idx]

        roi = roi_tensor.detach().cpu().numpy()
        return roi


class VGG16RoIHead(nn.Module):
    def __init__(self,
                 n_class,
                 roi_size,
                 spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        # in case roi_indices is  ndarray
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        # xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        xy_indices_and_rois = indices_and_rois
        indices_and_rois = xy_indices_and_rois.contiguous()

        # roi_pooling 사용시, feature 의 size 를 맞춰 주어야 하기 때문에
        f_height, f_width = x.size()[2:]

        roi_to_feature_scale = np.array([1., f_width, f_height, f_width, f_height])
        roi_to_feature_scale = totensor(roi_to_feature_scale).float()
        indices_and_rois *= roi_to_feature_scale
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev):
    import torch
    torch.manual_seed(111)
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()
