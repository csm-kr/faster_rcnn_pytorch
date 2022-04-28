import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
from torchvision.ops import RoIPool
import numpy as np
from utils import propose_region
from model.rpn import RPN


class FRCNNHead(nn.Module):
    def __init__(self, num_classes, roi_output_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = nn.Linear(4096, num_classes)
        self.loc_head = nn.Linear(4096, num_classes * 4)
        self.roi_pool = RoIPool(output_size=(roi_output_size, roi_output_size), spatial_scale=1.)
        self.fc = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096, 4096),
                                nn.ReLU(inplace=True)
                                )
        self.initialize()

    def initialize(self):
        for c in self.cls_head.children():
            if isinstance(c, nn.Linear):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)
        for c in self.loc_head.children():
            if isinstance(c, nn.Linear):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)
        for c in self.fc.children():
            if isinstance(c, nn.Linear):
                nn.init.normal_(c.weight, std=0.01)
                nn.init.constant_(c.bias, 0)

    def forward(self, x_features, boxes):
        device = boxes.get_device()
        filtered_boxes_numpy = np.array(boxes.detach().cpu()).astype(np.float32)

        # 3. scale original box w, h to feature w, h
        filtered_boxes_numpy[:, ::2] = filtered_boxes_numpy[:, ::2] * x_features.size(3)
        filtered_boxes_numpy[:, 1::2] = filtered_boxes_numpy[:, 1::2] * x_features.size(2)

        # 4. convert numpy boxes to list of tensors
        filtered_boxes_tensor = [torch.FloatTensor(filtered_boxes_numpy).to(device)]

        # --------------- RoI Pooling --------------- #
        x = self.roi_pool(x_features, filtered_boxes_tensor)           # [2000, 512, 7, 7]
        x = x.view(x.size(0), -1)                                      # 2000, 512 * 7 * 7

        # --------------- forward head --------------- #
        x = self.fc(x)                                                 # 2000, 4096
        cls = self.cls_head(x)                                         # 2000, 21
        loc = self.loc_head(x)                                         # 2000, 21 * 4
        cls = cls.view(1, -1, self.num_classes)                        # [1, 2000, 21]
        loc = loc.view(1, -1, self.num_classes * 4)                    # [1, 2000, 21 * 4]
        return cls, loc


class FasterRCNN(nn.Module):
    def __init__(self, rpn, head, coder):
        super().__init__()
        self.rpn = rpn
        self.head = head
        self.coder = coder

    def forward(self, x):
        cls_rpn, reg_rpn, x_features = self.rpn(x, True)
        pred = (cls_rpn, reg_rpn)
        pred_boxes = propose_region(pred=pred,
                                    coder=self.coder)
        cls, reg = self.head(x_features, pred_boxes)
        return cls, reg


if __name__ == '__main__':
    img = torch.randn([1, 3, 600, 1000])  # 37, 62
    img = torch.randn([1, 3, 800, 800]).cuda()  # 37, 62
    rpn = RPN(vgg16_bn(pretrained=True))
    head = FRCNNHead(num_classes=20, roi_output_size=7)
    from coder import FasterRCNN_Coder
    coder = FasterRCNN_Coder()
    coder.set_anchors((800, 800))
    coder.center_anchor = coder.center_anchor.cuda()
    Fasterrcnn = FasterRCNN(rpn, head, coder=coder).cuda()
    cls, reg = Fasterrcnn(img)

    print(cls.size())
    print(reg.size())



