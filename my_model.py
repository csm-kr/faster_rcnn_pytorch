import torch
import numpy as np
import torch.nn as nn
from torchvision.models import vgg16


class RegionProposal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0


class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0


class FRCNNHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0


class TargetBuilder:
    def __init__(self, module_name):
        super().__init__()
        self.module_name = module_name

    def build_rpn_target(self, bbox, anchor):
        assert self.module_name == 'rpn'
        return 0

    def build_frcnn_target(self, bbox, label, rois):
        assert self.module_name == 'frcnn'
        return 0


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
        self.head = FRCNNHead()
        self.rpn_target_builder = TargetBuilder('rpn')
        self.frcnn_target_builder = TargetBuilder('frcnn')

    def forward(self, x):
        return 0

    def predict(self, x):
        return -1


if __name__ == '__main__':
    model = FRCNN()
    print(model.extractor)
    print(model.classifier)