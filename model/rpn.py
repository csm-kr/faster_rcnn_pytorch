import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import vgg16_bn


class RPN(nn.Module):
    def __init__(self, conv_layers, num_anchors=9):
        super().__init__()
        self.conv_layers = nn.Sequential(*list(conv_layers.features.children())[:-1])  # after conv 5_3
        self.intermediate_layer = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.reg_layer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self.cls_layer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.intermediate_layer(x)
        cls = self.cls_layer(x)
        reg = self.reg_layer(x)
        return cls, reg


if __name__ == '__main__':
    img = torch.randn([2, 3, 600, 1000])  # 37, 62
    rpn = RPN(conv_layers=vgg16_bn(pretrained=True))
    cls, reg = rpn(img)

    print(cls.size())
    print(reg.size())
