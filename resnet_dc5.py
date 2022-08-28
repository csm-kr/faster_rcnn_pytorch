'''
GeneralizedRCNN(
  (backbone): ResNet(
    (stem): BasicStem(
      (conv1): Conv2d(
        3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
      )
    )
    (res2): Sequential(
      (0): BottleneckBlock(
        (shortcut): Conv2d(
          64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv1): Conv2d(
          64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
        (conv2): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
        (conv3): Conv2d(
          64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
      )
      (1): BottleneckBlock(
        (conv1): Conv2d(
          256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
        (conv2): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
        (conv3): Conv2d(
          64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
      )
      (2): BottleneckBlock(
        (conv1): Conv2d(
          256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
        (conv2): Conv2d(
          64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
        )
        (conv3): Conv2d(
          64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
      )
    )
    (res3): Sequential(
      (0): BottleneckBlock(
        (shortcut): Conv2d(
          256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
        (conv1): Conv2d(
          256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
          (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
        )
        (conv2): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
        )
        (conv3): Conv2d(
          128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
      )
      (1): BottleneckBlock(
        (conv1): Conv2d(
          512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
        )
        (conv2): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
        )
        (conv3): Conv2d(
          128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
      )
      (2): BottleneckBlock(
        (conv1): Conv2d(
          512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
        )
        (conv2): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
        )
        (conv3): Conv2d(
          128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
      )
      (3): BottleneckBlock(
        (conv1): Conv2d(
          512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
        )
        (conv2): Conv2d(
          128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
        )
        (conv3): Conv2d(
          128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
      )
    )
    (res4): Sequential(
      (0): BottleneckBlock(
        (shortcut): Conv2d(
          512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False
          (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
        )
        (conv1): Conv2d(
          512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
        )
      )
      (1): BottleneckBlock(
        (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
        )
      )
      (2): BottleneckBlock(
        (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
        )
      )
      (3): BottleneckBlock(
        (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
        )
      )
      (4): BottleneckBlock(
        (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
        )
      )
      (5): BottleneckBlock(
        (conv1): Conv2d(
          1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv2): Conv2d(
          256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
        )
        (conv3): Conv2d(
          256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
        )
      )
    )
    (res5): Sequential(
      (0): BottleneckBlock(
        (shortcut): Conv2d(
          1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
        )
        (conv1): Conv2d(
          1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
        (conv2): Conv2d(
          512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
        (conv3): Conv2d(
          512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
        )
      )
      (1): BottleneckBlock(
        (conv1): Conv2d(
          2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
        (conv2): Conv2d(
          512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
        (conv3): Conv2d(
          512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
        )
      )
      (2): BottleneckBlock(
        (conv1): Conv2d(
          2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
        (conv2): Conv2d(
          512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False
          (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
        )
        (conv3): Conv2d(
          512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
          (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
        )
      )
    )
  )
  (proposal_generator): RPN(
    (rpn_head): StandardRPNHead(
      (conv): Conv2d(
        2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        (activation): ReLU()
      )
      (objectness_logits): Conv2d(2048, 15, kernel_size=(1, 1), stride=(1, 1))
      (anchor_deltas): Conv2d(2048, 60, kernel_size=(1, 1), stride=(1, 1))
    )
    (anchor_generator): DefaultAnchorGenerator(
      (cell_anchors): BufferList()
    )
  )
  (roi_heads): StandardROIHeads(
    (box_pooler): ROIPooler(
      (level_poolers): ModuleList(
        (0): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
      )
    )
    (box_head): FastRCNNConvFCHead(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (fc1): Linear(in_features=100352, out_features=1024, bias=True)
      (fc_relu1): ReLU()
      (fc2): Linear(in_features=1024, out_features=1024, bias=True)
      (fc_relu2): ReLU()
    )
    (box_predictor): FastRCNNOutputLayers(
      (cls_score): Linear(in_features=1024, out_features=81, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=320, bias=True)
    )
  )
)
'''

import torch.nn as nn
from torchvision.models import resnet50


class Resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.resnet50 = resnet50(pretrained=pretrained)
        self.resnet50_list = nn.ModuleList(list(self.resnet50.children())[:-2])  # to layer 1
        # print(self.resnet50_list)

        # stride rebuild Conv2d(kernel_size=(1, 1), stride=(1, 1))-> Conv2d(kernel_size=(3, 3), stride=(2, 2))
        #                Conv2d(kernel_size=(1, 1), stride=(2, 2))-> Conv2d(kernel_size=(3, 3), stride=(1, 1))
        for i in range(5, 7):
            for j, m in enumerate(self.resnet50_list[i].children()):
                for k, b in enumerate(m.children()):  # bottleneck
                    if j == 0 and k == 0 and isinstance(b, nn.Conv2d):
                        b.stride = (2, 2)

                    elif isinstance(b, nn.Conv2d) and b.kernel_size == (3, 3):
                        b.stride = (1, 1)

        # make DC5
        for i in self.resnet50_list[7].children():
            for m in i.children():
                if isinstance(m, nn.Conv2d):
                    if m.kernel_size == (3, 3):
                        m.dilation = (2, 2)
                        m.padding = (2, 2)
                        m.stride = (1, 1)
                # for downsample
                if isinstance(m, nn.Sequential):
                    for k in m.children():
                        if isinstance(k, nn.Conv2d):
                            if k.kernel_size == (1, 1) and k.stride == (2, 2):
                                k.stride = (1, 1)

    def forward(self, x):

        x = self.resnet50_list[0](x)  # 7 x 7 conv 64
        x = self.resnet50_list[1](x)  # bn
        x = self.resnet50_list[2](x)  # relu
        x = self.resnet50_list[3](x)  # 3 x 3 maxpool

        x = self.resnet50_list[4](x)  # layer 1        # [B,  256, w/4, h/4]
        c3 = x = self.resnet50_list[5](x)  # layer 2   # [B,  512, w/8, h/8]
        c4 = x = self.resnet50_list[6](x)  # layer 3   # [B, 1024, w/16, h/16]
        c5 = x = self.resnet50_list[7](x)  # layer 4   # [B, 2048, w/32, h/32]

        return c5


if __name__ == '__main__':
    import torch
    from model import FRCNN
    # model = FRCNN(num_classes=81)
    # print(model)

    # from torchvision.models import resnet50
    # model = resnet50(pretrained=False)
    # print(model)

    model = Resnet50()
    print(model)

    img = torch.randn([3, 3, 600, 600])
    print(model(img).size())
