import torch


class FasterRCNNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0
