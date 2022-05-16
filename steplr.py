import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
import matplotlib.pyplot as plt


def scheduler_experiments():
    whole_epoch = 4
    model = torch.nn.Sequential(torch.nn.Linear(10, 10),
                                torch.nn.ReLU(inplace=False))

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=1e-3,
                                momentum=0.9,
                                weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[3], gamma=0.1)
    # scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=100)

    lr_tracker = []
    for epoch in range(whole_epoch):
        # get lr
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        lr_tracker.append(lr)
        scheduler.step()
        print(lr)

    plt.figure('lr_tracker')
    import numpy as np
    x = np.arange(len(lr_tracker))
    y = np.asarray(lr_tracker)
    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    scheduler_experiments()