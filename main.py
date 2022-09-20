import os
import torch
import visdom

import configargparse
from config import get_args_parser

# dataset / model / loss
from dataset.build import build_dataset
from model import FRCNN
from loss import FRCNNLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

# train and test
from train import train_one_epoch
from test import test_and_eval

# log
from log import XLLogSaver
# resume
from utils import resume


def main_worker(rank, opts):

    # 1. config
    print(opts)

    # 2. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. visdom
    vis = visdom.Visdom(port=opts.visdom_port)

    # 4. data(set/loader)
    train_loader, test_loader = build_dataset(opts)

    # 5. model
    model = FRCNN(num_classes=opts.num_classes)
    model = model.to(device)

    # 6. loss
    criterion = FRCNNLoss()

    # 7. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=opts.momentum,
                                weight_decay=opts.weight_decay)

    # 8. scheduler
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=opts.epoch, eta_min=0.00005)

    # 9. logger
    xl_log_saver = None
    if opts.rank == 0:
        xl_log_saver = XLLogSaver(xl_folder_name=os.path.join(opts.log_dir, opts.name),
                                  xl_file_name=opts.name,
                                  tabs=('epoch', 'mAP'))

    # 10. resume
    model, optimizer, scheduler = resume(opts, model, optimizer, scheduler)

    # set best results
    result_best = {'epoch': 0, 'mAP': 0.}

    for epoch in range(opts.start_epoch, opts.epoch):

        # 10. train one epoch
        train_one_epoch(epoch=epoch,
                        device=device,
                        vis=vis,
                        train_loader=train_loader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        opts=opts)

        # 11. test and evaluation
        result_best = test_and_eval(epoch=epoch,
                                    device=device,
                                    vis=vis,
                                    test_loader=test_loader,
                                    model=model,
                                    xl_log_saver=xl_log_saver,
                                    opts=opts,
                                    result_best=result_best,
                                    )
        scheduler.step()


if __name__ == '__main__':
    parser = configargparse.ArgumentParser('Faster rcnn training', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    main_worker(opts.rank, opts)