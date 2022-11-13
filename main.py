import os
import torch
import visdom

# config
import configargparse
from config import get_args_parser

# dataset / model / loss
from dataset.build import build_dataset
from models.build import build_model
from loss import FRCNNLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

# train and test
from train import train_one_epoch
from test import test_and_eval

# log
from log import XLLogSaver

# resume
from utils import resume

import torch.multiprocessing as mp


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
    model = build_model(opts)
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

        # 11. train one epoch
        train_one_epoch(opts=opts,
                        epoch=epoch,
                        device=device,
                        vis=vis,
                        train_loader=train_loader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler)

        # 12. test and evaluation
        test_and_eval(opts=opts,
                      epoch=epoch,
                      device=device,
                      vis=vis,
                      test_loader=test_loader,
                      model=model,
                      xl_log_saver=xl_log_saver,
                      result_best=result_best,
                      is_load=False)
        scheduler.step()


if __name__ == '__main__':
    parser = configargparse.ArgumentParser('Faster RCNN training', parents=[get_args_parser()])
    opts = parser.parse_args()

    if len(opts.gpu_ids) > 1:
        opts.distributed = True

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    if opts.distributed:
        mp.spawn(main_worker,
                 args=(opts,),
                 nprocs=opts.world_size,
                 join=True)
    else:
        main_worker(opts.rank, opts)
