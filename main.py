import os
import torch
import visdom

# config
import configargparse
from config import get_args_parser

# dataset / model / loss
# from datasets.build import build_dataloader
from new_datasets.build import build_dataloader
from models.build import build_model
from losses.build import build_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

# train and test
from train import train_one_epoch
from test import test_and_eval

# log
from log import XLLogSaver

# resume
from utils import init_for_distributed
from utils.util import resume

import torch.multiprocessing as mp


def main_worker(rank, opts):

    # 1. config
    print(opts)

    if opts.distributed:
        init_for_distributed(rank, opts)

    # 2. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. visdom
    vis = visdom.Visdom(port=opts.visdom_port)
    # vis = None

    # 4. data(set/loader)
    # seed = 0
    # torch.manual_seed(seed)
    train_loader, test_loader = build_dataloader(opts)

    # 5. model
    model = build_model(opts)
    model = model.to(device)

    # 6. loss
    criterion = build_loss(opts)

    # 7. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=opts.momentum,
                                weight_decay=opts.weight_decay)

    # 8. scheduler
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=opts.epoch, eta_min=0.00005)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[16, 22])

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
        # train_one_epoch(opts=opts,
        #                 epoch=epoch,
        #                 device=device,
        #                 vis=vis,
        #                 train_loader=train_loader,
        #                 model=model,
        #                 criterion=criterion,
        #                 optimizer=optimizer,
        #                 scheduler=scheduler)

        # 12. test and evaluation
        test_and_eval(opts=opts,
                      epoch=epoch,
                      device=device,
                      vis=vis,
                      test_loader=test_loader,
                      model=model,
                      xl_log_saver=xl_log_saver,
                      result_best=result_best,
                      is_load=True)

        scheduler.step()


if __name__ == '__main__':
    parser = configargparse.ArgumentParser('Faster RCNN training', parents=[get_args_parser()])
    opts = parser.parse_args()

    if len(opts.gpu_ids) > 1:
        opts.distributed = True

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 1

    if opts.distributed:
        mp.spawn(main_worker,
                 args=(opts,),
                 nprocs=opts.world_size,
                 join=True)
    else:
        main_worker(opts.rank, opts)