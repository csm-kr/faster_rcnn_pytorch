import os
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import propose_region
from evaluation.evaluator import Evaluator


@ torch.no_grad()
def test_and_eval(epoch, device, vis, test_loader, model, opts, xl_log_saver=None, result_best=None):

    # 0. evaluator
    evaluator = Evaluator(data_type=opts.data_type)  # opts.data_type : voc or coco

    # 1. load .pth
    checkpoint = torch.load(f=os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(epoch)),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tic = time.time()
    sum_loss = 0

    for idx, data in enumerate(test_loader):

        images = data[0]
        boxes = data[1]
        labels = data[2]
        # info = data[3][0]  # [{}]

        # 2. load data to device
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # 3. forward(predict)
        pred_bboxes, pred_labels, pred_scores = model.predict(images, opts)

        if opts.data_type == 'voc':

            info = data[3][0]  # [{}]
            info = (pred_bboxes, pred_labels, pred_scores, info['name'], info['original_wh'])

        elif opts.data_type == 'coco':

            img_id = test_loader.dataset.img_id[idx]
            img_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
            coco_ids = test_loader.dataset.coco_ids
            info = (pred_bboxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

        # 4. get info for evaluation
        evaluator.get_info(info)

        # 5. print log
        toc = time.time()
        if idx % 1000 == 0 or idx == len(test_loader) - 1:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Time : {time:.4f}\t'
                  .format(epoch,
                          idx,
                          len(test_loader),
                          time=toc - tic))

    # calculate mAP
    mAP = evaluator.evaluate(test_loader.dataset)
    print(mAP)

    if vis is not None:
        # loss plot
        vis.line(X=torch.ones((1, 1)).cpu() * epoch,  # step
                 Y=torch.Tensor([mAP]).unsqueeze(0).cpu(),
                 win='test_results_of_' + opts.name,
                 update='append',
                 opts=dict(xlabel='step',
                           ylabel='test',
                           title='test loss',
                           legend=['mAP']))
    if xl_log_saver is not None:
        xl_log_saver.insert_each_epoch(contents=(epoch, mAP))

    # save best.pth.tar
    if result_best is not None:
        if result_best['mAP'] < mAP:
            print("update best model")
            result_best['epoch'] = epoch
            result_best['mAP'] = mAP
            torch.save(checkpoint, os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.best.pth.tar'))

        return result_best

import argparse
from model import FRCNN
from loss import FRCNNLoss
from dataset.build import build_dataset
from config import get_args_parser


def test_worker(rank, opts):

    # 1. config
    print(opts)

    # 2. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. visdom
    vis = None

    # 4. data(set/loader)
    _, test_loader = build_dataset(opts)

    # 5. model
    model = FRCNN(opts.num_classes)
    model = model.to(device)

    # 6. loss
    # criterion = FRCNNLoss()

    test_and_eval(epoch=opts.test_epoch,
                  device=device,
                  vis=vis,
                  test_loader=test_loader,
                  model=model,
                  opts=opts,
                  )


if __name__ == '__main__':
    import configargparse

    parser = configargparse.ArgumentParser('faster rcnn testing', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    print(opts)
    test_worker(0, opts)