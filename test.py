import os
import time
import torch
from tqdm import tqdm
import numpy as np
from evaluation.evaluator import Evaluator
from evaluation.coco_eval import CocoEvaluator
from util.box_ops import box_cxcywh_to_xyxy
from utils.util import cxcy_to_xy


@ torch.no_grad()
def test_and_eval(opts, epoch, device, vis, test_loader, model, xl_log_saver=None, result_best=None, is_load=False):

    ### new evaluation
    iou_types = tuple(['bbox'])  # 'bbox'
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator = CocoEvaluator(test_loader.dataset.coco, iou_types)
    ###

    # 0. evaluator
    # evaluator = Evaluator(data_type=opts.data_type)  # opts.data_type : voc or coco

    # 1. device
    device = torch.device(f'cuda:{int(opts.gpu_ids[opts.rank])}')

    # 2. load pth
    checkpoint = None
    if is_load:
        checkpoint = torch.load(f=os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(epoch)),
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    tic = time.time()
    sum_loss = []

    for idx, data in enumerate(tqdm(test_loader)):

        images = data[0]
        targets = data[1]

        # cuda
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        boxes = targets[0]['boxes']
        labels = targets[0]['labels']

        # images = data[0]
        # boxes = data[1]
        # labels = data[2]
        #
        # # 2. load data to device
        # images = images.to(device)
        # boxes = [b.to(device) for b in boxes]
        # labels = [l.to(device) for l in labels]

        # 3. forward(predict)
        # pred_bboxes, pred_labels, pred_scores = model.module.predict(images, opts)
        boxes, labels, scores = model.module.predict(images, opts)

        # print(boxes.size())
        # print(labels.size())
        # print(scores.size())

        # cxcy2xyxy
        # boxes = box_cxcywh_to_xyxy(boxes)
        boxes = cxcy_to_xy(boxes).to(device)
        h, w = targets[0]['size']
        scale_fct = torch.stack([w, h, w, h], dim=0).to(device)
        boxes = boxes * scale_fct[None, :]

        # size 1, 100, 4
        boxes = boxes.unsqueeze(0)
        labels = labels.unsqueeze(0)
        scores = scores.unsqueeze(0)

        # print(boxes)
        # print(labels)
        # print(scores)
        # 다음 코드를 통과하기 위해서는 batch 로 묶여 있어야 한다.
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        print(results)

        res = {i['image_id'].item(): output for i, output in zip(targets, results)}
        print(res)
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if opts.data_type == 'voc':
        #
        #     info = data[3][0]  # [{}]
        #     info = (pred_bboxes, pred_labels, pred_scores, info['name'], info['original_wh'])
        #
        # elif opts.data_type == 'coco':
        #     # -- old dataset's version --
        #     # img_id = test_loader.dataset.img_id[idx]
        #     # img_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
        #     # coco_ids = test_loader.dataset.coco_ids
        #     # info = (pred_bboxes, pred_labels, pred_scores, img_id, img_info, coco_ids)
        #
        #     # -- new dataset's version --
        #     img_id = targets[0]['image_id']
        #     h, w = targets[0]['size']
        #     img_info = {'height': h.item(), 'width': w.item()}
        #     coco_ids = test_loader.dataset.coco.getCatIds()
        #     info = (pred_bboxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

        # 4. get info for evaluation
        # evaluator.get_info(info)

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
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
    mAP = stats[0]
    mean_loss = np.array(sum_loss).mean()

    print("mAP : ", mAP)
    print("mean Loss : ", mean_loss)
    print("Eval Time : {:.4f}".format(time.time() - tic))

    # if opts.rank == 0:
    #     mAP = evaluator.evaluate(test_loader.dataset)
    #     print(mAP)

    if vis is not None:
        # loss plot
        vis.line(X=torch.ones((1, 1)).cpu() * epoch,  # step
                 Y=torch.Tensor([mAP]).unsqueeze(0).cpu(),
                 win='test_results_of_' + opts.name,
                 update='append',
                 opts=dict(xlabel='step',
                           ylabel='test',
                           title='test_loss_of' + opts.name,
                           legend=['mAP']))

    if xl_log_saver is not None:
        xl_log_saver.insert_each_epoch(contents=(epoch, mAP))

    # save best.pth.tar
    if result_best is not None:
        if result_best['mAP'] < mAP:
            print("update best model")
            result_best['epoch'] = epoch
            result_best['mAP'] = mAP
            if checkpoint is None:
                checkpoint = {'epoch': epoch,
                              'model_state_dict': model.state_dict()}
            torch.save(checkpoint, os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.best.pth.tar'))
    return


from datasets.build import build_dataloader
from models.build import build_model
from config import get_args_parser


def test_worker(rank, opts):

    # 1. config
    print(opts)

    # 2. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. visdom
    vis = None

    # 4. data(set/loader)
    _, test_loader = build_dataloader(opts)

    # 5. model
    model = build_model(opts)
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