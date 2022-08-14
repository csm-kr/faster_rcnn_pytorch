import os
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import propose_region
from evaluation.evaluator import Evaluator


@ torch.no_grad()
def test_and_eval(epoch, device, vis, test_loader, model, xl_log_saver, opts, visualization=False):

    # 0. evaluator
    evaluator = Evaluator(data_type='voc')

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
        info = data[3][0]  # [{}]

        # 2. load data to device
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # 3. forward(predict)
        pred_bboxes, pred_labels, pred_scores = model.predict(images, visualization)
        eval_info = (pred_bboxes, pred_labels, pred_scores, info['name'], info['original_wh'])

        # 4. get info for evaluation
        evaluator.get_info(eval_info)

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

    xl_log_saver.insert_each_epoch(contents=(epoch, mAP))