import os
import time
import torch
from tqdm import tqdm


def train_one_epoch(opts, epoch, device, vis, train_loader, model, criterion, optimizer, scheduler):

    tic = time.time()
    model.train()

    for idx, data in enumerate(tqdm(train_loader)):

        images = data[0]
        boxes = data[1]
        labels = data[2]

        # cuda
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # forward and loss
        pred, target = model(images, boxes, labels)   # [cls, reg] - [B, 18, H', W'], [B, 36, H', W']
        loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss = criterion(pred, target)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % opts.vis_step == 0 or idx == len(train_loader) - 1:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'RPN_Cls_loss: {rpn_cls_loss:.4f}\t'
                  'RPN_Reg_loss: {rpn_reg_loss:.4f}\t'                  
                  'Fast_RCNN_Cls_loss: {fast_rcnn_cls_loss:.4f}\t'
                  'Fast_RCNN_Reg_loss: {fast_rcnn_reg_loss:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader),
                          loss=loss,
                          rpn_cls_loss=rpn_cls_loss,
                          rpn_reg_loss=rpn_reg_loss,
                          fast_rcnn_cls_loss=fast_rcnn_cls_loss,
                          fast_rcnn_reg_loss=fast_rcnn_reg_loss,
                          lr=lr,
                          time=toc - tic))

            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 5)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss]).unsqueeze(0).cpu(),
                         win='train_loss_of_' + opts.name,
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss', 'RPN CLS', 'RPN REG', 'FCNN CLS', 'FCNN REG']))

    if opts.rank == 0:
        save_path = os.path.join(opts.log_dir, opts.name, 'saves')
        os.makedirs(save_path, exist_ok=True)

        # not saving all epochs

        # checkpoint = {'epoch': epoch,
        #               'model_state_dict': model.state_dict(),
        #               'optimizer_state_dict': optimizer.state_dict(),
        #               'scheduler_state_dict': scheduler.state_dict()}
        #
        # torch.save(checkpoint, os.path.join(save_path, opts.name + '.{}.pth.tar'.format(epoch)))