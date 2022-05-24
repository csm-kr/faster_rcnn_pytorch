import torch
import visdom
from config import load_arguments

# dataset / model / loss
from dataset.build import build_dataset
from model.faster_rcnn import FRCNN
from test_model import FasterRCNNVGG16
from loss.faster_rcnn_loss import FRCNNLoss
from torch.optim.lr_scheduler import StepLR

# train and test
from train import train_one_epoch
from test import test_and_eval


def main_worker():
    # 1. config
    yaml_file = './yaml/faster_rcnn_config.yaml'
    config = load_arguments(yaml_file)

    # configuration with yaml
    train_config = config['train']
    data_config = config['data']
    model_config = config['model']

    # 2. device
    device_ids = train_config['device']
    device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = visdom.Visdom(port=train_config['port'])

    # 4. data(set/loader)
    train_loader, test_loader = build_dataset(data_config)

    # 5. model
    # model = FRCNN()
    model = FasterRCNNVGG16()
    model = model.to(device)

    # 6. loss
    criterion = FRCNNLoss()

    # 7. optimizer
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': 0.001 * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': 0.001, 'weight_decay': 0.0005}]
    optimizer = torch.optim.SGD(params=params,
                                momentum=0.9)

    # optimizer = torch.optim.SGD(params=model.parameters(),
    #                             lr=0.001,
    #                             momentum=0.9,
    #                             weight_decay=0.0005)

    # 8. scheduler
    scheduler = StepLR(optimizer=optimizer, step_size=8, gamma=0.1)   # 9

    for epoch in range(train_config['start_epoch'], train_config['epoch']):

        # 9. train one epoch
        train_one_epoch(epoch=epoch,
                        device=device,
                        vis=vis,
                        train_loader=train_loader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        opts=train_config)

        # 10. test and evaluation
        test_and_eval(epoch=epoch,
                      device=device,
                      vis=vis,
                      test_loader=test_loader,
                      model=model,
                      criterion=criterion,
                      opts=train_config,
                      visualization=False)

        scheduler.step()


if __name__ == '__main__':
    main_worker()
