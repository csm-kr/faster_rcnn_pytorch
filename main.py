import torch
import visdom
from config import load_arguments

# dataset / model / loss
from dataset.build import build_dataset
from model.faster_rcnn import FRCNN
from loss.faster_rcnn_loss import FRCNNLoss
from torch.optim.lr_scheduler import MultiStepLR

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
    model = FRCNN()
    model = model.to(device)

    # 6. loss
    criterion = FRCNNLoss()

    # 7. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=train_config['lr'],
                                momentum=0.9,
                                weight_decay=train_config['weight_decay'])

    # 8. scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[10], gamma=0.1)   # 8, 11

    for epoch in range(train_config['start_epoch'], train_config['epoch']):

        # 9. training
        train_one_epoch(epoch=epoch,
                        device=device,
                        vis=vis,
                        train_loader=train_loader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        opts=train_config)

        scheduler.step()

        # 10. test/evaluation
        test_and_eval(epoch=epoch,
                      device=device,
                      vis=vis,
                      test_loader=test_loader,
                      model=model,
                      criterion=criterion,
                      opts=train_config)


if __name__ == '__main__':
    main_worker()
