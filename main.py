from config import load_arguments
import torch
import visdom
from dataset.build import build_dataset
from coder import FasterRCNN_Coder
from model.build import build_model
from loss.build import build_loss
from train import train
from torch.optim.lr_scheduler import MultiStepLR


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
    model = build_model(model_config)
    model = model.to(device)

    # 6. loss
    coder = FasterRCNN_Coder()
    criterion = build_loss(model_config, coder)

    # 7. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=train_config['lr'],
                                momentum=0.9,
                                weight_decay=train_config['weight_decay'])

    # 8. scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[22], gamma=0.1)   # 8, 11

    # for statement
    for epoch in range(train_config['start_epoch'], train_config['epoch']):

        # 9. training
        train(epoch=epoch,
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


if __name__ == '__main__':
    main_worker()
