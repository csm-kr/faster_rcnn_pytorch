from config import load_arguments
import torch
import visdom
from dataset.build import build_dataset
from coder import FasterRCNN_Coder
from model.build import build_model
from loss.build import build_loss


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

    # 6. loss
    # 6 - 1 ) coder
    coder = FasterRCNN_Coder()
    loss = build_loss(model_config)


if __name__ == '__main__':
    main_worker()
