import os
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import propose_region



def test(epoch, device, vis, test_loader, model, criterion, optimizer, scheduler, opts):

    # 1. load .pth
    checkpoint = torch.load(f=os.path.join(opts['save_path'], opts['save_file_name'] + '.{}.pth.tar'.format(epoch)),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tic = time.time()
    sum_loss = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):

            images = data[0]
            boxes = data[1]
            labels = data[2]

            # 2. load data to device
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # 3. forward
            height, width = images.size()[2:]  # height, width
            size = (height, width)
            pred = model(images)
            # coder = criterion.coder
            loss, cls_loss, reg_loss = criterion(pred, boxes, labels, size)

            sum_loss += loss.item()

            # ---------- eval ----------
            pred_boxes = propose_region(pred=pred,
                                        coder=criterion.coder)
            pred_boxes = pred_boxes.cpu()

            # ---------------------------- visualization ----------------------------
            # 0. permute
            images = images.cpu()
            images = images.squeeze(0).permute(1, 2, 0)  # B, C, H, W --> H, W, C

            # 1. un normalization
            images *= torch.Tensor([0.229, 0.224, 0.225])
            images += torch.Tensor([0.485, 0.456, 0.406])

            # 2. RGB to BGR
            image_np = images.numpy()

            # 3. box scaling
            size_ = torch.FloatTensor([size[1], size[0], size[1], size[0]])
            pred_boxes *= size_
            bbox = pred_boxes
            plt.figure('result')
            plt.imshow(image_np)

            for i in range(len(bbox)):
                x1 = bbox[i][0]
                y1 = bbox[i][1]
                x2 = bbox[i][2]
                y2 = bbox[i][3]

                # class and score
                # plt.text(x=x1 - 5,
                #          y=y1 - 5,
                #          fontsize=10,
                #          bbox=dict(facecolor=[0, 0, 1],
                #                    alpha=0.5))

                # bounding box
                plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                              width=x2 - x1,
                                              height=y2 - y1,
                                              linewidth=1,
                                              edgecolor=[0, 0, 1],
                                              facecolor='none'))
            plt.show()


if __name__ == '__main__':
    from config import load_arguments
    from dataset.build import build_dataset
    from model.build import build_model
    from coder import FasterRCNN_Coder
    from loss.build import build_loss

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

    # 3. data(set/loader)
    train_loader, test_loader = build_dataset(data_config)

    # 4. model
    model = build_model(model_config)
    model = model.to(device)

    # 5. loss
    coder = FasterRCNN_Coder()
    criterion = build_loss(model_config, coder)

    test(epoch=30,
         device=device,
         vis=None,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         optimizer=None,
         scheduler=None,
         opts=train_config)
