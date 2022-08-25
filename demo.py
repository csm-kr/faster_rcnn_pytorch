import os
import cv2
import time
import glob
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import propose_region
from evaluation.evaluator import Evaluator

from dataset.build import det_transforms
from dataset.detection_transforms import FRCNNResizeOnlyImage
from torchvision import transforms as tfs

from PIL import Image


def demo_image_transforms(demo_image):

    transform_demo = tfs.Compose([tfs.ToTensor(),
                                  # FRCNNResizeOnlyImage(size=600, max_size=1000),
                                  tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

    demo_image = transform_demo(demo_image)
    demo_image = demo_image.unsqueeze(0)  # make batch
    return demo_image


@ torch.no_grad()
def demo(epoch, device, model, opts):

    # 1. make tensors
    demo_image_list = glob.glob(os.path.join(opts.demo_root, '*' + '.' + opts.demo_image_type))
    total_time = 0

    # 2. load .pth
    checkpoint = torch.load(f=os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.
                                           format(opts.demo_epoch)),
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for idx, img_path in enumerate(demo_image_list):

        # --------------------- img load ---------------------
        demo_image = Image.open(img_path).convert('RGB')
        demo_image = demo_image_transforms(demo_image).to(device)

        tic = time.time()
        _, _, _, im_show = model.predict(demo_image, opts)

        # save_files
        demo_result_path = os.path.join(opts.demo_root, 'detection_results')
        os.makedirs(demo_result_path, exist_ok=True)

        # 0 ~ 1 image -> 0~255 image
        im_show = cv2.convertScaleAbs(im_show, alpha=(255.0))
        cv2.imwrite(os.path.join(demo_result_path, os.path.basename(img_path)), im_show)

        # cv2.imshow('i', im_show)
        # cv2.waitKey(0)

        toc = time.time()
        inference_time = toc - tic
        total_time += inference_time

        if idx % 100 == 0 or idx == len(demo_image_list) - 1:
            # ------------------- check fps -------------------
            print('Step: [{}/{}]'.format(idx, len(demo_image_list)))
            print("fps : {:.4f}".format((idx + 1) / total_time))

        # if opts.data_type == 'voc':
        #     label_map = voc_label_map
        #     color_array = voc_color_array

    tic = time.time()


import argparse
from model import FRCNN
from loss import FRCNNLoss
from dataset.build import build_dataset
from config import get_args_parser


def demo_worker(rank, opts):

    # 1. config
    print(opts)

    # 2. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 5. model
    if opts.data_type == 'voc':
        opts.num_classes = 21
    if opts.data_type == 'coco':
        opts.num_classes = 81

    model = FRCNN(num_classes=opts.num_classes)
    model = model.to(device)

    demo(epoch=opts.demo_epoch,
         device=device,
         model=model,
         opts=opts)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('faster rcnn demo', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    print(opts)
    demo_worker(0, opts)