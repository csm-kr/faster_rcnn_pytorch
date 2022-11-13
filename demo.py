import os
import cv2
import time
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tfs


def demo_image_transforms(demo_image):

    transform_demo = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

    demo_image = transform_demo(demo_image)
    demo_image = demo_image.unsqueeze(0)  # make batch
    return demo_image


@ torch.no_grad()
def demo(opts, device, model):

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
        demo_image_pil = Image.open(img_path).convert('RGB')
        demo_image = demo_image_transforms(demo_image_pil).to(device)

        tic = time.time()
        pred_boxes, pred_labels, pred_scores = model.predict(demo_image, opts)

        im_show = visualize_detection_result(demo_image_pil, pred_boxes, pred_labels, pred_scores)

        # save_files
        demo_result_path = os.path.join(opts.demo_root, 'detection_results')
        os.makedirs(demo_result_path, exist_ok=True)

        # 0 ~ 1 image -> 0~255 image
        im_show = cv2.convertScaleAbs(im_show, alpha=(255.0))
        cv2.imwrite(os.path.join(demo_result_path, os.path.basename(img_path)), im_show)

        if opts.demo_vis:
            cv2.imshow('demo_results', im_show)
            cv2.waitKey(0)

        toc = time.time()
        inference_time = toc - tic
        total_time += inference_time

        if idx % 100 == 0 or idx == len(demo_image_list) - 1:
            # ------------------- check fps -------------------
            print('Step: [{}/{}]'.format(idx, len(demo_image_list)))
            print("fps : {:.4f}".format((idx + 1) / total_time))

    print("complete detection...!")
    return


def visualize_detection_result(x, bbox, label, score):

    '''
    x : pil image range - [0 255], uint8
    bbox : np.array, [num_obj, 4], float32
    label : np.array, [num_obj] int32
    score : np.array, [num_obj] float32
    '''

    img_width, img_height = x.size
    multiplier = np.array([img_width, img_height, img_width, img_height])
    bbox *= multiplier

    # 2. uint8 -> float32
    image_np = np.array(x).astype(np.float32) / 255.
    x_img = image_np
    im_show = cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR)

    for j in range(len(bbox)):

        if opts.data_type == 'voc':
            from utils import voc_color_array, voc_label_map
            label_list = list(voc_label_map.keys())
            color_array = voc_color_array

        elif opts.data_type == 'coco':
            from utils import coco_color_array, coco_label_map
            label_list = list(coco_label_map.keys())
            color_array = coco_color_array

        x_min = int(bbox[j][0])
        y_min = int(bbox[j][1])
        x_max = int(bbox[j][2])
        y_max = int(bbox[j][3])

        cv2.rectangle(im_show,
                      pt1=(x_min, y_min),
                      pt2=(x_max, y_max),
                      color=color_array[label[j]],
                      thickness=2)

        # text_size
        text_size = cv2.getTextSize(text=label_list[label[j]] + ' {:.2f}'.format(score[j].item()),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    thickness=1)[0]

        # text_rec
        cv2.rectangle(im_show,
                      pt1=(x_min, y_min),
                      pt2=(x_min + text_size[0] + 3, y_min + text_size[1] + 4),
                      color=color_array[label[j]],
                      thickness=-1)

        # put text
        cv2.putText(im_show,
                    text=label_list[label[j]] + ' {:.2f}'.format(score[j].item()),
                    org=(x_min + 10, y_min + 10),  # must be int
                    fontFace=0,
                    fontScale=0.4,
                    color=(0, 0, 0))

    return im_show

import configargparse
from models.build import build_model
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

    model = build_model(opts)
    model = model.to(device)

    demo(opts=opts,
         device=device,
         model=model,
         )


if __name__ == '__main__':

    parser = configargparse.ArgumentParser('faster rcnn demo', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    print(opts)
    demo_worker(0, opts)