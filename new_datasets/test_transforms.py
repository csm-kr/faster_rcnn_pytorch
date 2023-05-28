import os
from coco_dataset import COCODatasetV1
import new_datasets.transforms as T

import matplotlib.pyplot as plt
import numpy as np


# def visualize_image():
#     # ----------------- visualization -----------------
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#
#     # tensor to img
#     img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
#     img_vis *= std
#     img_vis += mean
#     img_vis = np.clip(img_vis, 0, 1)
#
#     plt.figure('input')
#     plt.imshow(img_vis)
#     print('num objects : {}'.format(len(boxes)))
#
#     for i in range(len(boxes)):
#         new_h_scale = new_w_scale = 1
#         # box_normalization of DetResize
#         # if self.transform.transforms[-2].box_normalization:
#         new_h_scale, new_w_scale = image.size()[1:]
#
#         x1 = boxes[i][0] * new_w_scale
#         y1 = boxes[i][1] * new_h_scale
#         x2 = boxes[i][2] * new_w_scale
#         y2 = boxes[i][3] * new_h_scale
#
#         # print(boxes[i], ':', self.coco_ids_to_class_names[self.coco_ids[labels[i]]])
#
#         # class
#         plt.text(x=x1 - 5,
#                  y=y1 - 5,
#                  s=str(self.coco_ids_to_class_names[self.coco_ids[labels[i]]]),
#                  bbox=dict(boxstyle='round4',
#                            facecolor=coco_color_array[labels[i]],
#                            alpha=0.9))
#
#         # bounding box
#         plt.gca().add_patch(Rectangle(xy=(x1, y1),
#                                       width=x2 - x1,
#                                       height=y2 - y1,
#                                       linewidth=1,
#                                       edgecolor=coco_color_array[labels[i]],
#                                       facecolor='none'))
#
#     plt.show()
#
#     return image, boxes, labels


if __name__ == '__main__':
    root = os.path.join(os.path.expanduser('~'), 'Desktop', 'data', 'coco')
    # root = 'D:\data\coco'
    image_set = 'val'
    img_folder = os.path.join(root, f'{image_set}2017')
    ann_file = os.path.join(root, 'annotations', f'instances_{image_set}2017.json')
    transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize(size=800, max_size=1333),
        # T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = COCODatasetV1(img_folder, ann_file, transforms, visualization=True)
    for i, (img, target) in enumerate(dataset):
        print(img.shape)
        print(target)
        if i == 10:
            break
