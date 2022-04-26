import os
import sys
from PIL import Image
import torchvision.transforms.functional as F
import torch
import matplotlib.pyplot as plt

'''
this code is related to the shape of various image processing library

- PIL image 
- cv2 image
- tensor 

'''

# 0. set root
root = './figures'
image_path = os.path.join(root, '000001.jpg')

# ------------------------------,-------------#
# 1. PIL
# -------------------------------------------#

# # 1-1) make PIL image
# img_pil = Image.open(image_path)
#
# # 1-2) visualize PIL image
# img_pil.show()
# # plt.figure('input')
# # plt.imshow(img_pil)
# # plt.show()
#
# # 1-3) size of PIL image
# print(img_pil.size)
#
# # 1-4) format / dtype / range
# print(img_pil.mode)
# print(img_pil.bits)
#
# # -------------------------------------------#
# # 2. cv2
# # -------------------------------------------#
#
# import cv2
# import numpy as np
#
# # 2-1) make PIL image
# img_cv2 = cv2.imread(image_path)
# # img_cv2[100][50] = np.array([255, 255, 255], dtype=np.uint8)
#
# # 2-2) visualize PIL image
# cv2.imshow('img_cv2', img_cv2)
# cv2.waitKey(0)
#
# # 2-3) size of cv2 image
# print(img_cv2.shape)
#
# # 2-4) format / dtype / range of cv2 image
# print(img_cv2.dtype)
# # -
# print(img_cv2.min, img_cv2.max)

# -------------------------------------------#
# 3. tensor(pytorch)
# -------------------------------------------#

# 3-1) make tensor
img_pil = Image.open(image_path)
img_tensor = F.to_tensor(img_pil)

# 3-2) visualize tensor image

# 3-3) size of tensor image
print(img_tensor.size())
print(img_tensor.shape)

# # 2-4) format / dtype / range of cv2 image
# print(img_cv2.dtype)
print(img_tensor.dtype)
# print(img_cv2.min, img_cv2.max)

plt.figure('input')
plt.imshow(img_tensor.permute(1, 2, 0))
plt.show()

# question
# img_tensor[:, 50, 100] = torch.Tensor([0, 0, 0])
# plt.figure('input')
# plt.imshow(img)
# plt.show()
# # print(img_tensor.size())
# print(img.size)
# img.show()

# print(img)
#
# import cv2
# import numpy as np
# img_cv2 = cv2.imread(image_path)
# img_cv2[100][50] = np.array([255, 255, 255], dtype=np.uint8)
#
# cv2.imshow('input', img_cv2)
# cv2.waitKey(0)