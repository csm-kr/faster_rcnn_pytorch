# Faster RCNN Pytorch 

re-implementation of faster rcnn (NIPS2015)

### data set
- [x] VOC 2007 

### data augmentation (for implementation of original paper)
- [x] Resize
- [x] Horizontal Flip

### RESULTS

#### 1. qualitative result

Please refer to https://arxiv.org/abs/1506.01497

VOC

|methods     |  Traning   |   Testing  | Resolution |   AP50    |
|------------|------------|------------|------------| --------- |
|papers      |2007        |  2007      | **1*       |   69.9    |
|papers      |2007 + 2012 |  2007      | **1*       |   73.2    |
|experiments |2007 + 2012 |  2007      | -          |      _    |


**1* A way to resize frcnn is to make the image different size if the original image is different.

#### 2. quantitative result

