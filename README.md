# Faster RCNN Pytorch 

re-implementation of faster rcnn (NIPS2015)

Please refer to https://arxiv.org/abs/1506.01497

### data set
- [x] VOC  
- [x] COCO

### data augmentation (for implementation of original paper)
- [x] Resize
- [x] Horizontal Flip

### TRAINING

- **Use approximate joint training**
- optimizer : SGD
- epoch : 15 
- initial learning rate 0.001
- weight decay : 5e-4
- momentum : 0.9

### RESULTS

#### 1. qualitative result

VOC

|methods     |  Traning   |   Testing  | Resolution |   AP50          |
|------------|------------|------------|------------| --------------- |
|papers      |2007        |  2007      | **         |   69.9          |
|papers      |2007 + 2012 |  2007      | **         |   73.2          |
|this repo   |2007        |  2007      | **         |   71.00(+1.10)  |
|this repo   |2007 + 2012 |  2007      | **         |   75.81(+2.61)  |

COCO

|methods     |  Traning    |   Testing  | Resolution |   mAP@[.5 .95]  |
|------------|-------------|------------|------------| --------------- |
|papers      |train        |  COCOval   | **         |   21.2          |
|papers      |trainval     |  COCOval   | **         |   -             |
|this repo   |COCOtrain2017|  minival   | **         |   -             |

** A way to resize frcnn is to make the image different size if the original image is different.

#### 2. quantitative result

VOC

![000001_input](./figures/000001_.jpg)
![000001_result](./figures/000001.jpg)

![000015](./figures/000010.jpg)
![000021](./figures/000021.jpg)


### Quick Start for test

1 - download pth.tar files 

- VOC
- faster_rcnn_voc.best.pth.tar [here](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EaOuSelMyTJKin5B5C2k8D4BzXIC9Ej62CArAUXrpk9Hgg) (about 1GB)
- COCO
- faster_rcnn_cococ.best.pth.tar here

2 - put tar file in like this (in saves)

```
dataset
evaluation
figures
logs
    |-- faster_rcnn_voc
        |-- saves
            |-- faster_rcnn_voc.best.pth.tar    
    |-- faster_rcnn_coco
        |-- saves
            |-- faster_rcnn_coco.best.pth.tar
anchor.py
...
main.py
...
utils.py
```

3 - set root and run test.py
```
test.py --name faster_rcnn_voc --root your own root --data_type voc
test.py --name faster_rcnn_coco --root your own root --data_type coco
```


### Process of faster rcnn

![Process](./figures/faster_rcnn_process.jpg)

### citation
If you found this implementation and pretrained model helpful, please consider citation
```
@misc{csm-kr_Faster_RCNN_Pytorch,
  author={Sungmin, Cho},
  publisher = {GitHub},
  title={Faster_RCNN_Pytorch},
  url={https://github.com/csm-kr/Faster_RCNN_Pytorch//},
  year={2022},
}
```
