# Faster RCNN Pytorch 
re-implementation of faster rcnn (NIPS2015)

Please refer to https://arxiv.org/abs/1506.01497

### Dataset
- [x] VOC  
- [x] COCO

### Data Augmentation (for implementation of original paper)
- [x] Resize
- [x] Horizontal Flip

### Training Setting
- **Use approximate joint training**
```
- batch size : 1
- optimizer : SGD
- epoch : 13 
- initial learning rate 0.001
- weight decay : 5e-4
- momentum : 0.9
- scheduler : cosineannealing LR (min : 5e-5)
```

### Results

- VOC

|methods     |  Traning   |   Testing  | Resolution |   AP50          |
|------------|------------|------------|------------| --------------- |
|papers      |2007        |  2007      | **         |   69.9          |
|papers      |2007 + 2012 |  2007      | **         |   73.2          |
|this repo   |2007        |  2007      | **         |   71.00(+1.10)  |
|this repo   |2007 + 2012 |  2007      | **         |   75.81(+2.61)  |


![000001_input](https://user-images.githubusercontent.com/18729104/226520415-315a3c36-05e0-4ba7-84ee-9a5fba8d8553.jpg)
![000001_result](https://user-images.githubusercontent.com/18729104/226520424-8de893c3-58de-4ad5-8ddf-41ebfd276d8a.jpg)

![000015](https://user-images.githubusercontent.com/18729104/226520427-127c0a52-b372-4585-92b2-7319649390b9.jpg)
![000021](https://user-images.githubusercontent.com/18729104/226520433-00aede93-5c95-42bf-89b5-99ed0a248de0.jpg)

- COCO

|methods     |  Traning    |   Testing  | Resolution |   mAP@[.5 .95]  |
|------------|-------------|------------|------------| --------------- |
|papers      |train        |  COCOval   | **         |   21.2          |
|papers      |trainval     |  COCOval   | **         |   -             |
|this repo   |COCOtrain2017|  minival   | **         |   20.7(-0.50%)  |

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.207
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.199
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.352
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.209
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.055
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.354
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.510
```

** A way to resize frcnn is to make the image different size if the original image is different.

![000000000025](https://user-images.githubusercontent.com/18729104/226520439-d9f1ddba-e064-4984-9ec1-b01b3c70df8f.jpg)

![000000000036](https://user-images.githubusercontent.com/18729104/226520456-bd397e45-8e67-44a4-8812-cb7950a612ae.jpg)


### Quick Start for test

1 - download pth.tar files 

- VOC
- faster_rcnn_voc.best.pth.tar [here](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EaOuSelMyTJKin5B5C2k8D4BzXIC9Ej62CArAUXrpk9Hgg) (about 1GB)
- COCO
- faster_rcnn_coco.best.pth.tar [here](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/Efu3JLCm7RFNgGzRp-dNzYABWsFh-VrCUUCQ-rGNfbTk7A) (about 1GB)

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
test.py --config ./config_files/faster_rcnn_voc_test.txt
test.py --config ./config_files/faster_rcnn_coco_test.txt
```

### Quick Start for demo
1 - run demo.py for demo : demo at demo figures voc or coco
```
demo.py --config ./config_files/faster_rcnn_coco_demo.txt
demo.py --config ./config_files/faster_rcnn_voc_demo.txt
```

### Quick Start for train
1 - set your data root at config

config files is as follows

```
# name
name = faster_rcnn_coco

# data 
data_root = put your root in this part
data_type = coco

# training
epoch = 13
batch_size = 1

# testing
thres = 0.05
```

2 - run main.py for train 

```
main.py --config ./config_files/faster_rcnn_coco_train.txt
main.py --config ./config_files/faster_rcnn_voc_train.txt
```

### Process of faster rcnn

![Process](https://user-images.githubusercontent.com/18729104/226520463-c4c054bb-9dd6-4ce9-ac78-5826d7ba7cf3.jpg)

### Citation
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
