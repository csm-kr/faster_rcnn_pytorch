from torch.utils.data import DataLoader
from dataset.voc_dataset import VOC_Dataset
from dataset.coco_dataset import COCO_Dataset
import dataset.detection_transforms as det_transforms


def build_dataset(opts):

    size = 600
    max_size = 1000

    transform_train = det_transforms.DetCompose([
        # ------------- for Tensor augmentation -------------
        # det_transforms.DetRandomPhotoDistortion(),
        det_transforms.DetRandomHorizontalFlip(),
        det_transforms.DetToTensor(),
        # ------------- for Tensor augmentation -------------
        # det_transforms.DetRandomZoomOut(max_scale=3),
        # det_transforms.DetRandomZoomIn(),
        det_transforms.DetResize(size=size, max_size=max_size, box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    transform_test = det_transforms.DetCompose([
        det_transforms.DetToTensor(),
        det_transforms.DetResize(size=size, max_size=max_size, box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    if opts.data_type == 'voc':
        train_set = VOC_Dataset(opts.root,
                                split='train',
                                download=True,
                                transform=transform_train,
                                visualization=False)

        test_set = VOC_Dataset(opts.root,
                               split='test',
                               download=True,
                               transform=transform_test,
                               visualization=False)

        # train_loader = DataLoader(train_set,
        #                           batch_size=opts.batch_size,
        #                           collate_fn=train_set.collate_fn,
        #                           shuffle=False,
        #                           num_workers=opts.num_workers,
        #                           pin_memory=True)

        train_loader = DataLoader(train_set,
                                  batch_size=opts.batch_size,
                                  collate_fn=train_set.collate_fn,
                                  shuffle=True,
                                  num_workers=opts.num_workers,
                                  pin_memory=True)

        test_loader = DataLoader(test_set,
                                 batch_size=1,
                                 collate_fn=test_set.collate_fn,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)
        opts.num_classes = 21

    elif opts.data_type == 'coco':

        train_set = COCO_Dataset(root=opts.root,
                                 split='train',
                                 download=True,
                                 transform=transform_train,
                                 visualization=False)

        test_set = COCO_Dataset(root=opts.root,
                                split='val',
                                download=True,
                                transform=transform_test,
                                visualization=False)

        train_loader = DataLoader(train_set,
                                  batch_size=opts.batch_size,
                                  collate_fn=train_set.collate_fn,
                                  shuffle=True,
                                  num_workers=opts.num_workers,
                                  pin_memory=True)

        # train_loader = DataLoader(train_set,
        #                           batch_size=opts.batch_size,
        #                           collate_fn=train_set.collate_fn,
        #                           shuffle=False,
        #                           num_workers=0,
        #                           pin_memory=False)

        test_loader = DataLoader(test_set,
                                 batch_size=1,
                                 collate_fn=test_set.collate_fn,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)
        opts.num_classes = 81

    return train_loader, test_loader



