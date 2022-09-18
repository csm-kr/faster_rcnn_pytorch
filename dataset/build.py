from torch.utils.data import DataLoader
from dataset.voc_dataset import VOC_Dataset
from dataset.coco_dataset import COCO_Dataset
import dataset.detection_transforms as det_transforms


def build_dataset(opts):

    size = 600
    max_size = 1000

    size = 800
    max_size = 1333

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # origin
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

    transform_train = det_transforms.DetCompose([
        # ------------- for Tensor augmentation -------------
        # det_transforms.DetRandomPhotoDistortion(),
        det_transforms.DetRandomHorizontalFlip(),
        det_transforms.DetRandomSelect(
            det_transforms.DetCompose([
                det_transforms.DetToTensor(),
                det_transforms.DetRandomZoomIn(),
                det_transforms.DetRandomResize([400, 500, 600])]),
            det_transforms.DetCompose([
                det_transforms.DetToTensor(),
                det_transforms.DetRandomResize(scales)])
        ),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])
    transform_test = det_transforms.DetCompose([
        det_transforms.DetToTensor(),
        det_transforms.DetResize(size=size, max_size=max_size, box_normalization=True),
        det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    # transform_train = make_coco_transforms('train')
    # transform_test = make_coco_transforms('val')

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

import dataset.transforms as T

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')