from torch.utils.data import DataLoader
from datasets.voc_dataset import VOC_Dataset
from datasets.coco_dataset import COCO_Dataset
from torch.utils.data.distributed import DistributedSampler
import datasets.transforms_ as T


def build_dataloader(opts):

    size = (opts.resize, opts.resize)
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_train = T.Compose([
        T.RandomPhotoDistortion(),
        T.RandomHorizontalFlip(),
        T.RandomZoomOut(max_scale=2),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ]),
        ),
        # T.Resize(size),
        normalize
    ])

    transform_test = T.Compose([
        T.RandomResize([800], max_size=1333),
        # FIXME add resize for fixed size image
        # T.Resize(size),
        normalize
    ])

    # size = (opts.resize, opts.resize)
    #
    # transform_train = det_transforms.DetCompose([
    #     # ------------- for Tensor augmentation -------------
    #     det_transforms.DetRandomPhotoDistortion(),
    #     det_transforms.DetRandomHorizontalFlip(),
    #     det_transforms.DetToTensor(),
    #     # ------------- for Tensor augmentation -------------
    #     det_transforms.DetRandomZoomOut(max_scale=3),
    #     det_transforms.DetRandomZoomIn(),
    #     det_transforms.DetResize(size=size, box_normalization=True),
    #     det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    # ])
    # transform_test = det_transforms.DetCompose([
    #     det_transforms.DetToTensor(),
    #     det_transforms.DetResize(size=size, box_normalization=True),
    #     det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    # ])

    train_loader = None
    test_loader = None

    if opts.data_type == 'voc':
        train_set = VOC_Dataset(opts.data_root,
                                split='train',
                                resize=opts.resize,
                                download=True,
                                transform=transform_train,
                                mosaic_transform=opts.mosaic_transform,
                                visualization=False)

        test_set = VOC_Dataset(opts.data_root,
                               split='test',
                               resize=opts.resize,
                               download=True,
                               transform=transform_test,
                               visualization=False)

        train_loader = DataLoader(train_set,
                                  batch_size=opts.batch_size,
                                  collate_fn=train_set.collate_fn,
                                  shuffle=True,
                                  num_workers=opts.num_workers,
                                  pin_memory=True)

        if opts.distributed:
            train_loader = DataLoader(train_set,
                                      batch_size=int(opts.batch_size / opts.world_size),
                                      collate_fn=train_set.collate_fn,
                                      shuffle=False,
                                      num_workers=int(opts.num_workers / opts.world_size),
                                      pin_memory=True,
                                      sampler=DistributedSampler(dataset=train_set),
                                      drop_last=False)

        test_loader = DataLoader(test_set,
                                 batch_size=1,
                                 collate_fn=test_set.collate_fn,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)
        opts.num_classes = 21

    elif opts.data_type == 'coco':

        train_set = COCO_Dataset(root=opts.data_root,
                                 split='train',
                                 resize=opts.resize,
                                 download=True,
                                 transform=transform_train,
                                 mosaic_transform=opts.mosaic_transform,
                                 visualization=False)

        test_set = COCO_Dataset(root=opts.data_root,
                                split='val',
                                resize=opts.resize,
                                download=True,
                                transform=transform_test,
                                visualization=False)

        train_loader = DataLoader(train_set,
                                  batch_size=opts.batch_size,
                                  collate_fn=train_set.collate_fn,
                                  shuffle=True,
                                  num_workers=opts.num_workers,
                                  pin_memory=True)

        if opts.distributed:
            train_loader = DataLoader(train_set,
                                      batch_size=int(opts.batch_size / opts.world_size),
                                      collate_fn=train_set.collate_fn,
                                      shuffle=False,
                                      num_workers=int(opts.num_workers / opts.world_size),
                                      pin_memory=True,
                                      sampler=DistributedSampler(dataset=train_set),
                                      drop_last=False)

        test_loader = DataLoader(test_set,
                                 batch_size=1,
                                 collate_fn=test_set.collate_fn,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)
        opts.num_classes = 81

    return train_loader, test_loader



