from torch.utils.data import DataLoader
from new_datasets.coco_dataset import COCODatasetV1
from torch.utils.data.distributed import DistributedSampler
import new_datasets.transforms as T
import os
import torch


def build_dataloader(opts):

    root = opts.data_root
    image_set = 'train'
    img_folder = os.path.join(root, f'{image_set}2017')
    ann_file = os.path.join(root, 'annotations', f'instances_{image_set}2017.json')

    image_set_ = 'val'
    img_folder_ = os.path.join(root, f'{image_set_}2017')
    ann_file_ = os.path.join(root, 'annotations', f'instances_{image_set_}2017.json')

    transforms_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize(size=800, max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # dataset = COCODatasetV1(img_folder, ann_file, transforms_train, visualization=True)

    transforms_test = T.Compose([
        T.Resize(size=800, max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = None
    test_loader = None

    if opts.data_type == 'coco':

        train_set = COCODatasetV1(img_folder, ann_file, transforms_train, visualization=False)
        test_set = COCODatasetV1(img_folder_, ann_file_, transforms_test, visualization=False)

        # for i, (img, target) in enumerate(train_set):
        #     if i == 0:
        #         print(target)
        #     else:
        #         break

        # train_sampler = torch.utils.data.RandomSampler(train_set)
        train_sampler = torch.utils.data.SequentialSampler(train_set)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, opts.batch_size, drop_last=True)
        train_loader = DataLoader(dataset=train_set,
                                  batch_sampler=train_batch_sampler,
                                  collate_fn=train_set.collate_fn,
                                  num_workers=opts.num_workers,
                                  pin_memory=True)

        # train_loader = DataLoader(train_set,
        #                           batch_size=opts.batch_size,
        #                           collate_fn=train_set.collate_fn,
        #                           # shuffle=True,
        #                           num_workers=opts.num_workers,
        #                           pin_memory=True)

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
                                     num_workers=int(opts.num_workers / opts.world_size),
                                     pin_memory=True)

        opts.num_classes = 91

    return train_loader, test_loader
