from torch.utils.data import DataLoader
from new_datasets.coco_dataset import COCODatasetV1, _coco_remove_images_without_annotations
from torch.utils.data.distributed import DistributedSampler
import new_datasets.transforms as T


def build_dataloader(opts):

    transforms_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms_test = T.Compose([
        T.RandomResize([800], max_size=1333),
        # FIXME add resize for fixed size image
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = None
    test_loader = None

    if opts.data_type == 'coco':

        train_set = COCODatasetV1()
        train_set = _coco_remove_images_without_annotations(train_set)  # len:  117266

        test_set = COCODatasetV1()
        train_loader = DataLoader(train_set,
                                  batch_size=opts.batch_size,
                                  collate_fn=train_set.dataset.collate_fn,
                                  shuffle=True,
                                  num_workers=opts.num_workers,
                                  pin_memory=True)

        if opts.distributed:
            train_loader = DataLoader(train_set,
                                      batch_size=int(opts.batch_size / opts.world_size),
                                      collate_fn=train_set.dataset.collate_fn,
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
