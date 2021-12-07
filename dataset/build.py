from torch.utils.data import DataLoader
from dataset.voc_dataset import VOC_Dataset
from dataset.coco_dataset import COCO_Dataset
import dataset.detection_transforms as det_transforms


def build_dataset(data_config):

    if len(data_config['size']) == 2 and data_config['size'][0] != data_config['size'][1]:
        size = min(data_config['size'])
        max_size = max(data_config['size'])
    else:
        max_size = None
        if len(data_config['size']) == 1:
            size = (data_config['size'], data_config['size'])
        else:
            size = (data_config['size'][0], data_config['size'][1])

    transform_train = det_transforms.DetCompose([
        # ------------- for Tensor augmentation -------------
        det_transforms.DetRandomPhotoDistortion(),
        det_transforms.DetRandomHorizontalFlip(),
        det_transforms.DetToTensor(),
        # ------------- for Tensor augmentation -------------
        det_transforms.DetRandomZoomOut(max_scale=3),
        det_transforms.DetRandomZoomIn(),
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

    if data_config['data_type'] == 'voc':
        train_set = VOC_Dataset(data_config['root'],
                                split='train',
                                download=True,
                                transform=transform_train,
                                visualization=data_config['visualization'])

        test_set = VOC_Dataset(data_config['root'],
                               split='test',
                               download=True,
                               transform=transform_test,
                               visualization=data_config['visualization'])

        train_loader = DataLoader(train_set,
                                  batch_size=data_config['batch_size'],
                                  collate_fn=train_set.collate_fn,
                                  shuffle=True,
                                  num_workers=data_config['num_workers'],
                                  pin_memory=True)

        test_loader = DataLoader(train_set,
                                 batch_size=1,
                                 collate_fn=test_set.collate_fn,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

        return train_loader, test_loader



