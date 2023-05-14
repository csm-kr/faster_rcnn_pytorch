import os
import math
import torch
import torchvision
import transforms as T
from torch.utils.data import DataLoader
from coco_utils import ConvertCocoPolysToMask


class COCODatasetV1(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        # 117266
        self.ids = list(sorted(self.coco.imgToAnns.keys()))
        self._parses = ConvertCocoPolysToMask()
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self._parses(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        batch[0] = self.batched_tensor_from_tensor_list(batch[0])
        return batch

    def max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batched_tensor_from_tensor_list(self, images, size_divisible=32):
        # if torchvision._is_tracing():
        #     # batch_images() does not export well to ONNX
        #     # call _onnx_batch_images() instead
        #     return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        return batched_imgs


if __name__ == '__main__':
    # root, image_set, transforms
    root = "D:\data\\coco"
    image_set = "train"

    img_folder = os.path.join(root, f'{image_set}2017')
    ann_file = os.path.join(root, 'annotations', f'instances_{image_set}2017.json')
    transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = COCODatasetV1(img_folder, ann_file, transforms)

    img, target = dataset.__getitem__(0)
    print("the shape of imgs :", img.size())
    print("target['boxes'] :", target['boxes'])
    print("target keys :", target.keys())
    print("len: ", dataset.__len__())

    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler,
                                                        batch_size=2,
                                                        drop_last=True)
    data_loader = DataLoader(dataset,
                             batch_sampler=train_batch_sampler,
                             num_workers=4,
                             collate_fn=dataset.collate_fn)

    for i, (img, target) in enumerate(data_loader):
        print(img.shape)
        '''
        torch.Size([2, 3, 800, 1152])
        torch.Size([2, 3, 800, 1088])
        torch.Size([2, 3, 1056, 1216])
        torch.Size([2, 3, 800, 1088])
        '''
        if i == 4:
            break
