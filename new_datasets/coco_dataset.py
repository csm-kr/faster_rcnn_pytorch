import math
import torch
import torchvision
import transforms as T
from coco_utils import ConvertCocoPolysToMask


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    if not isinstance(dataset, torchvision.datasets.CocoDetection):
        raise TypeError(
            f"This function expects dataset of type torchvision.datasets.CocoDetection, instead  got {type(dataset)}"
        )
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


class COCODatasetV1(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
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

    # right transforms
    transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_folder = "D:\data\\coco\\train2017"
    ann_file = "D:\data\coco\\annotations\instances_train2017.json"
    dataset = COCODatasetV1(img_folder, ann_file, transforms)   # len:  118xxx
    dataset = _coco_remove_images_without_annotations(dataset)  # len:  117266

    img, target = dataset.__getitem__(0)
    print("the shape of imgs :", img.size())
    print("target['boxes'] :", target['boxes'])
    print("target keys :", target.keys())
    print("len: ", dataset.__len__())

    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler,
                                                        batch_size=2,
                                                        drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=train_batch_sampler,
                                              num_workers=4,
                                              collate_fn=dataset.dataset.collate_fn)

    for img, target in data_loader:
        print(img.shape)
        '''
        torch.Size([2, 3, 800, 1152])
        torch.Size([2, 3, 800, 1088])
        torch.Size([2, 3, 1056, 1216])
        torch.Size([2, 3, 800, 1088])
        '''


