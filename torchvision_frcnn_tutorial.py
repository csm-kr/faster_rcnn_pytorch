import cv2
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as T


def visualize_detection_result(img_pil, boxes, labels, scores):
    """
    img_pil : pil image range - [0 255], uint8
    boxes : torch.Tensor, [num_obj, 4], torch.float32
    labels : torch.Tensor, [num_obj] torch.int64
    scores : torch.Tensor, [num_obj] torch.float32
    """

    # 1. uint8 -> float32
    image_np = np.array(img_pil).astype(np.float32) / 255.
    x_img = image_np
    im_show = cv2.cvtColor(x_img, cv2.COLOR_RGB2BGR)

    for j in range(len(boxes)):

        label_list = list(coco_labels_map.keys())
        color_array = coco_colors_array

        x_min = int(boxes[j][0])
        y_min = int(boxes[j][1])
        x_max = int(boxes[j][2])
        y_max = int(boxes[j][3])

        cv2.rectangle(im_show,
                      pt1=(x_min, y_min),
                      pt2=(x_max, y_max),
                      color=color_array[labels[j]],
                      thickness=2)

        # text_size
        text_size = cv2.getTextSize(text=label_list[labels[j]] + ' {:.2f}'.format(scores[j].item()),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    thickness=1)[0]

        # text_rec
        cv2.rectangle(im_show,
                      pt1=(x_min, y_min),
                      pt2=(x_min + text_size[0] + 3, y_min + text_size[1] + 4),
                      color=color_array[labels[j]],
                      thickness=-1)

        # put text
        cv2.putText(im_show,
                    text=label_list[labels[j]] + ' {:.2f}'.format(scores[j].item()),
                    org=(x_min + 10, y_min + 10),  # must be int
                    fontFace=0,
                    fontScale=0.4,
                    color=(0, 0, 0))

    # cv2.imshow(...) : float values in the range [0, 1]
    cv2.imshow('result', im_show)
    cv2.waitKey(0)

    # cv2.imwrite(...) : int values in the range [0, 255]
    # im_show = im_show * 255
    # cv2.imwrite("result.png", im_show)
    return 0


def demo(img_path, threshold):
    """
    demo faster rcnn
    :param img_path: image path (default - soccer.png)
    :param threshold: the threshold of object detection score (default - 0.9)
    :return: None
    """

    # 1. load image
    img_pil = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    img = transform(img_pil)
    batch_img = [img]

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    pred = model(batch_img)

    # 2. remove first batch
    pred_dict = pred[0]
    '''
    pred_dict 
    {'boxes' : tensor,
     'labels' : tensor,
     'scores' : tensor}
    '''

    # 3. get pred boxes and labels, scores
    pred_boxes = pred_dict['boxes']    # [N, 1]
    pred_labels = pred_dict['labels']  # [N]
    pred_scores = pred_dict['scores']  # [N]

    # 4. Get pred according to threshold
    indices = pred_scores >= threshold
    pred_boxes = pred_boxes[indices]
    pred_labels = pred_labels[indices]
    pred_scores = pred_scores[indices]

    # 5. visualize
    visualize_detection_result(img_pil, pred_boxes, pred_labels, pred_scores)


if __name__ == '__main__':

    coco_labels_list = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    coco_labels_map = {k: v for v, k in enumerate(coco_labels_list)}
    np.random.seed(1)
    coco_colors_array = np.random.randint(256, size=(91, 3)) / 255

    # demo
    demo('./soccer.png', threshold=0.9)