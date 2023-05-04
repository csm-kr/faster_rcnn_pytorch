import time
import torch
import numpy as np
from utils.util import cxcy_to_xy, xy_to_cxcy


class FRCNNAnchorMaker(object):

    def __init__(self, base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        self.base_size = base_size
        self.ratios = ratios
        self.anchor_scales = anchor_scales
        self.anchor_base = self.generate_anchor_base()

    def generate_anchor_base(self):

        px = self.base_size / 2.
        py = self.base_size / 2.

        anchor_base = np.zeros((len(self.ratios) * len(self.anchor_scales), 4), dtype=np.float32)
        for i in range(len(self.ratios)):
            for j in range(len(self.anchor_scales)):
                w = self.base_size * self.anchor_scales[j] * np.sqrt(self.ratios[i])
                h = self.base_size * self.anchor_scales[j] * np.sqrt(1. / self.ratios[i])

                index = i * len(self.anchor_scales) + j
                anchor_base[index, 0] = px - w / 2.
                anchor_base[index, 1] = py - h / 2.
                anchor_base[index, 2] = px + w / 2.
                anchor_base[index, 3] = py + h / 2.

        return anchor_base

    def _enumerate_shifted_anchor(self,
                                  origin_image_size):

        origin_height, origin_width = origin_image_size
        width = origin_width // self.base_size
        height = origin_height // self.base_size
        feat_stride = self.base_size

        shift_x = np.arange(0, width * feat_stride, feat_stride)
        shift_y = np.arange(0, height * feat_stride, feat_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)

        A = self.anchor_base.shape[0]
        K = shift.shape[0]
        anchor = self.anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
        anchor = anchor.reshape((K * A, 4)).astype(np.float32)

        divisor = np.array([origin_width, origin_height, origin_width, origin_height])
        anchor /= divisor

        return anchor


class FasterRCNN_Anchor(object):
    def create_anchors(self, image_size, num_pooling=4):
        print('make retina anchors')

        # get height and width of features
        image_height, image_width = size = image_size  # h, w
        feature_height, feature_width = image_height, image_width

        for i in range(num_pooling):
            feature_height = feature_height // 2
            feature_width = feature_width // 2

        areas = [128, 256, 512]
        aspect_ratios = np.array([0.5, 1.0, 2.0])
        center_anchors = []

        # 4. make anchors
        for i in range(feature_height):                    # f_h
            for j in range(feature_width):                 # f_w
                c_x = (j + 0.5) / feature_width            # (0-1 scaling)
                c_y = (i + 0.5) / feature_height           # (0-1 scaling)
                for aspect_ratio in aspect_ratios:
                    for area in areas:
                        w = (area / image_width) * np.sqrt(aspect_ratio)
                        h = (area / image_height) / np.sqrt(aspect_ratio)
                        anchor = [c_x,
                                  c_y,
                                  w,
                                  h]
                        center_anchors.append(anchor)

        center_anchors = np.array(center_anchors).astype(np.float32)
        center_anchors = torch.FloatTensor(center_anchors)  # .to(device)

        # -------------------- 5. ignore the cross-boundary anchors --------------------

        # 5.1. convert corner anchors
        corner_anchors = cxcy_to_xy(center_anchors)

        # 5.2. check cross-boundary anchors
        keep = ((corner_anchors[:, 0] >= 0) & (corner_anchors[:, 1] >= 0) \
                & (corner_anchors[:, 2] < 1) & (corner_anchors[:, 3] < 1))

        # center_anchors = center_anchors[keep]
        # At (600, 1000) image has 20646 all anchors but the number of cross-boundary anchors is 7652.

        visualization = False
        if visualization:

            # original
            corner_anchors = cxcy_to_xy(center_anchors)

            # # center anchor clamp 방식!
            # corner_anchors = cxcy_to_xy(center_anchors).clamp(0, 1)
            # center_anchors = xy_to_cxcy(corner_anchors)

            from matplotlib.patches import Rectangle
            import matplotlib.pyplot as plt

            size = 300
            img = torch.ones([size, size, 3], dtype=torch.float32)

            plt.imshow(img)
            axes = plt.axes()
            axes.set_xlim([-1 * size, 3 * size])
            axes.set_ylim([-1 * size, 3 * size])

            for anchor in corner_anchors:
                x1 = anchor[0] * size
                y1 = anchor[1] * size
                x2 = anchor[2] * size
                y2 = anchor[3] * size

                plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                              width=x2 - x1,
                                              height=y2 - y1,
                                              linewidth=1,
                                              edgecolor=[0, 1, 0],
                                              facecolor='none'
                                              ))
            plt.show()

        return center_anchors, keep


if __name__ == '__main__':
    retina_anchor = FasterRCNN_Anchor()

    image_sizes = [[600, 1000], [800, 800], [880, 960]]
    # tic = time.time()
    # # ** 1st experiments
    # first_anchors = []
    # for image_size in image_sizes:
    #     anchor = retina_anchor.create_anchors(image_size=image_size, num_pooling=4)
    #     first_anchors.append(anchor)
    #     print(time.time() - tic)
    # print("whole time: ", time.time() - tic)

    # # make retina anchors
    # # 0.06779313087463379
    # # make retina anchors
    # # 0.16553187370300293
    # # make retina anchors
    # # 0.30914855003356934

    # ** 2nd experiments 0.0009989738464355469
    # tic = time.time()
    frcnn_anchor_maker = FRCNNAnchorMaker()
    second_anchors = []
    anchor_base = frcnn_anchor_maker.generate_anchor_base()
    # print(time.time() - tic)
    for image_size in image_sizes:
        tic = time.time()
        anchor = frcnn_anchor_maker._enumerate_shifted_anchor(origin_image_size=image_size)
        anchor = torch.from_numpy(anchor).cuda()
        # n_anchor = anchor.shape[0] / ((image_size[0] // 16) * (image_size[1] // 16))
        # print("num_anchors : ", n_anchor)
        # second_anchors.append(anchor)
        print(time.time() - tic)
    print("whole time: ", time.time() - tic)
    # print('a')
    # center_anchor = anchor
    # print(center_anchor)

    # 0.0010082721710205078
    # 0.001994609832763672
    # 0.002991914749145508

    # 약 1000배 차이



