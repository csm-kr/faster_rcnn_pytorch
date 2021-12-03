import time
import torch
import numpy as np
from utils import cxcy_to_xy, xy_to_cxcy


class FasterRCNN_Anchor(object):
    def create_anchors(self, image_size, num_pooling=4):
        print('make retina anchors')

        # get height and width of features
        image_height, image_width = size = image_size  # h, w
        feature_height, feature_width = image_height, image_width

        for i in range(num_pooling):
            feature_height = feature_height // 2
            feature_width = feature_width // 2

        print(feature_height)
        print(feature_width)

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

        center_anchors = center_anchors[keep]
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

        return center_anchors

    # def generate_anchors(self, image_size, num_pooling=4):
    #
    #     print('make retina anchors')
    #     # get height and width of features
    #     image_height, image_width = size = image_size  # h, w
    #     feature_height, feature_width = image_height, image_width
    #     for i in range(num_pooling):
    #         feature_height = feature_height // 2
    #         feature_width = feature_width // 2
    #     print(feature_height)
    #     print(feature_width)
    #     areas = [128, 256, 512]
    #     aspect_ratios = np.array([0.5, 1.0, 2.0])
    #     center_anchors = []
    #
    #     grid_arange = np.arange(grid_size)
    #     xx, yy = np.meshgrid(grid_arange, grid_arange)  # + 0.5  # grid center, [fmsize*fmsize,2]
    #     m_grid = np.concatenate([np.expand_dims(xx, axis=-1), np.expand_dims(yy, -1)], axis=-1) + 0.5
    #     m_grid = m_grid
    #     xy = torch.from_numpy(m_grid)
    #
    #     anchors_wh = np.array(anchors_wh)  # numpy 로 변경
    #     wh = torch.from_numpy(anchors_wh)
    #
    #     xy = xy.view(grid_size, grid_size, 1, 2).expand(grid_size, grid_size, 5, 2).type(torch.float32)  # centor
    #     wh = wh.view(1, 1, 5, 2).expand(grid_size, grid_size, 5, 2).type(torch.float32)  # w, h
    #     center_anchors = torch.cat([xy, wh], dim=3).to(device)


if __name__ == '__main__':
    retina_anchor = FasterRCNN_Anchor()
    tic = time.time()
    anchor = retina_anchor.create_anchors(image_size=(600, 1000), num_pooling=4)
    print(time.time() - tic)
    center_anchor = anchor
    print(center_anchor)


