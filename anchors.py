# Anchor generator for RetinaNet

import numpy as np
import torch
import torch.nn as nn


class AnchorGenerator(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super().__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6]
        if strides is None:
            self.strides = [2**x for x in self.pyramid_levels]  # [8, 16, 32, 64]
        if sizes is None:
            self.sizes = [2 ** (x + 1) for x in self.pyramid_levels]  # [16, 32, 64, 128]
        if ratios is None:
            self.ratios = np.array([0.2, 0.5, 1])  ## TODO: modify the aspect ratios? original: [0.5, 1, 2]
        if scales is None:
            self.scales = np.array([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image_shape):
        """
        Construct the the list of left-top and right-bottom coordinates [x1, y1, x2, y2] of all the anchors for the given image shape.
        Args:
            image_shape: image shape, a list of 2 integers [H, W]

        Returns:
            all_anchors: a tensor with shape (num_total_anchors, 4), [x1, y1, x2, y2]
        """
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2**x - 1) // (2**x) for x in self.pyramid_levels]  # [[28, 28], [14, 14], [7, 7], [4, 4]] for 224x224 input

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        return torch.from_numpy(all_anchors.astype(np.float32)).cuda()


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios x scales w.r.t. a reference window.
    """
    if ratios is None:
        ratios = np.array([0.2, 0.5, 1])
    if scales is None:
        scales = np.array([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    """
    Apply shift on the anchor grid using stride & anchor width & height to generate the final set of anchor positions.
    """
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    A = anchors.shape[0]  # number of anchors per position
    K = shifts.shape[0]  # number of anchor positions in the feature map

    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))  # (K*A, 4)

    return all_anchors
