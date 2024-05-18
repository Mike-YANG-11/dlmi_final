import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super().__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2**x for x in self.pyramid_levels]  # [8, 16, 32, 64, 128]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]  # [32, 64, 128, 256, 512]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):

        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [
            (image_shape + 2**x - 1) // (2**x) for x in self.pyramid_levels
        ]  # [[28, 28], [14, 14], [7, 7], [4, 4], [2, 2]] for 224x224 input

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)  # with shape (1, 9441, 4)
        # contains all the anchors [x1, y1, x2, y2] for all pyramid levels

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])  # [1.00, 1.26, 1.59]

    num_anchors = len(ratios) * len(scales)  # 9

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    # np.tile(scales, (2, len(ratios))).T =
    # [[1.         1.        ]
    #  [1.25992105 1.25992105]
    #  [1.58740105 1.58740105]
    #  [1.         1.        ]
    #  [1.25992105 1.25992105]
    #  [1.58740105 1.58740105]
    #  [1.         1.        ]
    #  [1.25992105 1.25992105]
    #  [1.58740105 1.58740105]]

    # anchors =
    # [[ 0.          0.         32.         32.        ]
    #  [ 0.          0.         40.3174736  40.3174736 ]
    #  [ 0.          0.         50.79683366 50.79683366]
    #  [ 0.          0.         32.         32.        ]
    #  [ 0.          0.         40.3174736  40.3174736 ]
    #  [ 0.          0.         50.79683366 50.79683366]
    #  [ 0.          0.         32.         32.        ]
    #  [ 0.          0.         40.3174736  40.3174736 ]
    #  [ 0.          0.         50.79683366 50.79683366]] for base_size=32

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]
    # areas = [1024., 1625.50, 2580.32, 1024., 1625.50, 2580.32, 1024., 1625.50, 2580.32] for base_size=32

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    # anchors =
    # [[ 0.          0.         45.254834   22.627417  ]
    #  [ 0.          0.         57.01751796 28.50875898]
    #  [ 0.          0.         71.83757109 35.91878555]
    #  [ 0.          0.         32.         32.        ]
    #  [ 0.          0.         40.3174736  40.3174736 ]
    #  [ 0.          0.         50.79683366 50.79683366]
    #  [ 0.          0.         22.627417   45.254834  ]
    #  [ 0.          0.         28.50875898 57.01751796]
    #  [ 0.          0.         35.91878555 71.83757109]] for base_size=32

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    # anchors =
    # [[-22.627417   -11.3137085   22.627417    11.3137085 ]
    #  [-28.50875898 -14.25437949  28.50875898  14.25437949]
    #  [-35.91878555 -17.95939277  35.91878555  17.95939277]
    #  [-16.         -16.          16.          16.        ]
    #  [-20.1587368  -20.1587368   20.1587368   20.1587368 ]
    #  [-25.39841683 -25.39841683  25.39841683  25.39841683]
    #  [-11.3137085  -22.627417    11.3137085   22.627417  ]
    #  [-14.25437949 -28.50875898  14.25437949  28.50875898]
    #  [-17.95939277 -35.91878555  17.95939277  35.91878555]] for base_size=32

    return anchors


def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2**x - 1) // (2**x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    # shift_x =
    # [  4.  12.  20.  28.  36.  44.  52.  60.  68.  76.  84.  92. 100. 108.
    #  116. 124. 132. 140. 148. 156. 164. 172. 180. 188. 196. 204. 212. 220.] for C3
    # [  8.  24.  40.  56.  72.  88. 104. 120. 136. 152. 168. 184. 200. 216.] for C4
    # [ 16.  48.  80. 112. 144. 176. 208.] for C5
    # [ 32.  96. 160. 224.] for C6
    # [ 64. 192.] for C7

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # shifts.shape = (784, 4) for C3, (196, 4) for C4, (49, 4) for C5, (16, 4) for C6, (4, 4) for C7

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]  # number of anchors per position
    K = shifts.shape[0]  # number of anchor positions in the feature map
    # 784 for C3, 196 for C4, 49 for C5, 16 for C6, 4 for C7

    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    # (784, 4) for C3, (196, 4) for C4, (49, 4) for C5, (16, 4) for C6, (4, 4) for C7

    return all_anchors


anchor = Anchors()
image = torch.randn(8, 3, 224, 224)
anchors = anchor(image)
