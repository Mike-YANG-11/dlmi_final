# loss functions

import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Focal Loss for Segmentation
class SegFocalLoss(nn.Module):
    def __init__(self, alpha=-1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        # flatten prediction and target tensors
        preds = preds.reshape(-1)  # [N, 1, H, W] -> [N*H*W]
        targets = targets.reshape(-1)  # [N, 1, H, W] -> [N*H*W]

        # compute binary cross-entropy
        bce_loss = F.binary_cross_entropy(preds, targets, reduction="none")
        p_t = preds * targets + (1 - preds) * (1 - targets)
        focal_loss = ((1 - p_t) ** self.gamma) * bce_loss

        if self.alpha >= 0:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_factor * focal_loss

        return focal_loss.mean()


# Dice Loss for Segmentation
class SegDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, smooth=1e-6):
        # flatten prediction and target tensors
        preds = preds.reshape(-1)  # [N, 1, H, W] -> [N*H*W]
        targets = targets.reshape(-1)  # [N, 1, H, W] -> [N*H*W]

        intersection = (preds * targets).sum()
        dice_coeff = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

        return 1 - dice_coeff


# Hausdorff Distance Loss for Segmentation
class SegHDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = np.zeros_like(img[batch][0])
            fg_mask[img[batch][0] > 0.5] = 255
            fg_mask[img[batch][0] <= 0.5] = 0
            fg_mask = fg_mask.astype(np.uint8)

            if fg_mask.any():
                bg_mask = 255 - fg_mask

                fg_dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 3)
                bg_dist = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 3)

                # distance transform of the boundary is equal to
                # the distance transform of the background plus the distance transform of the foreground
                field[batch] = fg_dist + bg_dist

        return field

    def forward(self, preds: torch.Tensor, masks: torch.Tensor, debug=False) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert preds.dim() == 4 or preds.dim() == 5, "Only 2D and 3D supported"
        assert preds.dim() == masks.dim(), "Prediction and target need to be of same dimension"

        preds_dt = torch.from_numpy(self.distance_field(preds.detach().cpu().numpy())).float().to(preds.device)
        masks_dt = torch.from_numpy(self.distance_field(masks.detach().cpu().numpy())).float().to(masks.device)

        # Hausdorff Distance Loss
        pred_error = (preds - masks) ** 2
        distance = preds_dt**self.alpha + masks_dt**self.alpha  # (Eq.8 in the paper)
        # distance = masks_dt ** self.alpha # (Eq.9 in the paper)

        dt_field = pred_error * distance
        hd_loss = dt_field.mean()

        if debug:
            return (
                hd_loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    preds_dt.cpu().numpy()[0, 0],
                    masks_dt.cpu().numpy()[0, 0],
                ),
            )
        else:
            return hd_loss


# IoU Calculation for Object Detection
def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


# Focal Loss for Object Detection
class DetFocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # with shape (number of total anchors, 4)

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            # boxes[idx, 0] = x1
            # boxes[idx, 1] = y1
            # boxes[idx, 2] = x2
            # boxes[idx, 3] = y2
            # boxes[idx, 4] = cls_id
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]  # remove the bbox with class_id = -1

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:  # no annotations for this image
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0,
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(
            dim=0, keepdim=True
        )


## SIoU + needle cost for Object Detection
class SIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7  ## avoid dividing 0
        self.needle_angle_cost_weight = 2

    def length_angle_to_endpoint_coordinates(self, bboxs):
        """Assume input_tensor is of shape (batch_size, 4)
        and each row is [center_x (x2), center_y (y2), angles_radians, length]
        """

        # Extract components
        centers = bboxs[:, :2]  # shape (batch_size, 2)
        angles_radians = bboxs[:, 2]  # shape (batch_size,)
        lengths = bboxs[:, 3]  # shape (batch_size,)

        # Calculate half-length offsets
        dx = 0.5 * lengths * torch.cos(angles_radians)
        dy = 0.5 * lengths * torch.sin(angles_radians)

        # Calculate the endpoints
        dx = dx.unsqueeze(1)  # Reshape for broadcasting, shape (batch_size, 1)
        dy = dy.unsqueeze(1)  # Reshape for broadcasting, shape (batch_size, 1)

        endpoints_1 = centers - torch.cat((dx, dy), dim=1)  # shape (batch_size, 2)
        endpoints_2 = centers + torch.cat((dx, dy), dim=1)  # shape (batch_size, 2)

        # Combine endpoints into one tensor
        endpoints = torch.cat((endpoints_1, endpoints_2), dim=1)  # shape (batch_size, 4)
        # print(f'endppoints {endpoints}\n{endpoints.shape}')
        return endpoints

    ## https://learnopencv.com/yolo-loss-function-siou-focal-loss/#aioseo-pytorch-implementation
    def forward(self, preds, targets):  ## [B, 4]

        pred_endpoints = self.length_angle_to_endpoint_coordinates(preds)  ## get (x1,y1, x3,y3)
        gt_endpoints = self.length_angle_to_endpoint_coordinates(targets)  ## TODO: use original value from json?
        pred_angles_degrees = preds[:, 2]  # shape (batch_size,)
        gt_angles_degrees = targets[:, 2]  # shape (batch_size,)

        ## get bottom-left corner (x0, y0) and up-right corner (x1, y1)
        ## torch max or min output a tuple of two tensors (value, indice)
        permute_pred_endpoints = torch.index_select(
            pred_endpoints, 1, torch.LongTensor([0, 2, 1, 3]).cuda()
        )  ## (x1, x3, y1, y3)
        permute_gt_endpoints = torch.index_select(gt_endpoints, 1, torch.LongTensor([0, 2, 1, 3]).cuda())
        b1_x0, b1_y0 = torch.min(permute_pred_endpoints[:, :2], 1)[0], torch.min(permute_pred_endpoints[:, 2:], 1)[0]
        b1_x1, b1_y1 = torch.max(permute_pred_endpoints[:, :2], 1)[0], torch.max(permute_pred_endpoints[:, 2:], 1)[0]
        b2_x0, b2_y0 = torch.min(permute_gt_endpoints[:, :2], 1)[0], torch.max(permute_gt_endpoints[:, 2:], 1)[0]
        b2_x1, b2_y1 = torch.max(permute_gt_endpoints[:, :2], 1)[0], torch.max(permute_gt_endpoints[:, 2:], 1)[0]

        # Intersection area
        inter = (torch.min(b1_x1, b2_x1) - torch.max(b1_x0, b2_x0)).clamp(0) * (
            torch.min(b1_y1, b2_y1) - torch.max(b1_y0, b2_y0)
        ).clamp(0)

        # Union Area
        w1, h1 = b1_x1 - b1_x0, b1_y1 - b1_y0 + self.eps
        w2, h2 = b2_x1 - b2_x0, b2_y1 - b2_y0 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps

        # IoU value of the bounding boxes
        iou = inter / union
        cw = torch.max(b1_x1, b2_x1) - torch.min(b1_x0, b2_x0)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y1, b2_y1) - torch.min(b1_y0, b2_y0)  # convex height
        s_cw = (b2_x0 + b2_x1 - b1_x0 - b1_x1) * 0.5
        s_ch = (b2_y0 + b2_y1 - b1_y0 - b1_y1) * 0.5
        sigma = torch.pow(s_cw**2 + s_ch**2, 0.5) + self.eps
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)

        # Angle Cost
        angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - np.pi / 4), 2)
        # print(f"angle {angle_cost}")

        # Distance Cost
        rho_x = (s_cw / (cw + self.eps)) ** 2
        rho_y = (s_ch / (ch + self.eps)) ** 2
        gamma = 2 - angle_cost
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        # print(f"dist {distance_cost}")

        # Shape Cost
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        # print(f"shape {shape_cost}")

        ## Needle Angle Cost
        sin_needle_angle = torch.sin(
            gt_angles_degrees - pred_angles_degrees
        )  ## sine should be as close to 0 as possibile
        needle_angle_cost = torch.pow(sin_needle_angle, 2)  ## fix negative value
        # print(f"needle angle {needle_angle_cost}")

        return 1 - (iou + 0.5 * (distance_cost + shape_cost)) + needle_angle_cost * self.needle_angle_cost_weight
