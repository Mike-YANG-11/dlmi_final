# loss functions

import cv2

import math

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

        if self.alpha > 0:
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


## FTLoss fo Segmentation
## https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch?scriptVersionId=68471013&cellId=19
class SegFocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SegFocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.4, beta=0.6, gamma=1):
        """
        TL = TP / (TP + alpha* FN + beta* FP)
        Set α > β to reduce false positive;  (maybe for Semi Sup?)
        set β > α will to reduce false negatives
        """   
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

### Error when evaluate:  Assertion `input_val >= zero && input_val <= one` failed.
## Combo Loss fo Segmentation
## https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch?scriptVersionId=68471013&cellId=27
class SegComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.5, beta=0.7):
        super(SegComboLoss, self).__init__()
        self.alpha = alpha    # weighted contribution of modified CE loss compared to Dice loss
        self.beta = beta          # < 0.5 penalises FP more, > 0.5 penalises FN more

    def forward(self, inputs, targets, smooth=1, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (self.beta * ((targets * torch.log(inputs)) + ((1 - self.beta) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.alpha * weighted_ce) - ((1 - self.alpha) * dice)
        
        return combo

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


# Loss for Object Detection
# https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/losses.py
class DetLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, siou_loss=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.siou_loss = siou_loss

    def coordinates_transform(self, anchors_pos):
        """
        Convert the left-top and right-bottom coordinates of the anchors to center, width, height, length, and angle.
        Args:
            anchors_pos (torch.Tensor): The anchor left-top and right-bottom coordinates. Shape (num_total_anchors, 4).
        Returns:
            ctr_x, ctr_y, width, height, length, angle (torch.Tensor): The center, width, height, length, and angle of the anchors. Each of shape (num_total_anchors,).
        """
        x1, y1 = anchors_pos[:, 0], anchors_pos[:, 1]
        x2, y2 = anchors_pos[:, 2], anchors_pos[:, 3]

        # center points
        anchors_ctr_x = (x1 + x2) / 2
        anchors_ctr_y = (y1 + y2) / 2

        # width, height
        anchors_width = x2 - x1
        anchors_height = y2 - y1

        # length
        anchors_length = torch.sqrt(torch.pow(x2 - x1, 2) + torch.pow(y2 - y1, 2))

        # angle (since the original anchor coordinates are left-top and right-bottom, the angle is in the left-top to right-bottom orientation)
        anchors_theta = torch.where(x1 == x2, torch.sign(y2 - y1) * math.pi / 2, torch.atan((y2 - y1) / (x2 - x1)))

        return anchors_ctr_x, anchors_ctr_y, anchors_width, anchors_height, anchors_length, anchors_theta

    def calc_iou(self, anchors_pos, cal_annotations):
        """
        Calculate the Intersection over Union (IoU) of anchors with center, angle, length annotations.
        Args:
            anchors_pos (torch.Tensor): The anchor left-top and right-bottom coordinates. Shape (num_total_anchors, 4).
            cal_annotations (torch.Tensor): The annotations with (center_x, center_y, angle, length). Shape (num_annotations, 4).

        Returns:
            IoU (torch.Tensor): The IoU of each anchor with all annotations. Shape (num_anchors, num_annotations).
        """
        # convert the center, angle, length annotations to left-top and right-bottom coordinates
        # Extract components
        centers = cal_annotations[:, :2]  # shape (num_annotations, 2)
        angles_radians = cal_annotations[:, 2]  # shape (num_annotations,)
        lengths = cal_annotations[:, 3]  # shape (num_annotations,)

        # Calculate half-length offsets
        dx = torch.abs(0.5 * lengths * torch.cos(angles_radians))  # shape (num_annotations,)
        dy = torch.abs(0.5 * lengths * torch.sin(angles_radians))  # shape (num_annotations,)

        # Calculate the endpoints
        dx = dx.unsqueeze(1)  # reshape for broadcasting, shape (num_annotations, 1)
        dy = dy.unsqueeze(1)  # reshape for broadcasting, shape (num_annotations, 1)

        left_top_points = centers - torch.cat((dx, dy), dim=1)  # shape (num_annotations, 2)
        right_bottom_points = centers + torch.cat((dx, dy), dim=1)  # shape (num_annotations, 2)

        # Combine endpoints into one tensor with top-left and bottom-right points
        bbox_annotations = torch.cat((left_top_points, right_bottom_points), dim=1)  # shape (num_annotations, 4)

        # calculate the area of annotation boxes
        area = (bbox_annotations[:, 2] - bbox_annotations[:, 0]) * (bbox_annotations[:, 3] - bbox_annotations[:, 1])

        # calculate the intersection width of each anchors with all annotations
        iw = torch.min(torch.unsqueeze(anchors_pos[:, 2], dim=1), bbox_annotations[:, 2]) - torch.max(
            torch.unsqueeze(anchors_pos[:, 0], 1), bbox_annotations[:, 0]
        )  # shape (num_anchors, num_annotations)

        # calculate the intersection height of each anchors with all annotations
        ih = torch.min(torch.unsqueeze(anchors_pos[:, 3], dim=1), bbox_annotations[:, 3]) - torch.max(
            torch.unsqueeze(anchors_pos[:, 1], 1), bbox_annotations[:, 1]
        )  # shape (num_anchors, num_annotations)

        # clamp the negative values to 0
        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)

        # calculate the union area of each anchors with all annotations
        ua = (
            torch.unsqueeze((anchors_pos[:, 2] - anchors_pos[:, 0]) * (anchors_pos[:, 3] - anchors_pos[:, 1]), dim=1) + area - iw * ih
        )  # shape (num_anchors, num_annotations)
        ua = torch.clamp(ua, min=1e-8)

        intersection = iw * ih

        IoU = intersection / ua

        return IoU

    def forward(self, classifications, regressions, anchors_pos, annotations):
        """
        Calculate the classification and SIoU regression loss for the object detection task.
        Args:
            classifications (torch.Tensor): The predicted class labels for each anchor. Shape (batch_size, num_total_anchors, num_classes).
            regressions (torch.Tensor): The predicted shifts of center, angle, length for each anchor. Shape (batch_size, num_total_anchors, 4).
            anchors_pos (torch.Tensor): The anchor left-top and right-bottom coordinates. Shape (num_total_anchors, 4).
            annotations (torch.Tensor): The annotations with center, angle, length, and class labels. Shape (batch_size, num_annotations, 5).
        Returns:
            total_loss (torch.Tensor): The mean detection loss of the current batch.
        """
        batch_size = classifications.shape[0]

        # list to store the classification and regression losses
        classification_losses = []
        regression_losses = []

        # convert left-top & right-bottom coordinates to center, width, height, length, and angle
        anchors_ctr_x, anchors_ctr_y, anchors_width, anchors_height, anchors_length, anchors_theta = self.coordinates_transform(anchors_pos)

        for idx in range(batch_size):
            classification = classifications[idx, :, :]  # shape (num_total_anchors, num_classes)
            regression = regressions[idx, :, :]  # shape (num_total_anchors, 4)
            annotation = annotations[idx, :, :]  # shape (num_annotations, 5)

            # remove the annotations with cls_id = -1
            annotation = annotation[annotation[:, 4] != -1]  # label -1 as background class

            # clamp the classification values to avoid log(0)
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            """ no annotations for this image"""

            if annotation.shape[0] == 0:
                # compute the focal loss for classification
                # all the anchors are negative
                bce_loss = -(torch.log(1.0 - classification))
                focal_loss = (classification**self.gamma) * bce_loss

                if self.alpha > 0:
                    alpha_factor = 1.0 - torch.ones(classification.shape).cuda() * self.alpha
                    focal_loss = alpha_factor * focal_loss

                classification_losses.append(focal_loss.sum())
                regression_losses.append(torch.tensor(0).float().cuda())  # no regression loss

                continue

            """ compute the loss for classification """

            # calculate the IoU between each anchor-annotation pair
            IoU = self.calc_iou(anchors_pos, annotation[:, :4])  # shape (num_total_anchors, num_annotations)

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # shape (num_total_anchors,) both

            # assign class labels to the anchors based on the IoU values
            # max IoU < 0.4 are negative anchors (all cls_id = 0)
            # max IoU > 0.5 are positive anchors (corresponding cls_id = 1)
            # max IoU in [0.4, 0.5) are ignored anchors (all cls_id = -1)

            # all the anchors are initialized as ignored anchors (all cls_id = -1)
            targets = torch.ones(classification.shape).cuda() * -1

            # assign negative anchors (all cls_id = 0)
            targets[torch.lt(IoU_max, 0.4), :] = 0

            # assign positive anchor indices
            positive_indices = torch.ge(IoU_max, 0.5)

            # assign each anchor its corresponding annotations (center_x, center_y, angle, length, cls_id) with which has the highest IoU
            assigned_annotations = annotation[IoU_argmax, :]  # shape (num_total_anchors, 5)

            # assign the targets with the one-hot encoding of the class labels
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            # assign the annotations with maximum IoU less than 0.5 to the anchors has the maximum IoU with them
            for annotation_idx in range(annotation.shape[0]):
                if torch.max(IoU[:, annotation_idx]) < 0.5:  # torch.max returns the value only if dim is not specified
                    assign_anchor_idx = torch.argmax(IoU[:, annotation_idx])
                    # here we can assign the anchor with the annotation without considering repeated assignments
                    # since we only have one annotation per image
                    # if there is more than one annotation per image, we need to modify the code here
                    targets[assign_anchor_idx, :] = 0
                    targets[assign_anchor_idx, annotation[annotation_idx, 4].long()] = 1
                    assigned_annotations[assign_anchor_idx, :] = annotation[annotation_idx, :]

            # final positive anchor indices
            positive_indices = torch.gt(targets.sum(dim=-1), 0)
            num_positive_anchors = positive_indices.sum()

            # compute the focal loss for classification
            bce_loss = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            p_t = torch.where(torch.eq(targets, 1.0), classification, 1.0 - classification)
            focal_loss = ((1 - p_t) ** self.gamma) * bce_loss

            if self.alpha > 0:
                alpha_factor = torch.ones(targets.shape).cuda() * self.alpha
                alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
                focal_loss = alpha_factor * focal_loss

            # ignore the anchors with all cls_id = -1
            focal_loss = torch.where(torch.ne(targets, -1.0), focal_loss, torch.zeros(focal_loss.shape).cuda())

            # append classification loss to the list
            classification_losses.append(focal_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            """ compute the loss for regression """

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                # -------------------------------------------------------------------------------
                # SIoU loss calculation using center, angle, length
                anchors_ctr_x_posit = anchors_ctr_x[positive_indices]
                anchors_ctr_y_posit = anchors_ctr_y[positive_indices]
                anchors_width_posit = anchors_width[positive_indices]
                anchors_height_posit = anchors_height[positive_indices]
                anchors_length_posit = anchors_length[positive_indices]
                anchors_theta_posit = anchors_theta[positive_indices]

                # use the predicted shifts to compute the final regression center, angle, length
                pred_ctr_x = regression[positive_indices, 0] * anchors_width_posit + anchors_ctr_x_posit
                pred_ctr_y = regression[positive_indices, 1] * anchors_height_posit + anchors_ctr_y_posit

                """ Design the regression for angle and length """
                ## TODO: check the design of the angle regression

                # version 1: the reference angle is the left-top to right-bottom orientation of the anchor
                pred_theta = regression[positive_indices, 2] + anchors_theta_posit

                # version 2: the reference angle is the horizontal orientation
                # this design is not good since the reference angle is consistent for all anchors)
                # pred_theta = regression[positive_indices, 2]

                # version 3: the reference angle is the orientation of the assigned annotation
                # this angle regression target no longer contatins the information of the direction of the angle
                # but only determines how sloped the line is (positive for more sloped, negative for less sloped)
                # anchors_theta_posit = anchors_theta_posit * torch.where(assigned_annotations[:, 4] == 0, 1, -1)
                # pred_theta = torch.zeros_like(anchors_theta_posit, dtype=torch.float32).cuda()
                # for pos_idx in range(pred_theta.shape[0]):
                #     if anchors_theta_posit[pos_idx] > 0:
                #         # left-top to right-bottom orientation: positive regression results in clockwise rotation
                #         pred_theta[pos_idx] = anchors_theta_posit[pos_idx] + regression[positive_indices, 2][pos_idx]
                #     else:
                #         # right-top to left-bottom orientation: positive regression results in counter-clockwise rotation
                #         pred_theta[pos_idx] = anchors_theta_posit[pos_idx] - regression[positive_indices, 2][pos_idx]

                # Although the length regression follows the original design, the training results are not good
                pred_length = torch.exp(regression[positive_indices, 3]) * anchors_length_posit  ## TODO: check this design
                """ End of design """

                # prediceted center, angle, length after regression
                pred_cal = torch.cat((pred_ctr_x.unsqueeze(1), pred_ctr_y.unsqueeze(1), pred_theta.unsqueeze(1), pred_length.unsqueeze(1)), dim=1)

                # calculate the SIoU loss
                reg_loss = self.siou_loss(pred_cal, assigned_annotations)
                # -------------------------------------------------------------------------------

                # -------------------------------------------------------------------------------
                # Following is the original regression loss calculation using bounding box coordinates

                # anchors_width_posit = anchors_width[positive_indices]
                # anchors_height_posit = anchors_height[positive_indices]
                # anchors_ctr_x_posit = anchors_ctr_x[positive_indices]
                # anchors_ctr_y_posit = anchors_ctr_y[positive_indices]

                # gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                # gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                # gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                # gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # # clip widths to 1
                # gt_widths = torch.clamp(gt_widths, min=1)
                # gt_heights = torch.clamp(gt_heights, min=1)

                # targets_dx = (gt_ctr_x - anchors_ctr_x_posit) / anchors_width_posit
                # targets_dy = (gt_ctr_y - anchors_ctr_y_posit) / anchors_height_posit
                # targets_dw = torch.log(gt_widths / anchors_width_posit)
                # targets_dh = torch.log(gt_heights / anchors_height_posit)

                # targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                # targets = targets.t()

                # if torch.cuda.is_available():
                #     targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                # else:
                #     targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                # negative_indices = 1 + (~positive_indices)

                # regression_diff = torch.abs(targets - regression[positive_indices, :])

                # regression_loss = torch.where(
                #     torch.le(regression_diff, 1.0 / 9.0),
                #     0.5 * 9.0 * torch.pow(regression_diff, 2),
                #     regression_diff - 0.5 / 9.0,
                # )
                # -------------------------------------------------------------------------------

                regression_losses.append(reg_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


## SIoU + needle cost for Object Detection
class SIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7  ## avoid dividing 0
        self.needle_angle_cost_weight = 2

    def length_angle_to_endpoint_coordinates(self, cals):
        """Assume input_tensor is of shape (batch_size, 4)
        and each row is [center_x (x2), center_y (y2), angles_radians, length]
        """

        # Extract components
        centers = cals[:, :2]  # shape (batch_size, 2)
        angles_radians = cals[:, 2]  # shape (batch_size,)
        lengths = cals[:, 3]  # shape (batch_size,)

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
        permute_pred_endpoints = torch.index_select(pred_endpoints, 1, torch.LongTensor([0, 2, 1, 3]).cuda())  ## (x1, x3, y1, y3)
        permute_gt_endpoints = torch.index_select(gt_endpoints, 1, torch.LongTensor([0, 2, 1, 3]).cuda())
        b1_x0, b1_y0 = torch.min(permute_pred_endpoints[:, :2], 1)[0], torch.min(permute_pred_endpoints[:, 2:], 1)[0]
        b1_x1, b1_y1 = torch.max(permute_pred_endpoints[:, :2], 1)[0], torch.max(permute_pred_endpoints[:, 2:], 1)[0]
        b2_x0, b2_y0 = torch.min(permute_gt_endpoints[:, :2], 1)[0], torch.max(permute_gt_endpoints[:, 2:], 1)[0]
        b2_x1, b2_y1 = torch.max(permute_gt_endpoints[:, :2], 1)[0], torch.max(permute_gt_endpoints[:, 2:], 1)[0]

        # Intersection area
        inter = (torch.min(b1_x1, b2_x1) - torch.max(b1_x0, b2_x0)).clamp(0) * (torch.min(b1_y1, b2_y1) - torch.max(b1_y0, b2_y0)).clamp(0)

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
        sin_needle_angle = torch.sin(gt_angles_degrees - pred_angles_degrees)  ## sine should be as close to 0 as possibile
        needle_angle_cost = torch.pow(sin_needle_angle, 2)  ## fix negative value
        # print(f"needle angle {needle_angle_cost}")

        return 1 - (iou + 0.5 * (distance_cost + shape_cost)) + needle_angle_cost * self.needle_angle_cost_weight
