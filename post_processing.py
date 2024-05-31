# Post-processing functions for the output of the model

import cv2

import numpy as np

import torch

from dataset import 按斜率滑動到裁剪範圍內


def detect_postprocessing(pred_cls, pred_reg, anchors_pos, image_width, image_height, conf_thresh=0.1, topk=1, with_aqe=False, factor=0.7):
    """
    Post-processing for the detection head predictions
    Args:
        pred_cls (torch.Tensor): predicted class scores. shape [num_total_anchors, num_classes]
        pred_reg (torch.Tensor): predicted regression values. shape [num_total_anchors, 5]
        anchors_pos (torch.Tensor): anchor positions. shape [num_total_anchors, 4]
        conf_thresh (float): confidence threshold for the detection head
        topk (int): number of top-k anchors to keep
        with_aqe (bool): whether to use angle quality estimation
        factor (float): the weight factor for the confidence score and angle quality score
    Returns:
        topk_score (torch.Tensor): top-k scores (combine confidence and angle quality if with_aqe). shape [k,]
        topk_endpoints (torch.Tensor): the predicted endpoints for the top-k anchors after confidence thresholding. shape [k, 4]
        topk_pred_cals (torch.Tensor): the predicted center, angle, length for the top-k anchors. shape [k, 4]
    """
    # find the highest confidence score for each anchor with the corresponding class id
    conf, cls_id = torch.max(pred_cls, dim=-1)  # both [num_total_anchors,]

    # confidence thresholding
    conf = torch.where(conf >= conf_thresh, conf, torch.zeros_like(conf))
    pred_reg[:, 3] = torch.where(conf >= conf_thresh, pred_reg[:, 3], torch.zeros_like(pred_reg[:, 3]))

    # get the score for each anchor
    if not with_aqe:
        score = conf
    else:
        # combine the confidence score and angle quality score
        aq = 1 - pred_reg[:, 3]  # [num_total_anchors,]
        score = factor * conf + (1 - factor) * aq  # [num_total_anchors,]

    # get top-k anchors with the highest scores
    topk_score, topk_idx = torch.topk(score, k=topk, dim=-1)  # both [k,]
    topk_id = cls_id[topk_idx]  # [k,]

    # get the top-k anchor positions
    topk_anchors_pos = anchors_pos[topk_idx]  # [k, 5]

    # transform the anchors with endpoints coordinates to the center, width, height, length, and angle format
    x1, y1 = topk_anchors_pos[:, 0], topk_anchors_pos[:, 1]
    x2, y2 = topk_anchors_pos[:, 2], topk_anchors_pos[:, 3]

    # center points
    topk_anchors_ctr_x = (x1 + x2) / 2
    topk_anchors_ctr_y = (y1 + y2) / 2

    # width, height
    topk_anchors_width = x2 - x1
    topk_anchors_height = y2 - y1

    # length
    topk_anchors_length = torch.sqrt(torch.pow(x2 - x1, 2) + torch.pow(y2 - y1, 2))

    # -------------------------------------------------------------------------------
    # remember to modify the following code to get the predicted center, angle, length when the regression targets are modified !!!!!
    # get the predicted center, angle, length of each anchor

    # rescale the regression targets due to target scaling in loss calculation
    target_scale = 0.5
    pred_reg[topk_idx, 0] = pred_reg[topk_idx, 0] * target_scale  # center x
    pred_reg[topk_idx, 1] = pred_reg[topk_idx, 1] * target_scale  # center y
    if not with_aqe:
        pred_reg[topk_idx, 3] = pred_reg[topk_idx, 3] * target_scale  # length
    else:
        pred_reg[topk_idx, 4] = pred_reg[topk_idx, 4] * target_scale  # length

    # get the predicted center, angle, length for the top-k anchors
    topk_pred_ctr_x = pred_reg[topk_idx, 0] * topk_anchors_width + topk_anchors_ctr_x  # [k,]
    topk_pred_ctr_y = pred_reg[topk_idx, 1] * topk_anchors_height + topk_anchors_ctr_y  # [k,]
    topk_pred_theta = pred_reg[topk_idx, 2]  # [k,]
    if not with_aqe:
        topk_pred_length = torch.exp(pred_reg[topk_idx, 3]) * topk_anchors_length
    else:
        topk_pred_sigma = pred_reg[topk_idx, 3]  # [k,]
        topk_pred_length = torch.exp(pred_reg[topk_idx, 4]) * topk_anchors_length  # [k,]

    # concate the predicted center, angle, length
    topk_pred_cals = torch.stack([topk_pred_ctr_x, topk_pred_ctr_y, topk_pred_theta, topk_pred_length], dim=-1)  # [k, 4]
    # -------------------------------------------------------------------------------

    # transform the center, angle, length to the endpoints
    topk_centers = topk_pred_cals[:, :2]  # [k, 2]
    topk_theta = topk_pred_cals[:, 2]  # [k,]
    topk_length = topk_pred_cals[:, 3]  # [k,]

    topk_dx = 0.5 * topk_length * torch.cos(topk_theta)  # [k,]
    topk_dy = 0.5 * topk_length * torch.sin(topk_theta)  # [k,]
    topk_dx = topk_dx.unsqueeze(-1)  # [k, 1]
    topk_dy = topk_dy.unsqueeze(-1)  # [k, 1]

    topk_endpoints_1 = topk_centers - torch.cat((topk_dx, topk_dy), dim=-1)  # [k, 2]
    topk_endpoints_2 = topk_centers + torch.cat((topk_dx, topk_dy), dim=-1)  # [k, 2]
    topk_endpoints = torch.cat((topk_endpoints_1, topk_endpoints_2), dim=-1)  # [k, 4]

    # check confidence value and cut the endpoints that are out of the image boundary
    for k in range(topk_endpoints.shape[0]):
        # confidence thresholding again
        if conf[topk_idx][k] == 0:
            topk_endpoints[k] = torch.zeros_like(topk_endpoints[k], dtype=torch.float32)
        else:
            endpoints = 按斜率滑動到裁剪範圍內(topk_endpoints[k], 0, 0, image_width, image_height)  # [2, 2]
            endpoints = [endpoints[0][0], endpoints[0][1], endpoints[1][0], endpoints[1][1]]  # [4,]
            topk_endpoints[k] = torch.tensor(endpoints, dtype=torch.float32)  # [4,]

    return topk_score, topk_id, topk_endpoints, topk_pred_cals
