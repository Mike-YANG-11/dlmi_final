# Visualization Functions

import cv2

import math

import numpy as np

import matplotlib.pyplot as plt

import torch

from dataset import 按斜率滑動到裁剪範圍內


# visualize some batches of image pairs (show T consecutive images, masks and line annotations)
def show_dataset_samples(consec_images, consec_masks, consec_cals, consec_endpoints, consec_labels, max_samples=2, figsize=(8, 8), font_size=10):
    # Show some samples in a batch
    for sample in range(consec_images.shape[0]):
        consec_image = consec_images[sample]
        consec_mask = consec_masks[sample]
        consec_cal = consec_cals[sample]
        consec_endpoint = consec_endpoints[sample]
        consec_label = consec_labels[sample]

        plt.figure(figsize=figsize)
        for t in range(len(consec_image)):
            # image
            image = consec_image[t].numpy()
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

            # mask
            mask = consec_mask[t].numpy()
            mask = (mask * 255).astype(np.uint8)

            # show images, masks, and annotations
            plt.subplot(2, len(consec_image), t + 1)
            plt.imshow(image, cmap="gray")
            plt.title("Image", fontsize=font_size)
            plt.axis("off")
            plt.subplot(2, len(consec_image), t + len(consec_image) + 1)
            plt.imshow(mask, cmap="gray")
            plt.title("GT Mask", fontsize=font_size)
            plt.axis("off")

        print(f"Sample {sample}:")

        # print annotations
        for t in range(len(consec_cal)):
            print("-" * 50)
            print(f"t = {t} annotations")
            print(f"ctr_x: {consec_cal[t, 0].item():.2f}, ctr_y: {consec_cal[t, 1].item():.2f}")
            print(f"angle: {consec_cal[t, 2].item():.4f}, length: {consec_cal[t, 3].item():.2f}")
            print(f"endpoint 1: ({consec_endpoint[t, 0].item():.2f}, {consec_endpoint[t, 1].item():.2f})")
            print(f"endpoint 2: ({consec_endpoint[t, 2].item():.2f}, {consec_endpoint[t, 3].item():.2f})")
            print(f"label: {consec_label[t].item()}")
        print("-" * 50)

        plt.show(block=False)
        plt.pause(15)
        plt.close()

        # break after showing the first 2 samples
        if sample == max_samples - 1:
            break


# visualize some batches of segmentation results
def show_seg_preds_only(consec_images, consec_masks, pred_masks, max_samples=2, figsize=(8, 8), font_size=10):
    # Show T consecutive images in a batch
    for sample in range(consec_images.shape[0]):
        consec_image = consec_images[sample]
        consec_mask = consec_masks[sample]
        pred_mask = pred_masks[sample]

        plt.figure(figsize=figsize)
        for t in range(len(consec_image)):
            # image
            image = consec_image[t].numpy()
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

            # mask
            mask = consec_mask[t].numpy()
            mask = (mask * 255).astype(np.uint8)

            # predicted mask (last frame)
            if t == len(consec_image) - 1:
                vis_pred_mask = pred_mask[0]  # [1, H, W] -> [H, W]
                # threshold the mask
                vis_pred_mask[vis_pred_mask <= 0.5] = 0
                vis_pred_mask[vis_pred_mask > 0.5] = 1
                # convert to dtype
                vis_pred_mask = vis_pred_mask.numpy()
                vis_pred_mask = (vis_pred_mask * 255).astype(np.uint8)
                vis_pred_mask = cv2.cvtColor(vis_pred_mask, cv2.COLOR_GRAY2BGR)
            else:
                # create a black image if not the last frame
                vis_pred_mask = np.zeros((consec_image[0].shape[-2], consec_image[0].shape[-1]))
                vis_pred_mask = (vis_pred_mask * 255).astype(np.uint8)
                vis_pred_mask = cv2.cvtColor(vis_pred_mask, cv2.COLOR_GRAY2BGR)

            # show image
            plt.subplot(3, len(consec_image), t + 1)
            plt.imshow(image, cmap="gray")
            plt.title("Image", fontsize=font_size)
            plt.axis("off")
            plt.subplot(3, len(consec_image), t + len(consec_image) + 1)
            plt.imshow(mask, cmap="gray")
            plt.title("GT Mask", fontsize=font_size)
            plt.axis("off")
            plt.subplot(3, len(consec_image), t + 2 * len(consec_image) + 1)
            plt.imshow(vis_pred_mask, cmap="gray")
            if t == len(consec_image) - 1:
                plt.title(f"Pred Mask", fontsize=font_size)
            else:
                plt.title("N/A", fontsize=font_size)
            plt.axis("off")

        plt.show(block=False)
        plt.pause(10)
        plt.close()

        # break after showing the first 2 samples
        if sample == max_samples - 1:
            break


# visualize some batches of image pairs (show T consecutive images and masks)
def show_preds_with_det_head(
    consec_images,
    consec_masks,
    pred_masks,
    pred_classifications,
    pred_regressions,
    anchors_pos,
    max_samples=2,
    topk=3,
    conf_thresh=0.1,
    with_aqe=False,
    figsize=(8, 8),
    font_size=10,
):
    # Show T consecutive images in a batch
    for sample in range(consec_images.shape[0]):
        consec_image = consec_images[sample]
        consec_mask = consec_masks[sample]
        pred_mask = pred_masks[sample]
        pred_cls = pred_classifications[sample]  # [num_total_anchors, num_classes]
        pred_reg = pred_regressions[sample]  # [num_total_anchors, 4]

        plt.figure(figsize=figsize)
        for t in range(len(consec_image)):
            # image
            image = consec_image[t].numpy()
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

            # mask
            mask = consec_mask[t].numpy()
            mask = (mask * 255).astype(np.uint8)

            # predicted mask and detections
            if t == len(consec_image) - 1:
                # predicted mask
                vis_pred_mask = pred_mask[0]
                vis_pred_mask[vis_pred_mask <= 0.5] = 0
                vis_pred_mask[vis_pred_mask > 0.5] = 1
                vis_pred_mask = vis_pred_mask.numpy()
                vis_pred_mask = (vis_pred_mask * 255).astype(np.uint8)
                # predicted detections
                vis_pred_mask = cv2.cvtColor(vis_pred_mask, cv2.COLOR_GRAY2BGR)
                # get the top-k detection endpoints
                topk_conf, topk_cls_id, topk_endpoints, topk_pred_cals = detect_postprocessing(
                    pred_cls, pred_reg, anchors_pos, conf_thresh=conf_thresh, topk=topk, with_aqe=with_aqe
                )
                # draw top-k endpoint
                for k in range(topk_conf.shape[0]):
                    endpoints = 按斜率滑動到裁剪範圍內(topk_endpoints[k], 0, 0, vis_pred_mask.shape[1], vis_pred_mask.shape[0])
                    x1, y1 = endpoints[0][0], endpoints[0][1]
                    x2, y2 = endpoints[1][0], endpoints[1][1]
                    x1, y1, x2, y2 = np.uint8(x1), np.uint8(y1), np.uint8(x2), np.uint8(y2)
                    print(x1, y1, x2, y2)
                    if k == 0:
                        color = (0, 255, 0)
                    elif k == 1:
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)
                    if not (x1 == x2 and y1 == y2):
                        vis_pred_mask = cv2.line(vis_pred_mask, (x1, y1), (x2, y2), color, 5)
                print(f"top-k conf: {topk_conf}")
                print(f"top-k cls id: {topk_cls_id}")
                print(f"top-k endpoints: {topk_endpoints}")
                print(f"top-k pred cals: {topk_pred_cals}")
            else:
                vis_pred_mask = np.zeros((consec_image[0].shape[-2], consec_image[0].shape[-1]))
                vis_pred_mask = (vis_pred_mask * 255).astype(np.uint8)
                vis_pred_mask = cv2.cvtColor(vis_pred_mask, cv2.COLOR_GRAY2BGR)

            # plot image
            plt.subplot(3, len(consec_image), t + 1)
            plt.imshow(image, cmap="gray")
            plt.title("Image", fontsize=font_size)
            plt.axis("off")
            plt.subplot(3, len(consec_image), t + len(consec_image) + 1)
            plt.imshow(mask, cmap="gray")
            plt.title("GT Mask", fontsize=font_size)
            plt.axis("off")
            plt.subplot(3, len(consec_image), t + 2 * len(consec_image) + 1)
            plt.imshow(vis_pred_mask, cmap="brg")
            if t == len(consec_image) - 1:
                plt.title(f"Prediction", fontsize=font_size)
            else:
                plt.title("N/A", fontsize=font_size)
            plt.axis("off")

        plt.show(block=False)
        plt.pause(10)
        plt.close()

        # break after showing the first 2 samples
        if sample == max_samples - 1:
            break


def detect_postprocessing(pred_cls, pred_reg, anchors_pos, conf_thresh=0.5, topk=3, with_aqe=False):
    """
    Post-processing for the detection head predictions
    Args:
        pred_cls (torch.Tensor): predicted class scores. shape [num_total_anchors, num_classes]
        pred_reg (torch.Tensor): predicted regression values. shape [num_total_anchors, 4]
        anchors_pos (torch.Tensor): anchor positions. shape [num_total_anchors, 4]
        theta_ref (str): reference angle format. "horizontal" or "left-top to right-bottom"
        conf_thresh (float): confidence threshold for the detection head
        topk (int): number of top-k anchors to keep
    Returns:
        topk_conf (torch.Tensor): top-k confidence scores. shape [k,]
        topk_cls_id (torch.Tensor): the predicted class ids for the top-k anchors. shape [k,]
        topk_endpoints (torch.Tensor): the predicted endpoints for the top-k anchors after confidence thresholding. shape [k, 4]
        topk_pred_cals (torch.Tensor): the predicted center, angle, length for the top-k anchors. shape [k, 4]
    """
    # assign each anchor to the class with the highest confidence score
    conf, cls_id = torch.max(pred_cls, dim=-1)  # both [num_total_anchors,]

    # get top-k anchors with the highest confidence scores
    topk_conf, topk_idx = torch.topk(conf, k=topk, dim=-1)  # both [k,]

    # get the corresponding class ids
    topk_cls_id = cls_id[topk_idx]  # [k,]

    # get the top-k anchor positions
    topk_anchors_pos = anchors_pos[topk_idx]  # [k, 4 or 5]

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
    # smooth L1 loss regression method
    # remember to modify the following code to get the predicted center, angle, length when the regression targets are modified !!!!!
    # get the predicted center, angle, length of each anchor

    # rescale the regression targets due to target scaling in loss calculation
    pred_reg[topk_idx, 0] = pred_reg[topk_idx, 0] / 2  # center x
    pred_reg[topk_idx, 1] = pred_reg[topk_idx, 1] / 2  # center y

    if with_aqe:  # AQE regression method
        pred_reg[topk_idx, 4] = pred_reg[topk_idx, 4] / 2  # length
    else:
        pred_reg[topk_idx, 3] = pred_reg[topk_idx, 3] / 2  # length

    # get the predicted center, angle, length for the top-k anchors
    topk_pred_ctr_x = pred_reg[topk_idx, 0] * topk_anchors_width + topk_anchors_ctr_x  # [k,]
    topk_pred_ctr_y = pred_reg[topk_idx, 1] * topk_anchors_height + topk_anchors_ctr_y  # [k,]
    topk_pred_theta = pred_reg[topk_idx, 2]  # [k,]

    if with_aqe:  # AQE regression method
        topk_pred_sigma = pred_reg[topk_idx, 3]  # [k,]
        print(topk_pred_sigma)
        topk_pred_length = torch.exp(pred_reg[topk_idx, 4]) * topk_anchors_length  # [k,]
    else:
        topk_pred_length = torch.exp(pred_reg[topk_idx, 3]) * topk_anchors_length  # [k,]

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

    # apply confidence thresholding
    topk_endpoints = torch.where(topk_conf.unsqueeze(-1).repeat(1, 4) > conf_thresh, topk_endpoints, torch.zeros_like(topk_endpoints))

    return topk_conf, topk_cls_id, topk_endpoints, topk_pred_cals
