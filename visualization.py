# Visualization Functions

import cv2

import math

import numpy as np

import matplotlib.pyplot as plt

import torch

from post_processing import detect_postprocessing


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
    conf_thresh=0.2,
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
                vis_pred_mask = torch.zeros_like(pred_mask[0])
                vis_pred_mask = vis_pred_mask.numpy().astype(np.uint8)

                # predicted detection results
                vis_pred_mask = cv2.cvtColor(vis_pred_mask, cv2.COLOR_GRAY2BGR)

                # get the top-k detection endpoints
                topk_score, topk_endpoints, topk_pred_cals = detect_postprocessing(
                    pred_cls,
                    pred_reg,
                    anchors_pos,
                    vis_pred_mask.shape[1],
                    vis_pred_mask.shape[0],
                    conf_thresh=conf_thresh,
                    topk=topk,
                    with_aqe=with_aqe,
                )
                # draw top-k endpoint (red: top-1, green: top-2, blue: top-3)
                for k in range(topk_score.shape[0]):
                    x1, y1, x2, y2 = topk_endpoints[k].int()
                    x1, y1, x2, y2 = np.uint8(x1), np.uint8(y1), np.uint8(x2), np.uint8(y2)
                    print(x1, y1, x2, y2)
                    if k == 0:
                        color = (255, 0, 0)
                    elif k == 1:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    if not (x1 == x2 and y1 == y2):
                        vis_pred_mask = cv2.line(vis_pred_mask, (x1, y1), (x2, y2), color, 5)
                print(f"top-k score: {topk_score}")
                print(f"top-k endpoints: {topk_endpoints}")
                print(f"top-k pred cals: {topk_pred_cals}")

                # predicted mask
                vis_pred_mask[pred_mask[0] > 0.5, :] = 255
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
