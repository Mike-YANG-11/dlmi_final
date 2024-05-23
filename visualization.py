# Visualization Functions

import cv2

import numpy as np

import matplotlib.pyplot as plt


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
def show_seg_preds_only(consec_images, consec_masks, preds, max_samples=2, figsize=(8, 8), font_size=10):
    # Show T consecutive images in a batch
    for sample in range(consec_images.shape[0]):
        consec_image = consec_images[sample]
        consec_mask = consec_masks[sample]
        pred = preds[sample]

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
                pred_mask = pred[0]  # [1, H, W] -> [H, W]
                # threshold the mask
                pred_mask[pred_mask <= 0.5] = 0
                pred_mask[pred_mask > 0.5] = 1
                # convert to dtype
                pred_mask = pred_mask.numpy()
                pred_mask = (pred_mask * 255).astype(np.uint8)
                pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
            else:
                # create a black image if not the last frame
                pred_mask = np.zeros((consec_image[0].shape[-2], consec_image[0].shape[-1]))
                pred_mask = (pred_mask * 255).astype(np.uint8)
                pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

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
            plt.imshow(pred_mask, cmap="gray")
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
# def show_preds_with_det_head(
#     consec_images, consec_masks, preds=None, endpoints_id_0=None, endpoints_id_1=None, max_cls_score=None, figsize=(8, 8), font_size=10
# ):
#     # Show T consecutive images in a batch
#     for sample in range(consec_images.shape[0]):
#         consec_image = consec_images[sample]
#         consec_mask = consec_masks[sample]
#         pred = None if preds is None else preds[sample]
#         endpoint_id_0 = None if endpoints_id_0 is None else endpoints_id_0[sample]
#         endpoint_id_1 = None if endpoints_id_1 is None else endpoints_id_1[sample]
#         max_cls_conf = None if max_cls_score is None else max_cls_score[sample]

#         plt.figure(figsize=figsize)
#         for t in range(len(consec_image)):
#             # image
#             image = consec_image[t].numpy()
#             image = np.clip(image, 0, 1)
#             image = (image * 255).astype(np.uint8)

#             # mask
#             mask = consec_mask[t].numpy()
#             mask = (mask * 255).astype(np.uint8)

#             # predicted mask (last frame) if available
#             if preds is not None:
#                 if t == len(consec_image) - 1:
#                     pred_mask = pred[0]
#                     pred_mask[pred_mask <= 0.5] = 0
#                     pred_mask[pred_mask > 0.5] = 1
#                     pred_mask = pred_mask.numpy()
#                     pred_mask = (pred_mask * 255).astype(np.uint8)
#                     pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
#                     # draw cls 0 endpoints
#                     endpoint_id_0 = endpoint_id_0.numpy()
#                     x1 = np.clip(endpoint_id_0[0], 0, pred_mask.shape[1]).astype(np.uint8)
#                     y1 = np.clip(endpoint_id_0[1], 0, pred_mask.shape[0]).astype(np.uint8)
#                     x2 = np.clip(endpoint_id_0[2], 0, pred_mask.shape[1]).astype(np.uint8)
#                     y2 = np.clip(endpoint_id_0[3], 0, pred_mask.shape[0]).astype(np.uint8)
#                     pred_mask = cv2.line(pred_mask, (x1, y1), (x2, y2), (0, 255, 0), 5)
#                     # draw cls 1 endpoints
#                     endpoint_id_1 = endpoint_id_1.numpy()
#                     x1 = np.clip(endpoint_id_1[0], 0, pred_mask.shape[1]).astype(np.uint8)
#                     y1 = np.clip(endpoint_id_1[1], 0, pred_mask.shape[0]).astype(np.uint8)
#                     x2 = np.clip(endpoint_id_1[2], 0, pred_mask.shape[1]).astype(np.uint8)
#                     y2 = np.clip(endpoint_id_1[3], 0, pred_mask.shape[0]).astype(np.uint8)
#                     pred_mask = cv2.line(pred_mask, (x1, y1), (x2, y2), (0, 0, 255), 5)
#                 else:
#                     pred_mask = np.zeros((consec_image[0].shape[-2], consec_image[0].shape[-1]))
#                     pred_mask = (pred_mask * 255).astype(np.uint8)
#                     pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

#             # show image
#             if preds is None:
#                 plt.subplot(2, len(consec_image), t + 1)
#                 plt.imshow(image, cmap="gray")
#                 plt.title("Image", fontsize=font_size)
#                 plt.axis("off")
#                 plt.subplot(2, len(consec_image), t + len(consec_image) + 1)
#                 plt.imshow(mask, cmap="gray")
#                 plt.title("GT Mask", fontsize=font_size)
#                 plt.axis("off")
#             else:
#                 plt.subplot(3, len(consec_image), t + 1)
#                 plt.imshow(image, cmap="gray")
#                 plt.title("Image", fontsize=font_size)
#                 plt.axis("off")
#                 plt.subplot(3, len(consec_image), t + len(consec_image) + 1)
#                 plt.imshow(mask, cmap="gray")
#                 plt.title("GT Mask", fontsize=font_size)
#                 plt.axis("off")
#                 plt.subplot(3, len(consec_image), t + 2 * len(consec_image) + 1)
#                 plt.imshow(pred_mask, cmap="brg")
#                 if t == len(consec_image) - 1:
#                     plt.title(f"{max_cls_conf[0]:.2f}, {max_cls_conf[1]:.2f}", fontsize=font_size)
#                 else:
#                     plt.title("N/A", fontsize=font_size)
#                 plt.axis("off")

#         plt.show(block=False)
#         plt.pause(10)
#         plt.close()

#         # break after showing the first 2 samples
#         if sample == 1:
#             break
