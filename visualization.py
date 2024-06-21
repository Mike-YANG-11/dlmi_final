# Visualization Functions

import cv2

import matplotlib.pyplot as plt

import numpy as np

import torch

import tqdm

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
    topk=1,
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
                vis_pred_mask = torch.zeros_like(pred_mask[0])
                vis_pred_mask = vis_pred_mask.numpy().astype(np.uint8)

                # predicted detection results
                vis_pred_mask = cv2.cvtColor(vis_pred_mask, cv2.COLOR_GRAY2BGR)

                # get the top-k detection endpoints
                topk_score, topk_id, topk_endpoints, topk_pred_cals, topk_pred_sigma = detect_postprocessing(
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
                    x1, y1, x2, y2 = np.uint8(topk_endpoints[k])
                    if k == 0:
                        color = (255, 0, 0)
                    elif k == 1:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    if not (x1 == x2 and y1 == y2):
                        vis_pred_mask = cv2.line(vis_pred_mask, (x1, y1), (x2, y2), color, 5)
                print(f"top-k score: {topk_score}")
                print(f"top-k id: {topk_id}")
                print(f"top-k endpoints: {topk_endpoints}")
                print(f"top-k pred cals: {topk_pred_cals}")
                print(f"top-k pred sigma: {topk_pred_sigma}")

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


def save_video(model, device, video_loader, image_size, buffer_num_sample, time_window, anchors_pos):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("1320_hard.mp4", fourcc, 15.0, (image_size * 3 + 10, image_size))  # [W, H] !!!!!
    model.to(device)
    model.eval()

    with torch.no_grad():
        for step, buffer in enumerate(tqdm(video_loader)):
            for t in range(buffer_num_sample):  # iterate over the buffer to get samples

                # Initialize the video frame ready to be written ([H, W, 3] !!!)
                video_frame_image = np.zeros((image_size, image_size * 3 + 10, 3)).astype(np.uint8)

                original_image = buffer["images"][:, t + 2 : t + time_window, :, :].squeeze()  # [H, W]
                vis_image = buffer["images"][:, t + 2 : t + time_window, :, :].squeeze()  # [H, W]
                gt_mask = buffer["masks"][:, t + 2 : t + time_window, :, :].squeeze()  # [H, W]

                # Convert the images to numpy arrays
                original_image = original_image.cpu().numpy()
                original_image = (original_image * 255).astype(np.uint8)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                original_image = cv2.putText(original_image, "Input", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Convert the images to numpy arrays
                vis_image = vis_image.cpu().numpy()
                vis_image = (vis_image * 255).astype(np.uint8)
                gt_mask = gt_mask.cpu().numpy()
                gt_mask = (gt_mask * 255).astype(np.uint8)

                # draw the gt mask on the original image (red)
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
                vis_image[gt_mask == 255] = [0, 255, 0]

                # forward pass
                input_images = buffer["images"][:, t : t + time_window, :, :].to(device)  # [1, T, H, W]
                pred_masks, pred_classifications, pred_regressions = model(input_images)
                # [1, 1, H, W], [1, num_total_anchors, num_classes], [1, num_total_anchors, 4]

                # ------------------------------------------------------------------
                # segmentation head
                # ------------------------------------------------------------------
                # draw the predicted mask countour on the original image (blue)
                pred_masks = pred_masks.squeeze().cpu().numpy()
                pred_masks = np.where(pred_masks > 0.5, 255, 0).astype(np.uint8)
                # dilate the mask to make the contour thicker
                kernel = np.ones((5, 5), np.uint8)
                pred_masks = cv2.dilate(pred_masks, kernel, iterations=1)
                contours, hierarchy = cv2.findContours(pred_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                vis_mask_image = cv2.drawContours(vis_image.copy(), contours, -1, (0, 20, 200), 2)

                # show seg text on the image
                vis_mask_image = cv2.putText(vis_mask_image, "Seg", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 20, 200), 2)

                # ------------------------------------------------------------------
                # detection head
                # ------------------------------------------------------------------
                # detection head post-processing
                pred_cls = pred_classifications.squeeze(0)  # [num_total_anchors, num_classes]
                pred_reg = pred_regressions.squeeze(0)  # [num_total_anchors, 5]

                # get the top-k detection endpoints
                topk_score, topk_id, topk_endpoints, topk_pred_cals, topk_pred_sigma = detect_postprocessing(
                    pred_cls,
                    pred_reg,
                    anchors_pos,
                    image_size,
                    image_size,
                    conf_thresh=0.1,
                    topk=1,
                    with_aqe=True,
                )
                top1_endpoints = topk_endpoints[0]
                top1_pred_sigma = topk_pred_sigma[0]
                topk_score = topk_score[0]

                if topk_score < 0.1:
                    top1_pred_sigma = 1

                if top1_endpoints.sum() == 0:
                    vis_detect_image = vis_image
                else:
                    x1, y1, x2, y2 = np.uint8(top1_endpoints.cpu())
                    detect_line_mask = np.zeros((image_size, image_size)).astype(np.uint8)
                    detect_line_mask = cv2.line(detect_line_mask, (x1, y1), (x2, y2), 255, 8)
                    contours, hierarchy = cv2.findContours(detect_line_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    vis_detect_image = cv2.drawContours(vis_image, contours, -1, (0, 255, 255), 2)

                # show the confidence score and sigma on the image
                vis_detect_image = cv2.putText(
                    vis_detect_image, f"Score: {topk_score:.2f}", (121, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )
                vis_detect_image = cv2.putText(
                    vis_detect_image, f"AQ: {(1 - top1_pred_sigma):.2f}", (145, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )
                vis_detect_image = cv2.putText(vis_detect_image, "Det", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # ------------------------------------------------------------------

                # write the video frame
                video_frame_image[:, :image_size, :] = original_image
                video_frame_image[:, image_size : image_size + 5, :] = 150
                video_frame_image[:, image_size + 5 : image_size * 2 + 5, :] = vis_mask_image
                video_frame_image[:, image_size * 2 + 5 : image_size * 2 + 10, :] = 150
                video_frame_image[:, image_size * 2 + 10 :, :] = vis_detect_image
                video_writer.write(video_frame_image)

    video_writer.release()
