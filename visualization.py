# Visualization Functions

import numpy as np

import matplotlib.pyplot as plt


# visualize some batches of image pairs (show T consecutive images and masks)
def show_image_pairs(consec_images, consec_masks, preds=None, figsize=(8, 8)):
    # Show T consecutive images in a batch
    for sample in range(consec_images.shape[0]):
        consec_image = consec_images[sample]
        consec_mask = consec_masks[sample]
        pred = None if preds is None else preds[sample]

        plt.figure(figsize=figsize)
        for t in range(len(consec_image)):
            # image
            image = consec_image[t].numpy()
            image[image <= 0] = 0
            image[image >= 1] = 1
            image = (image * 255).astype(np.uint8)

            # mask
            mask = consec_mask[t].numpy()
            mask = (mask * 255).astype(np.uint8)

            # predicted mask (last frame) if available
            if preds is not None:
                if t == len(consec_image) - 1:
                    pred_mask = pred[0]
                    pred_mask[pred_mask <= 0.5] = 0
                    pred_mask[pred_mask > 0.5] = 1
                    pred_mask = pred_mask.numpy()
                else:
                    pred_mask = np.zeros((consec_image[0].shape[-2], consec_image[0].shape[-1]))
                pred_mask = (pred_mask * 255).astype(np.uint8)

            # show image
            if preds is None:
                plt.subplot(2, len(consec_image), t + 1)
                plt.imshow(image, cmap="gray")
                plt.title("Image")
                plt.axis("off")
                plt.subplot(2, len(consec_image), t + len(consec_image) + 1)
                plt.imshow(mask, cmap="gray")
                plt.title("GT Mask")
                plt.axis("off")
            else:
                plt.subplot(3, len(consec_image), t + 1)
                plt.imshow(image, cmap="gray")
                plt.title("Image")
                plt.axis("off")
                plt.subplot(3, len(consec_image), t + len(consec_image) + 1)
                plt.imshow(mask, cmap="gray")
                plt.title("GT Mask")
                plt.axis("off")
                plt.subplot(3, len(consec_image), t + 2 * len(consec_image) + 1)
                plt.imshow(pred_mask, cmap="gray")
                if t == len(consec_image) - 1:
                    plt.title("Predicted Mask")
                else:
                    plt.title("N/A")
                plt.axis("off")

        plt.show(block=False)
        plt.pause(10)
        plt.close()

        # break after showing the first 2 samples
        if sample == 1:
            break
