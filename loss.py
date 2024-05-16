# loss functions

import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        # flatten prediction and target tensors
        preds = preds.reshape(-1)  # [N, 1, H, W] -> [N*H*W]
        targets = targets.reshape(-1)  # [N, 1, H, W] -> [N*H*W]

        alpha_factor = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        # compute binary cross-entropy
        bce_loss = F.binary_cross_entropy(preds, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, smooth=1e-6):
        # flatten prediction and target tensors
        preds = preds.reshape(-1)  # [N, 1, H, W] -> [N*H*W]
        targets = targets.reshape(-1)  # [N, 1, H, W] -> [N*H*W]

        intersection = (preds * targets).sum()
        dice_coeff = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

        return 1 - dice_coeff


# Hausdorff Distance Loss
class HDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0):
        super(HDTLoss, self).__init__()
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
