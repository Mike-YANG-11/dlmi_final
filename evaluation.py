# Evaluation Functions

import numpy as np

import torch

from sklearn.decomposition import PCA

from tqdm import tqdm


# Segmentation Dice Score Function
def seg_dice_score(preds, masks):
    smooth = 1  # avoid division by zero
    preds = preds.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
    masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
    preds = preds > 0.5
    masks = masks > 0

    intersection = (preds & masks).float().sum()
    dice = (2 * intersection + smooth) / (preds.float().sum() + masks.float().sum() + smooth)

    return dice


# Segmentation IoU Score Function
def seg_iou_score(preds, masks):
    smooth = 1  # avoid division by zero
    preds = preds.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
    masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
    preds = preds > 0.5
    masks = masks > 0

    intersection = (preds & masks).float().sum()
    union = (preds | masks).float().sum()
    iou = (intersection + smooth) / (union + smooth)

    return iou


# Recall Score Function
# def recall_score(preds, masks):
#     smooth = 1  # avoid division by zero
#     preds = preds.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
#     masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
#     preds = preds > 0.5
#     masks = masks > 0

#     intersection = (preds & masks).float().sum()
#     recall = (intersection + smooth) / (masks.float().sum() + smooth)

#     return recall


# Precision Score Function
# def precision_score(preds, masks):
#     smooth = 1  # avoid division by zero
#     preds = preds.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
#     masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
#     preds = preds > 0.5
#     masks = masks > 0

#     intersection = (preds & masks).float().sum()
#     precision = (intersection + smooth) / (preds.float().sum() + smooth)

#     return precision


# EA Score by <Deep Hough Transform for Semantic Line Detection>
# https://github.com/Hanqer/deep-hough-transform/blob/master/metric.py
class Line(object):
    def __init__(self, coordinates=[0, 0, 1, 1]):
        """
        coordinates: [y0, x0, y1, x1]
        """
        assert isinstance(coordinates, list)
        assert len(coordinates) == 4
        assert coordinates[0] != coordinates[2] or coordinates[1] != coordinates[3]
        self.__coordinates = coordinates

    @property
    def coord(self):
        return self.__coordinates

    @property
    def length(self):
        start = np.array(self.coord[:2])
        end = np.array(self.coord[2::])
        return np.sqrt(((start - end) ** 2).sum())

    def angle(self):
        y0, x0, y1, x1 = self.coord
        if x0 == x1:
            return -np.pi / 2
        return np.arctan((y0 - y1) / (x0 - x1))

    def rescale(self, rh, rw):
        coor = np.array(self.__coordinates)
        r = np.array([rh, rw, rh, rw])
        self.__coordinates = np.round(coor * r).astype(np.int).tolist()

    def __repr__(self):
        return str(self.coord)


def sa_metric(angle_p, angle_g):
    d_angle = np.abs(angle_p - angle_g)
    d_angle = min(d_angle, np.pi - d_angle)
    d_angle = d_angle * 2 / np.pi
    return max(0, (1 - d_angle)) ** 2


def se_metric(coord_p, coord_g, size):
    c_p = [(coord_p[0] + coord_p[2]) / 2, (coord_p[1] + coord_p[3]) / 2]
    c_g = [(coord_g[0] + coord_g[2]) / 2, (coord_g[1] + coord_g[3]) / 2]
    d_coord = np.abs(c_p[0] - c_g[0]) ** 2 + np.abs(c_p[1] - c_g[1]) ** 2
    d_coord = np.sqrt(d_coord) / max(size[0], size[1])
    return max(0, (1 - d_coord)) ** 2


def len_metric(coord_p, coord_g, size):
    len_p = np.sqrt((coord_p[0] + coord_p[2]) ** 2 + (coord_p[1] + coord_p[3]) ** 2)
    len_g = np.sqrt((coord_g[0] + coord_g[2]) ** 2 + (coord_g[1] + coord_g[3]) ** 2)  ## bbox length for bbox branch, seg length for seg branch?
    len_bias = abs(len_g - len_p) / np.sqrt(size[0] ** 2 + size[1] ** 2)
    return max(0, (1 - len_bias)) ** 2


def mask2Line(idx, pca):
    # Convert the binary mask to 2D point coordinate list
    mask_2d = np.stack([idx[0], idx[1]], axis=1)

    # Fit PCA on the target mask points
    pca.fit(mask_2d)
    mask_1d = pca.transform(mask_2d)
    mask_2d_new = pca.inverse_transform(mask_1d)
    mask_2d_new = sorted(mask_2d_new, key=lambda x: x[0])
    mask_line_end_points = [
        int(mask_2d_new[0][1]),
        int(mask_2d_new[0][0]),
        int(mask_2d_new[-1][1]),
        int(mask_2d_new[-1][0]),
    ]
    if mask_line_end_points[0] == mask_line_end_points[2]:
        mask_line_end_points[2] += 1
    if mask_line_end_points[1] == mask_line_end_points[3]:
        mask_line_end_points[3] += 1
    line = Line(mask_line_end_points)
    return line


def eal_metric(pred_idx, mask_idx, size):
    # Find the line from the points in the binary mask using PCA
    pca = PCA(n_components=1)

    # Find the line from the points in the binary mask using PCA
    pred_line = mask2Line(pred_idx, pca)
    mask_line = mask2Line(mask_idx, pca)

    # Calculate the EA score
    se = se_metric(pred_line.coord, mask_line.coord, size=size)
    sa = sa_metric(pred_line.angle(), mask_line.angle())
    sl = len_metric(pred_line.coord, mask_line.coord, size=size)
    return sa * se * sl


def ea_metric(pred_idx, mask_idx, size):
    # Convert the binary mask to 2D point coordinate list
    pred_2d = np.stack([pred_idx[0], pred_idx[1]], axis=1)
    mask_2d = np.stack([mask_idx[0], mask_idx[1]], axis=1)

    # Find the line from the points in the binary mask using PCA
    pca = PCA(n_components=1)

    # Fit PCA on the predicted mask points
    pca.fit(pred_2d)
    pred_1d = pca.transform(pred_2d)
    pred_2d_new = pca.inverse_transform(pred_1d)
    pred_2d_new = sorted(pred_2d_new, key=lambda x: x[0])
    pred_line_end_points = [
        int(pred_2d_new[0][1]),
        int(pred_2d_new[0][0]),
        int(pred_2d_new[-1][1]),
        int(pred_2d_new[-1][0]),
    ]
    if pred_line_end_points[0] == pred_line_end_points[2]:
        pred_line_end_points[2] += 1
    if pred_line_end_points[1] == pred_line_end_points[3]:
        pred_line_end_points[3] += 1
    pred_line = Line(pred_line_end_points)

    # Fit PCA on the target mask points
    pca.fit(mask_2d)
    mask_1d = pca.transform(mask_2d)
    mask_2d_new = pca.inverse_transform(mask_1d)
    mask_2d_new = sorted(mask_2d_new, key=lambda x: x[0])
    mask_line_end_points = [
        int(mask_2d_new[0][1]),
        int(mask_2d_new[0][0]),
        int(mask_2d_new[-1][1]),
        int(mask_2d_new[-1][0]),
    ]
    if mask_line_end_points[0] == mask_line_end_points[2]:
        mask_line_end_points[2] += 1
    if mask_line_end_points[1] == mask_line_end_points[3]:
        mask_line_end_points[3] += 1
    mask_line = Line(mask_line_end_points)

    # Calculate the EA score
    se = se_metric(pred_line.coord, mask_line.coord, size=size)
    sa = sa_metric(pred_line.angle(), mask_line.angle())
    return sa * se


def line_evaluate(preds, masks):
    preds = preds.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
    masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]

    # Calculate the EA score for each sample in the batch
    ea_score_list = []
    eal_score_list = []
    p_count = 0
    n_count = 0
    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0

    for i in range(preds.shape[0]):
        pred = preds[i].cpu().numpy()
        mask = masks[i].cpu().numpy()

        # Convert the predicted mask to coordinates
        pred_idx = np.where(pred > 0.5)
        mask_idx = np.where(mask > 0)

        # count ground truth positive and negative samples number
        if len(mask_idx[1]) > 0:
            p_count += 1
        else:
            n_count += 1

        # count predicted tp, fp, tn, fn samples number
        pixel_num_threshold = 1
        if (
            len(pred_idx[0]) > pixel_num_threshold and len(mask_idx[1]) > pixel_num_threshold
        ):  # Predicted mask and ground truth mask are both not empty
            ea_score = ea_metric(pred_idx, mask_idx, size=pred.shape)
            ea_score_list.append(ea_score)
            eal_score = eal_metric(pred_idx, mask_idx, size=pred.shape)
            eal_score_list.append(eal_score)
            if ea_score >= 0.9:
                tp_count += 1
            else:
                fn_count += 1
        elif (
            len(pred_idx[0]) > pixel_num_threshold and len(mask_idx[1]) <= pixel_num_threshold
        ):  # Predicted mask is not empty but ground truth mask are empty
            fp_count += 1
        elif (
            len(pred_idx[0]) <= pixel_num_threshold and len(mask_idx[1]) > pixel_num_threshold
        ):  # Predicted mask is empty but ground truth mask are not empty
            fn_count += 1
        elif (
            len(pred_idx[0]) <= pixel_num_threshold and len(mask_idx[1]) <= pixel_num_threshold
        ):  # Predicted mask and ground truth mask are both empty
            tn_count += 1

    return np.mean(ea_score_list), np.mean(eal_score_list), p_count, n_count, tp_count, fp_count, tn_count, fn_count


# Evaluation Function
def evaluate(model, device, loader, seg_focal_loss, seg_dice_loss, model_name, det_loss=None, anchors_pos=None):
    print("Evaluating the model...")
    model.eval()
    model.to(device)

    # Initialize variables to store the total loss and score
    total_loss = 0.0
    total_seg_focal_loss = 0.0
    total_seg_dice_loss = 0.0
    total_seg_dice_score = 0.0
    total_seg_iou_score = 0.0
    if model_name == "Video-Retina-UNETR":
        total_det_cls_loss = 0.0
        total_det_reg_loss = 0.0

    # Initialize variables to store the EA score and line metrics
    total_seg_ea_score = 0.0
    total_seg_eal_score = 0.0
    total_p_count = 0
    total_n_count = 0
    total_seg_tp_count = 0
    total_seg_fp_count = 0
    total_seg_tn_count = 0
    total_seg_fn_count = 0

    with torch.no_grad():
        for step, samples in enumerate(tqdm(loader)):
            # Image data
            images = samples["images"].to(device)  # [N, T, H, W]

            # Segmentation ground truth masks
            masks = samples["masks"].to(device)  # [N, T, H, W]

            # Detection ground truth annotations
            if model_name == "Video-Retina-UNETR":
                cals = samples["cals"].to(device)  # [N, T, 4]
                labels = samples["labels"].to(device)  # [N, T]
                labels = labels.unsqueeze(-1)  # [N, T, 1]
                annotations = torch.cat([cals, labels], dim=-1)  # [N, T, 5]

            # Forward pass
            if model_name == "Video-Retina-UNETR":
                pred_masks, classifications, regressions = model(images)
                # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4]
            else:
                pred_masks = model(images)  # [N, 1, H, W]

            # Calculate loss (use the last frame as the target mask)
            masks = masks[:, -1, :, :].unsqueeze(1)  # [N, 1, H, W]
            fl = seg_focal_loss(pred_masks, masks)
            dl = seg_dice_loss(pred_masks, masks)
            if model_name == "Video-Retina-UNETR":  # with the detection head
                annotations = annotations[:, -1, :].unsqueeze(1)  # [N, 1, 5]
                cl, rl = det_loss(classifications, regressions, anchors_pos, annotations)

            # Calculate total loss
            loss = fl + dl
            if model_name == "Video-Retina-UNETR":  # with the detection head
                loss = loss + cl + rl

            # Calculate the segmentation Dice score & IoU score
            seg_dscore = seg_dice_score(pred_masks, masks)
            seg_iscore = seg_iou_score(pred_masks, masks)

            # Calculate EA score and line metrics
            mean_seg_ea_score, mean_seg_eal_score, p_count, n_count, seg_tp_count, seg_fp_count, seg_tn_count, seg_fn_count = line_evaluate(
                pred_masks, masks
            )

            # Accumulate loss & score
            total_loss += loss.item()
            total_seg_focal_loss += fl.item()
            total_seg_dice_loss += dl.item()
            total_seg_dice_score += seg_dscore.item()
            total_seg_iou_score += seg_iscore.item()
            if model_name == "Video-Retina-UNETR":
                total_det_cls_loss += cl.item()
                total_det_reg_loss += rl.item()

            # Accumulate EA score and line metrics
            total_seg_ea_score += mean_seg_ea_score
            total_seg_eal_score += mean_seg_eal_score
            total_p_count += p_count
            total_n_count += n_count
            total_seg_tp_count += seg_tp_count
            total_seg_fp_count += seg_fp_count
            total_seg_tn_count += seg_tn_count
            total_seg_fn_count += seg_fn_count

    # Calculate the line metrics
    seg_needle_recall_score = total_seg_tp_count / (total_seg_tp_count + total_seg_fn_count + 1e-6)
    seg_needle_precision_score = total_seg_tp_count / (total_seg_tp_count + total_seg_fp_count + 1e-6)
    seg_needle_specificity_score = total_seg_tn_count / (total_seg_tn_count + total_seg_fp_count + 1e-6)

    results = {
        "Loss": total_loss / len(loader),
        "Segmentation Focal Loss": total_seg_focal_loss / len(loader),
        "Segmentation Dice Loss": total_seg_dice_loss / len(loader),
        "Segmentation Dice Score": total_seg_dice_score / len(loader),
        "Segmentation IoU Score": total_seg_iou_score / len(loader),
        "Segmentation Needle EA Score": total_seg_ea_score / len(loader),
        "Segmentation Needle EAL Score": total_seg_eal_score / len(loader),
        "Needle Positive Count": total_p_count,
        "Needle Negative Count": total_n_count,
        "Segmentation Needle TP Count": total_seg_tp_count,
        "Segmentation Needle FP Count": total_seg_fp_count,
        "Segmentation Needle TN Count": total_seg_tn_count,
        "Segmentation Needle FN Count": total_seg_fn_count,
        "Segmentation Needle Recall Score": seg_needle_recall_score,
        "Segmentation Needle Precision Score": seg_needle_precision_score,
        "Segmentation Needle Specificity Score": seg_needle_specificity_score,
    }

    if model_name == "Video-Retina-UNETR":
        results["Detection Classification Loss"] = total_det_cls_loss / len(loader)
        results["Detection Regression Loss"] = total_det_reg_loss / len(loader)

    return results


def results_dictioanry(model_name, type):
    """
    Construct a dictionary to store the results for the model.
    """
    if type == "best_val_results":
        results_dict = {
            "Loss": float("inf"),
            "Segmentation Focal Loss": float("inf"),
            "Segmentation Dice Loss": float("inf"),
            "Segmentation Dice Score": 0.0,
            "Segmentation IoU Score": 0.0,
            "Segmentation Needle EA Score": 0.0,
            "Segmentation Needle EAL Score": 0.0,
            "Needle Positive Count": 0,
            "Needle Negative Count": 0,
            "Segmentation Needle TP Count": 0,
            "Segmentation Needle FP Count": 0,
            "Segmentation Needle TN Count": 0,
            "Segmentation Needle FN Count": 0,
            "Segmentation Needle Recall Score": 0.0,
            "Segmentation Needle Precision Score": 0.0,
            "Segmentation Needle Specificity Score": 0.0,
        }

        # Add detection metrics for Video-Retina-UNETR
        if model_name == "Video-Retina-UNETR":
            results_dict["Detection Classification Loss"] = float("inf")
            results_dict["Detection Regression Loss"] = float("inf")

    elif type == "running_results":
        results_dict = {
            "Loss": 0.0,
            "Segmentation Focal Loss": 0.0,
            "Segmentation Dice Loss": 0.0,
            "Segmentation Dice Score": 0.0,
            "Segmentation IoU Score": 0.0,
        }

        # Add detection metrics for Video-Retina-UNETR
        if model_name == "Video-Retina-UNETR":
            results_dict["Detection Classification Loss"] = 0.0
            results_dict["Detection Regression Loss"] = 0.0

    return results_dict
