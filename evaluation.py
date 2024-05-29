# Evaluation Functions

import numpy as np

import torch

from sklearn.decomposition import PCA

from tqdm import tqdm

from post_processing import detect_postprocessing


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


def seg_eal_metric(pred_idx, mask_idx, size, without_len=False):
    # Find the line from the points in the binary mask using PCA
    pca = PCA(n_components=1)

    # Find the line from the points in the binary mask using PCA
    pred_line = mask2Line(pred_idx, pca)
    mask_line = mask2Line(mask_idx, pca)

    # Calculate the EA score
    se = se_metric(pred_line.coord, mask_line.coord, size=size)
    sa = sa_metric(pred_line.angle(), mask_line.angle())
    if without_len:
        return sa * se

    # Calculate the EAL score
    sl = len_metric(pred_line.coord, mask_line.coord, size=size)
    return sa * se * sl


def det_eal_metric(pred_endpoints, annotation, size, without_len=False):
    pred_endpoints = pred_endpoints.cpu()
    annotation = annotation.cpu()

    # get the predicted line
    pred_x1, pred_y1, pred_x2, pred_y2 = np.int_(pred_endpoints)
    if pred_x1 == pred_x2:
        pred_x2 += 1
    if pred_y1 == pred_y2:
        pred_y2 += 1
    pred_line = Line([pred_y1, pred_x1, pred_y2, pred_x2])

    # get the ground truth line
    # Extract components
    center = annotation[:2]  # shape (2,)
    angles_radian = annotation[2]  # shape (1,)
    length = annotation[3]  # shape (1,)

    # Calculate half-length offsets
    dx = 0.5 * length * torch.cos(angles_radian)
    dy = 0.5 * length * torch.sin(angles_radian)

    # Calculate the endpoints
    gt_x1 = center[0] - dx
    gt_y1 = center[1] - dy
    gt_x2 = center[0] + dx
    gt_y2 = center[1] + dy
    gt_x1, gt_y1, gt_x2, gt_y2 = np.int_([gt_x1, gt_y1, gt_x2, gt_y2])
    if gt_x1 == gt_x2:
        gt_x2 += 1
    if gt_y1 == gt_y2:
        gt_y2 += 1
    gt_line = Line([gt_y1, gt_x1, gt_y2, gt_x2])

    # Calculate the EA score
    se = se_metric(pred_line.coord, gt_line.coord, size=size)
    sa = sa_metric(pred_line.angle(), gt_line.angle())
    if without_len:
        return sa * se

    # Calculate the EAL score
    sl = len_metric(pred_line.coord, gt_line.coord, size=size)
    return sa * se * sl


def seg_line_evaluate(preds, masks, ea_threshold=0.9, eal_threshold=0.9):
    preds = preds.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
    masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]

    # Calculate the EA, EAL score for each sample in the batch
    ea_score_list = []
    eal_score_list = []
    p_count = 0  # positive count
    n_count = 0  # negative count
    count_ea = np.array([0, 0, 0, 0])  # [TP, FP, TN, FN]
    count_eal = np.array([0, 0, 0, 0])  # [TP, FP, TN, FN]

    for i in range(preds.shape[0]):
        pred = preds[i].cpu().numpy()
        mask = masks[i].cpu().numpy()

        # Convert the predicted mask to coordinates
        pred_idx = np.where(pred > 0.5)
        mask_idx = np.where(mask > 0)

        # check if ground truth and prediction are empty
        # count predicted tp, fp, tn, fn samples number
        pixel_num_threshold = 1
        pred_empty = len(pred_idx[0]) <= pixel_num_threshold
        gt_empty = len(mask_idx[1]) <= pixel_num_threshold

        # count ground truth positive and negative samples number
        if not gt_empty:
            p_count += 1
        else:
            n_count += 1

        if not pred_empty and not gt_empty:
            # EA score
            ea_score = seg_eal_metric(pred_idx, mask_idx, size=pred.shape, without_len=True)
            ea_score_list.append(ea_score)
            if ea_score >= ea_threshold:
                count_ea[0] += 1  # TP
            else:
                count_ea[3] += 1  # FN
            # EAL score
            eal_score = seg_eal_metric(pred_idx, mask_idx, size=pred.shape)
            eal_score_list.append(eal_score)
            if eal_score >= eal_threshold:
                count_eal[0] += 1  # TP
            else:
                count_eal[3] += 1  # FN
        elif not pred_empty and gt_empty:
            count_ea[1] += 1  # FP
            count_eal[1] += 1  # FP
        elif pred_empty and not gt_empty:
            count_ea[3] += 1  # FN
            count_eal[3] += 1  # FN
        elif pred_empty and gt_empty:
            count_ea[2] += 1  # TN
            count_eal[2] += 1  # TN

    return (
        np.mean(ea_score_list) if len(ea_score_list) > 0 else 0.0,
        np.mean(eal_score_list) if len(eal_score_list) > 0 else 0.0,
        p_count,
        n_count,
        count_ea,
        count_eal,
    )


def det_line_evaluate(
    pred_classifications,  # [N, num_total_anchors, num_classes]
    pred_regressions,  # [N, num_total_anchors, 4]
    annotations,  # [N, 1, 5]
    anchors_pos,  # [num_total_anchors, 4]
    image_width,
    image_height,
    conf_thresh=0.1,
    with_aqe=False,
    ea_threshold=0.9,
    eal_threshold=0.9,
):
    # Calculate the EA, EAL score for each sample in the batch
    ea_score_list = []
    eal_score_list = []
    p_count = 0  # positive count
    n_count = 0  # negative count
    count_ea = np.array([0, 0, 0, 0])  # [TP, FP, TN, FN]
    count_eal = np.array([0, 0, 0, 0])  # [TP, FP, TN, FN]

    for i in range(pred_classifications.shape[0]):
        pred_cls = pred_classifications[i]  # [num_total_anchors, num_classes]
        pred_reg = pred_regressions[i]  # [num_total_anchors, 4]
        annotation = annotations[i]  # [1, 5]
        annotation = annotation.squeeze(0)  # [5]

        # get the top-1 detection endpoints
        _, topk_endpoints, _ = detect_postprocessing(
            pred_cls,
            pred_reg,
            anchors_pos,
            image_width,
            image_height,
            conf_thresh=conf_thresh,
            topk=1,
            with_aqe=with_aqe,
        )
        top1_endpoints = topk_endpoints[0]

        # check if ground truth and prediction are empty
        gt_empty = annotation[4] == -1
        pred_empty = torch.equal(top1_endpoints.cpu(), torch.tensor([0, 0, 0, 0], dtype=torch.float32))

        # count ground truth positive and negative samples number
        if not gt_empty:
            p_count += 1
        else:
            n_count += 1

        if not pred_empty and not gt_empty:
            # EA score
            ea_score = det_eal_metric(top1_endpoints, annotation, size=(image_width, image_height), without_len=True)
            ea_score_list.append(ea_score)
            if ea_score >= ea_threshold:
                count_ea[0] += 1  # TP
            else:
                count_ea[3] += 1  # FN
            # EAL score
            eal_score = det_eal_metric(top1_endpoints, annotation, size=(image_width, image_height))
            eal_score_list.append(eal_score)
            if eal_score >= eal_threshold:
                count_eal[0] += 1  # TP
            else:
                count_eal[3] += 1  # FN
        elif not pred_empty and gt_empty:
            count_ea[1] += 1  # FP
            count_eal[1] += 1  # FP
        elif pred_empty and not gt_empty:
            count_ea[3] += 1  # FN
            count_eal[3] += 1  # FN
        elif pred_empty and gt_empty:
            count_ea[2] += 1  # TN
            count_eal[2] += 1  # TN

    return (
        np.mean(ea_score_list) if len(ea_score_list) > 0 else 0.0,
        np.mean(eal_score_list) if len(eal_score_list) > 0 else 0.0,
        p_count,
        n_count,
        count_ea,
        count_eal,
    )


# Evaluation Function
def evaluate(model, device, loader, seg_focal_loss, seg_dice_loss, model_name, det_loss=None, anchors_pos=None, with_aqe=False):
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

    # Initialize variables to store EA, EAL score based line metrics
    # Segmentation
    total_seg_p_count = 0
    total_seg_n_count = 0
    total_seg_ea_score = 0.0
    total_seg_eal_score = 0.0
    total_seg_count_ea = np.array([0, 0, 0, 0])  # [TP, FP, TN, FN], EA score based
    total_seg_count_eal = np.array([0, 0, 0, 0])  # [TP, FP, TN, FN], EAL score based
    # Detection
    if model_name == "Video-Retina-UNETR":
        total_det_p_count = 0
        total_det_n_count = 0
        total_det_ea_score = 0.0
        total_det_eal_score = 0.0
        total_det_count_ea = np.array([0, 0, 0, 0])  # [TP, FP, TN, FN], EA score based
        total_det_count_eal = np.array([0, 0, 0, 0])  # [TP, FP, TN, FN], EAL score based

    with torch.no_grad():
        for step, samples in enumerate(tqdm(loader)):
            # Image data
            images = samples["images"].to(device)  # [N, T, H, W]

            # Segmentation ground truth masks
            masks = samples["masks"].to(device)  # [N, T, H, W]
            masks = masks[:, -1, :, :].unsqueeze(1)  # [N, 1, H, W]

            # Detection ground truth annotations
            if model_name == "Video-Retina-UNETR":
                cals = samples["cals"].to(device)  # [N, T, 4]
                labels = samples["labels"].to(device)  # [N, T]
                labels = labels.unsqueeze(-1)  # [N, T, 1]
                annotations = torch.cat([cals, labels], dim=-1)  # [N, T, 5]
                annotations = annotations[:, -1, :].unsqueeze(1)  # [N, 1, 5]

            # Forward pass
            if model_name == "Video-Retina-UNETR":
                pred_masks, pred_classifications, pred_regressions = model(images)
                # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4 or 5]
            else:
                pred_masks = model(images)  # [N, 1, H, W]

            # Calculate loss (use the last frame as the target mask)
            fl = seg_focal_loss(pred_masks, masks)
            dl = seg_dice_loss(pred_masks, masks)
            if model_name == "Video-Retina-UNETR":  # with the detection head
                cl, rl = det_loss(pred_classifications, pred_regressions, anchors_pos, annotations)

            # Calculate total loss
            loss = fl + dl
            if model_name == "Video-Retina-UNETR":  # with the detection head
                loss = loss + cl + rl

            # Calculate the segmentation Dice score & IoU score
            seg_dscore = seg_dice_score(pred_masks, masks)
            seg_iscore = seg_iou_score(pred_masks, masks)

            # Calculate EA, EAL score and line metrics
            # Segmentation
            (
                mean_seg_ea_score,
                mean_seg_eal_score,
                seg_p_count,
                seg_n_count,
                seg_count_ea,
                seg_count_eal,
            ) = seg_line_evaluate(pred_masks, masks)
            # Detection
            if model_name == "Video-Retina-UNETR":
                (
                    mean_det_ea_score,
                    mean_det_eal_score,
                    det_p_count,
                    det_n_count,
                    det_count_ea,
                    det_count_eal,
                ) = det_line_evaluate(
                    pred_classifications, pred_regressions, annotations, anchors_pos, images.shape[-1], images.shape[-2], with_aqe=with_aqe
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

            # Needle EA & EAL Score based metrics
            # Segmentation
            total_seg_ea_score += mean_seg_ea_score
            total_seg_eal_score += mean_seg_eal_score
            total_seg_p_count += seg_p_count
            total_seg_n_count += seg_n_count
            total_seg_count_ea += seg_count_ea  # [TP, FP, TN, FN]
            total_seg_count_eal += seg_count_eal  # [TP, FP, TN, FN]
            # Detection
            if model_name == "Video-Retina-UNETR":
                total_det_ea_score += mean_det_ea_score
                total_det_eal_score += mean_det_eal_score
                total_det_p_count += det_p_count
                total_det_n_count += det_n_count
                total_det_count_ea += det_count_ea  # [TP, FP, TN, FN]
                total_det_count_eal += det_count_eal  # [TP, FP, TN, FN]

    results = {
        "Loss": total_loss / len(loader),
        "Segmentation Focal Loss": total_seg_focal_loss / len(loader),
        "Segmentation Dice Loss": total_seg_dice_loss / len(loader),
        "Segmentation Dice Score": total_seg_dice_score / len(loader),
        "Segmentation IoU Score": total_seg_iou_score / len(loader),
        "Segmentation Needle EA Score": total_seg_ea_score / len(loader),
        "Segmentation Needle EAL Score": total_seg_eal_score / len(loader),
        "Needle Positive Count": total_seg_p_count,
        "Needle Negative Count": total_seg_n_count,
        # EA Score based
        "Segmentation Needle TP Count": total_seg_count_ea[0],
        "Segmentation Needle FP Count": total_seg_count_ea[1],
        "Segmentation Needle TN Count": total_seg_count_ea[2],
        "Segmentation Needle FN Count": total_seg_count_ea[3],
        "Segmentation Needle Recall Score": total_seg_count_ea[0] / (total_seg_count_ea[0] + total_seg_count_ea[3] + 1e-6),
        "Segmentation Needle Precision Score": total_seg_count_ea[0] / (total_seg_count_ea[0] + total_seg_count_ea[1] + 1e-6),
        "Segmentation Needle Specificity Score": total_seg_count_ea[2] / (total_seg_count_ea[2] + total_seg_count_ea[1] + 1e-6),
        # EAL Score based
        "Segmentation Needle TP Count (EAL)": total_seg_count_eal[0],
        "Segmentation Needle FP Count (EAL)": total_seg_count_eal[1],
        "Segmentation Needle TN Count (EAL)": total_seg_count_eal[2],
        "Segmentation Needle FN Count (EAL)": total_seg_count_eal[3],
        "Segmentation Needle Recall Score (EAL)": total_seg_count_eal[0] / (total_seg_count_eal[0] + total_seg_count_eal[3] + 1e-6),
        "Segmentation Needle Precision Score (EAL)": total_seg_count_eal[0] / (total_seg_count_eal[0] + total_seg_count_eal[1] + 1e-6),
        "Segmentation Needle Specificity Score (EAL)": total_seg_count_eal[2] / (total_seg_count_eal[2] + total_seg_count_eal[1] + 1e-6),
    }

    if model_name == "Video-Retina-UNETR":
        results["Detection Classification Loss"] = total_det_cls_loss / len(loader)
        results["Detection Regression Loss"] = total_det_reg_loss / len(loader)
        results["Detection Needle EA Score"] = total_det_ea_score / len(loader)
        results["Detection Needle EAL Score"] = total_det_eal_score / len(loader)
        # EA Score based
        results["Detection Needle TP Count"] = total_det_count_ea[0]
        results["Detection Needle FP Count"] = total_det_count_ea[1]
        results["Detection Needle TN Count"] = total_det_count_ea[2]
        results["Detection Needle FN Count"] = total_det_count_ea[3]
        results["Detection Needle Recall Score"] = total_det_count_ea[0] / (total_det_count_ea[0] + total_det_count_ea[3] + 1e-6)
        results["Detection Needle Precision Score"] = total_det_count_ea[0] / (total_det_count_ea[0] + total_det_count_ea[1] + 1e-6)
        results["Detection Needle Specificity Score"] = total_det_count_ea[2] / (total_det_count_ea[2] + total_det_count_ea[1] + 1e-6)
        # EAL Score based
        results["Detection Needle TP Count (EAL)"] = total_det_count_eal[0]
        results["Detection Needle FP Count (EAL)"] = total_det_count_eal[1]
        results["Detection Needle TN Count (EAL)"] = total_det_count_eal[2]
        results["Detection Needle FN Count (EAL)"] = total_det_count_eal[3]
        results["Detection Needle Recall Score (EAL)"] = total_det_count_eal[0] / (total_det_count_eal[0] + total_det_count_eal[3] + 1e-6)
        results["Detection Needle Precision Score (EAL)"] = total_det_count_eal[0] / (total_det_count_eal[0] + total_det_count_eal[1] + 1e-6)
        results["Detection Needle Specificity Score (EAL)"] = total_det_count_eal[2] / (total_det_count_eal[2] + total_det_count_eal[1] + 1e-6)

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
            # EA Score based
            "Segmentation Needle TP Count": 0,
            "Segmentation Needle FP Count": 0,
            "Segmentation Needle TN Count": 0,
            "Segmentation Needle FN Count": 0,
            "Segmentation Needle Recall Score": 0.0,
            "Segmentation Needle Precision Score": 0.0,
            "Segmentation Needle Specificity Score": 0.0,
            # EAL Score based
            "Segmentation Needle TP Count (EAL)": 0,
            "Segmentation Needle FP Count (EAL)": 0,
            "Segmentation Needle TN Count (EAL)": 0,
            "Segmentation Needle FN Count (EAL)": 0,
            "Segmentation Needle Recall Score (EAL)": 0.0,
            "Segmentation Needle Precision Score (EAL)": 0.0,
            "Segmentation Needle Specificity Score (EAL)": 0.0,
        }

        # Add detection metrics for Video-Retina-UNETR
        if model_name == "Video-Retina-UNETR":
            results_dict["Detection Classification Loss"] = float("inf")
            results_dict["Detection Regression Loss"] = float("inf")
            results_dict["Detection Needle EA Score"] = 0.0
            results_dict["Detection Needle EAL Score"] = 0.0
            # EA Score based
            results_dict["Detection Needle TP Count"] = 0
            results_dict["Detection Needle FP Count"] = 0
            results_dict["Detection Needle TN Count"] = 0
            results_dict["Detection Needle FN Count"] = 0
            results_dict["Detection Needle Recall Score"] = 0.0
            results_dict["Detection Needle Precision Score"] = 0.0
            results_dict["Detection Needle Specificity Score"] = 0.0
            # EAL Score based
            results_dict["Detection Needle TP Count (EAL)"] = 0
            results_dict["Detection Needle FP Count (EAL)"] = 0
            results_dict["Detection Needle TN Count (EAL)"] = 0
            results_dict["Detection Needle FN Count (EAL)"] = 0
            results_dict["Detection Needle Recall Score (EAL)"] = 0.0
            results_dict["Detection Needle Precision Score (EAL)"] = 0.0
            results_dict["Detection Needle Specificity Score (EAL)"] = 0.0

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
