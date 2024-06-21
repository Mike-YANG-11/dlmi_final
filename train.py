# Main script to train the Video-UNETR model

import datetime
import json
import logging
import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import wandb

from anchors import AnchorGenerator

from functools import partial

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm

from dataset import CustomDataset, Augmentation, UnlabeledDataset, PseudoDataset, AugmentationImgMaskOnly
from evaluation import evaluate, seg_dice_score, seg_iou_score, results_dictioanry
from visualization import show_dataset_samples, show_seg_preds_only, show_preds_with_det_head
from model import VideoUnetr, VideoRetinaUnetr
from loss import SegFocalLoss, SegDiceLoss, SegFocalTverskyLoss, SegIoULoss, DetLoss, SIoULoss, AQELoss, mse_loss, sigmoid_rampup
from pseudolabel import generate_pl


# Construct the datasets
def construct_datasets(config):
    image_size = config["Model"]["image_size"]
    batch_size = config["Train"]["batch_size"]
    time_window = config["Model"]["time_window"]
    buffer_num_sample = config["Train"]["buffer_num_sample"]
    line_width = config["Data"]["line_width"]
    det_num_classes = config["Train"]["det_num_classes"]

    train_transform = Augmentation(resized_crop=True, color_jitter=True, horizontal_flip=True, image_size=image_size)
    valid_transform = Augmentation(resized_crop=False, color_jitter=False, horizontal_flip=False, image_size=image_size)

    train_dataset_list = []
    for folder_name in config["Data"]["Train_folder"].values():
        subdataset = CustomDataset(
            os.path.join(config["Data"]["folder_dir"], folder_name),
            transform=train_transform,
            time_window=time_window,
            buffer_num_sample=buffer_num_sample,
            line_width=line_width,
            det_num_classes=det_num_classes,
        )
        train_dataset_list.append(subdataset)

    valid_dataset_list = []
    for folder_name in config["Data"]["Val_folder"].values():
        subdataset = CustomDataset(
            os.path.join(config["Data"]["folder_dir"], folder_name),
            transform=valid_transform,
            time_window=time_window,
            buffer_num_sample=buffer_num_sample,
            line_width=line_width,
            det_num_classes=det_num_classes,
        )
        valid_dataset_list.append(subdataset)

    test_med_dataset_list = []
    for folder_name in config["Data"]["Test_folder"]["Medium"].values():
        subdataset = CustomDataset(
            os.path.join(config["Data"]["folder_dir"], folder_name),
            transform=valid_transform,
            time_window=time_window,
            buffer_num_sample=buffer_num_sample,
            line_width=line_width,
            det_num_classes=det_num_classes,
        )
        test_med_dataset_list.append(subdataset)

    test_hard_dataset_list = []
    for folder_name in config["Data"]["Test_folder"]["Hard"].values():
        subdataset = CustomDataset(
            os.path.join(config["Data"]["folder_dir"], folder_name),
            transform=valid_transform,
            time_window=time_window,
            buffer_num_sample=buffer_num_sample,
            line_width=line_width,
            det_num_classes=det_num_classes,
        )
        test_hard_dataset_list.append(subdataset)

    """ training dataset """
    train_dataset = ConcatDataset(train_dataset_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    """ validation dataset """
    valid_dataset = ConcatDataset(valid_dataset_list)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    """ medium test dataset """
    medium_test_dataset = ConcatDataset(test_med_dataset_list)
    medium_test_loader = DataLoader(medium_test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    """ hard test dataset """
    hard_test_dataset = ConcatDataset(test_hard_dataset_list)
    hard_test_loader = DataLoader(hard_test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}")
    print(f"Number of medium test samples: {len(medium_test_dataset)}")
    print(f"Number of hard test samples: {len(hard_test_dataset)}")

    return train_loader, valid_loader, medium_test_loader, hard_test_loader


def construct_unlabeled_datasets(config):
    image_size = config["Model"]["image_size"]
    batch_size = config["Train"]["batch_size"]
    time_window = config["Model"]["time_window"]

    valid_transform = Augmentation(resized_crop=False, color_jitter=False, horizontal_flip=False, image_size=image_size)

    dataset_list = []
    for folder_name in config["Data"]["Unlabeled_folder"]:
        subdataset = UnlabeledDataset(os.path.join(config["Data"]["folder_dir"], folder_name), transform=valid_transform, time_window=time_window)
        dataset_list.append(subdataset)

    unlabeled_dataset = ConcatDataset(dataset_list)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)  ## avoid 1 at the end

    print(f"Number of unlabeled samples: {len(unlabeled_dataset)}")
    return unlabeled_loader


# Training Function
def train(
    config,
    model,
    device,
    train_loader,
    valid_loader,
    optimizer,
    seg_focal_loss,
    seg_dice_loss,
    seg_ft_loss,
    seg_iou_loss,
    model_name,
    logger,
    checkpoint_dir,
    det_loss=None,
    scheduler=None,
    visualize=True,
):
    def train_one_epoch(model, optimizer, train_loader, running_results, train_det_head=True, ignore_index=None):
        # parameters for iteration over the buffer
        time_window = config["Model"]["time_window"]
        buffer_num_sample = config["Train"]["buffer_num_sample"]

        # total number of training steps in one epoch & count the current step
        ep_train_steps = len(train_loader) * buffer_num_sample
        train_step_count = 0

        with tqdm(total=ep_train_steps) as pbar:
            for buffer in train_loader:  # loader getitem() returns a buffer
                for t in range(buffer_num_sample):  # iterate over the buffer to get samples
                    # Image data
                    images = buffer["images"][:, t : t + time_window, :, :].to(device)  # [N, T, H, W]

                    # Segmentation ground truth masks
                    masks = buffer["masks"][:, t : t + time_window, :, :].to(device)  # [N, T, H, W]
                    masks = masks[:, -1, :, :].unsqueeze(1)  # [N, 1, H, W]

                    # Detection ground truth annotations
                    if model_name == "Video-Retina-UNETR" and train_det_head:
                        cals = buffer["cals"][:, t : t + time_window, :].to(device)  # [N, T, 4]
                        labels = buffer["labels"][:, t : t + time_window].to(device)  # [N, T]
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
                    fl = seg_focal_loss(pred_masks, masks, ignore_index=ignore_index)
                    dl = seg_dice_loss(pred_masks, masks, ignore_index=ignore_index)
                    # ftl = seg_ft_loss(pred_masks, masks)
                    # il = seg_iou_loss(pred_masks, masks)
                    if model_name == "Video-Retina-UNETR" and train_det_head:  # with the detection head
                        cl, rl = det_loss(pred_classifications, pred_regressions, anchors_pos, annotations)

                    # Calculate total loss
                    loss = dl  # + fl  # + il #  ftl
                    if model_name == "Video-Retina-UNETR" and train_det_head:  # with the detection head
                        loss = loss + 0.2 * (cl + rl)  ## TODO: adaptively modify the weight for the detection loss

                    # mean teacher: consistency cost
                    ## https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py#L260
                    if config["Validation"]["ema"] != 0 and config["Validation"]["mean_teacher"]:
                        ema_model.train()
                        if model_name == "Video-Retina-UNETR":
                            ema_pred_masks, ema_pred_classifications, ema_pred_regressions = ema_model(images)
                            # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4 or 5]
                        else:
                            ema_pred_masks = ema_model(images)  # [N, 1, H, W]
                        consistency_weight = config["Validation"]["consistency_weight"] * sigmoid_rampup(epoch, epochs)
                        consistency_l = consistency_weight * mse_loss(pred_masks, ema_pred_masks) / pred_masks.shape[0]
                        if train_step_count == 10 or train_step_count == 50:
                            print(f"loss {loss.item()} cons_l {consistency_l.item()}")
                        loss += consistency_l
                        consistency_l = consistency_l.item()
                    else:
                        consistency_l = 0

                    # Calculate the segmentation Dice score & IoU score
                    seg_dscore = seg_dice_score(pred_masks, masks)
                    seg_iscore = seg_iou_score(pred_masks, masks)

                    # update running loss & score
                    running_results["Loss"] += dl.item() + consistency_l  # + fl.item()
                    running_results["Segmentation Focal Loss"] += 0  # fl.item()
                    running_results["Segmentation Dice Loss"] += dl.item()
                    running_results["Segmentation Dice Score"] += seg_dscore.item()
                    running_results["Segmentation IoU Score"] += seg_iscore.item()
                    if model_name == "Video-Retina-UNETR" and train_det_head:  # with the detection head
                        running_results["Loss"] += cl.item() + rl.item()
                        running_results["Detection Classification Loss"] += cl.item()
                        running_results["Detection Regression Loss"] += rl.item()

                    # Gradient accumulation normalization
                    loss = loss / accumulation_steps
                    loss.backward()

                    # Update weights
                    if (train_step_count + 1) % accumulation_steps == 0 or train_step_count == ep_train_steps - 1:
                        # Gradient clipping
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                        optimizer.step()
                        optimizer.zero_grad()

                        if config["Validation"]["ema"]:
                            ema_model.update_parameters(model)

                    train_step_count += 1
                    pbar.update(1)

                    if train_step_count == ep_train_steps:
                        return running_results

    def pl_train_one_epoch(model, optimizer, train_loader, running_results, train_det_head=True, ignore_index=None):
        for step, samples in enumerate(tqdm(train_loader)):
            # Image data
            images = samples["images"].to(device)  # [N, T, H, W]

            # Segmentation ground truth masks
            masks = samples["masks"].to(device)  # [N, T, H, W]
            masks = masks[:, -1, :, :].unsqueeze(1)  # [N, 1, H, W]

            # Detection ground truth annotations
            if model_name == "Video-Retina-UNETR" and train_det_head:
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
            fl = seg_focal_loss(pred_masks, masks, ignore_index=ignore_index)
            dl = seg_dice_loss(pred_masks, masks, ignore_index=ignore_index)
            # ftl = seg_ft_loss(pred_masks, masks)
            # il = seg_iou_loss(pred_masks, masks)
            if model_name == "Video-Retina-UNETR" and train_det_head:  # with the detection head
                cl, rl = det_loss(pred_classifications, pred_regressions, anchors_pos, annotations)

            # Calculate total loss
            loss = dl  # + fl  # + il+  fl  #  ftl
            if model_name == "Video-Retina-UNETR" and train_det_head:  # with the detection head
                loss = loss + 0.2 * (cl + rl)  ## TODO: adaptively modify the weight for the detection loss

            # mean teacher: consistency cost
            ## https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py#L260
            if config["Validation"]["ema"] != 0 and config["Validation"]["mean_teacher"]:
                ema_model.train()
                if model_name == "Video-Retina-UNETR":
                    ema_pred_masks, ema_pred_classifications, ema_pred_regressions = ema_model(images)
                    # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4 or 5]
                else:
                    ema_pred_masks = ema_model(images)  # [N, 1, H, W]
                consistency_weight = config["Validation"]["consistency_weight"] * sigmoid_rampup(epoch, epochs)
                consistency_l = consistency_weight * mse_loss(pred_masks, ema_pred_masks) / pred_masks.shape[0]
                if step == 10 or step == 50:
                    print(f"loss {loss.item()} cons_l {consistency_l.item()}")
                loss += consistency_l
                consistency_l = consistency_l.item()
            else:
                consistency_l = 0

            # Calculate the segmentation Dice score & IoU score
            seg_dscore = seg_dice_score(pred_masks, masks)
            seg_iscore = seg_iou_score(pred_masks, masks)

            # update running loss & score
            running_results["Loss"] += dl.item() + consistency_l  # + fl.item()
            running_results["Segmentation Focal Loss"] += 0  # fl.item()
            running_results["Segmentation Dice Loss"] += dl.item()
            running_results["Segmentation Dice Score"] += seg_dscore.item()
            running_results["Segmentation IoU Score"] += seg_iscore.item()
            if model_name == "Video-Retina-UNETR" and train_det_head:  # with the detection head
                running_results["Loss"] += cl.item() + rl.item()
                running_results["Detection Classification Loss"] += cl.item()
                running_results["Detection Regression Loss"] += rl.item()

            # Gradient accumulation normalization
            loss = loss / accumulation_steps
            loss.backward()

            # Update weights
            if (step + 1) % accumulation_steps == 0:
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

                if config["Validation"]["ema"] != 0:
                    ema_model.update_parameters(model)

        return running_results

    def visualize_sample(model, loader, model_name, anchors_pos=None):
        time_window = config["Model"]["time_window"]
        buffer_num_sample = config["Train"]["buffer_num_sample"]
        with torch.no_grad():
            for buffer in loader:
                for t in range(buffer_num_sample):  # iterate over the buffer to get samples
                    vis_images = buffer["images"][:, t : t + time_window, :, :].to(device)  # [N, T, H, W]
                    vis_masks = buffer["masks"][:, t : t + time_window, :, :]  # [N, T, H, W]

                    # Forward pass
                    if model_name == "Video-Retina-UNETR":
                        vis_pred_masks, vis_classifications, vis_regressions = model(vis_images)
                        # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4]
                    else:
                        vis_pred_masks = model(vis_images)  # [N, 1, H, W]

                    # Detach and move to CPU
                    vis_images = vis_images.detach().cpu()
                    vis_pred_masks = vis_pred_masks.detach().cpu()
                    if model_name == "Video-Retina-UNETR":
                        vis_classifications = vis_classifications.detach().cpu()  # [N, num_total_anchors, num_classes]
                        vis_regressions = vis_regressions.detach().cpu()  # [N, num_total_anchors, 4]
                        anchors_pos = anchors_pos.detach().cpu()  # [num_total_anchors, 4]

                    # Visualize the output
                    if model_name == "Video-Retina-UNETR":
                        show_preds_with_det_head(
                            vis_images,
                            vis_masks,
                            vis_pred_masks,
                            vis_classifications,
                            vis_regressions,
                            anchors_pos,
                            topk=1,
                            with_aqe=config["Train"]["with_aqe"],
                        )
                        anchors_pos = anchors_pos.cuda()
                    else:
                        show_seg_preds_only(consec_images=vis_images, consec_masks=vis_masks, pred_masks=vis_pred_masks)

                    break
                break
            # -------------------------------------------------------------------------------
            # original version
            # -------------------------------------------------------------------------------
            # for _, vis_samples in enumerate(loader):
            #     # Move to device & forward pass
            #     vis_images = vis_samples["images"].to(device)
            #     vis_masks = vis_samples["masks"]

            #     # Forward pass
            #     if model_name == "Video-Retina-UNETR":
            #         vis_pred_masks, vis_classifications, vis_regressions = model(vis_images)
            #         # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4]
            #     else:
            #         vis_pred_masks = model(vis_images)  # [N, 1, H, W]

            #     # Detach and move to CPU
            #     vis_images = vis_images.detach().cpu()
            #     vis_pred_masks = vis_pred_masks.detach().cpu()
            #     if model_name == "Video-Retina-UNETR":
            #         vis_classifications = vis_classifications.detach().cpu()  # [N, num_total_anchors, num_classes]
            #         vis_regressions = vis_regressions.detach().cpu()  # [N, num_total_anchors, 4]
            #         anchors_pos = anchors_pos.detach().cpu()  # [num_total_anchors, 4]

            #     # Visualize the output
            #     if model_name == "Video-Retina-UNETR":
            #         show_preds_with_det_head(
            #             vis_images,
            #             vis_masks,
            #             vis_pred_masks,
            #             vis_classifications,
            #             vis_regressions,
            #             anchors_pos,
            #             topk=1,
            #             with_aqe=config["Train"]["with_aqe"],
            #         )
            #         anchors_pos = anchors_pos.cuda()
            #     else:
            #         show_seg_preds_only(consec_images=vis_images, consec_masks=vis_masks, pred_masks=vis_pred_masks)

            #     break
            # -------------------------------------------------------------------------------

    epochs = config["Train"]["epochs"]
    accumulation_steps = config["Train"]["accumulation_steps"]
    experiment_id = config["Train"]["experiment_id"]

    ### Pseudo Label Training ==================================================
    ## if model validation if good enough, then generate pseudo labels at current epoch
    pl_model_thres = config["Semi_Supervise"]["Pseudo_label"]["model_thres"]
    folder_note = config["Semi_Supervise"]["Pseudo_label"]["folder_note"]
    root_dir = config["Semi_Supervise"]["Pseudo_label"]["root_dir"]

    train_pl = False  ## train with pseudo label or not
    if pl_model_thres is not None:
        pl_dir = os.path.join(root_dir, f"{model_name}_{str(experiment_id)}")  # _{folder_note}
        unlabeled_loader = construct_unlabeled_datasets(config)
        df = pd.DataFrame(columns=["img_root", "img_names", "mask_path", "confidence"])
        train_pl_loader = None
        running_pl_results = None
    ### ========================================================================

    ## https://pytorch.org/docs/stable/optim.html#putting-it-all-together-ema
    if config["Validation"]["ema"] != 0:
        ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config["Validation"]["ema"]))
        ema_model.to(device)

    logger.info(f"Model: {model_name}")
    logger.info(f"Experiment ID: {experiment_id}")

    model.train()
    model.to(device)

    # Initialize best validation results
    best_val_results = results_dictioanry(model_name=model_name, type="best_val_results")

    # Initialize early stopping count
    early_stop_count = 0

    # Initialize the anchors position for the detection head
    if model_name == "Video-Retina-UNETR":
        image_size = config["Model"]["image_size"]
        anchor_generator = AnchorGenerator()
        anchors_pos = anchor_generator([image_size, image_size])

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Reset running results
        running_results = results_dictioanry(model_name=model_name, type="running_results")

        running_results = train_one_epoch(model, optimizer, train_loader, running_results)

        ep_train_steps = len(train_loader) * config["Train"]["buffer_num_sample"]

        ## Semi Supervise: Train with Pseudo Label
        if train_pl:
            print("Training PL...")
            ignore = None
            if config["Semi_Supervise"]["Pseudo_label"]["pl_version"] == 2:
                ignore = 2
            running_pl_results = results_dictioanry(model_name="", type="running_results")
            running_pl_results = pl_train_one_epoch(
                model,
                optimizer,
                train_loader=train_pl_loader,
                running_results=running_pl_results,
                train_det_head=False,  ## ## only train seg head
                ignore_index=ignore,
            )  ## ignore low confidence pixels in v2 when calculating loss

        if scheduler is not None:
            # Adjust learning rate
            scheduler.step()

        # Evaluate the model on validation data
        if config["Validation"]["ema"] != 0:
            # Update bn statistics for the ema_model at the end
            torch.optim.swa_utils.update_bn(train_loader, ema_model)
            if train_pl:
                torch.optim.swa_utils.update_bn(train_pl_loader, ema_model)
            # Use ema_model to make predictions on test data
            if model_name == "Video-Retina-UNETR":
                val_results = evaluate(
                    config,
                    ema_model,
                    device,
                    valid_loader,
                    seg_focal_loss,
                    seg_dice_loss,
                    model_name,
                    det_loss,
                    anchors_pos,
                    with_aqe=config["Train"]["with_aqe"],
                )
            else:
                val_results = evaluate(config, ema_model, device, valid_loader, seg_focal_loss, seg_dice_loss, model_name)
        else:
            if model_name == "Video-Retina-UNETR":
                val_results = evaluate(
                    config,
                    model,
                    device,
                    valid_loader,
                    seg_focal_loss,
                    seg_dice_loss,
                    model_name,
                    det_loss,
                    anchors_pos,
                    with_aqe=config["Train"]["with_aqe"],
                )
            else:
                val_results = evaluate(config, model, device, valid_loader, seg_focal_loss, seg_dice_loss, model_name)

        # Print running loss on training data & loss on validation data
        print("--------------------------------------------------")
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {running_results['Loss']/ep_train_steps:.6f}")
        if train_pl:
            print(f"Training PL Loss: {running_pl_results['Loss']/ep_train_steps:.6f}")
        print(f"Validation Loss: {val_results['Loss']:.6f}")

        # Save the best model & visualize the output if the model is improved
        if (
            val_results["Loss"] < best_val_results["Loss"]
            or val_results["Segmentation Dice Loss"] < best_val_results["Segmentation Dice Loss"]
            or val_results["Segmentation Needle EA Score"] > best_val_results["Segmentation Needle EA Score"]
            or val_results["Segmentation Needle EAL Score"] > best_val_results["Segmentation Needle EAL Score"]
        ):
            # Update the best validation loss & IoU score and reset early stopping count
            best_val_results = val_results
            early_stop_count = 0

            # Save the best model
            if model_name == "Video-Retina-UNETR":
                save_path = os.path.join(checkpoint_dir, f"video_retina_unetr_{experiment_id}.pth")
            else:
                save_path = os.path.join(checkpoint_dir, f"video_unetr_{experiment_id}.pth")
            if config["Validation"]["ema"] != 0:
                torch.save(ema_model.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print("The best model is saved!")

            # Visualize the output
            if visualize:
                model.eval()

                # Visualize the output on training data (show the first batch)
                print("Visualizing the output on training data...")
                if model_name == "Video-Retina-UNETR":
                    visualize_sample(model, loader=train_loader, model_name=model_name, anchors_pos=anchors_pos)
                else:
                    visualize_sample(model, loader=train_loader, model_name=model_name)

                # Visualize the output on validation data (show the first batch)
                print("Visualizing the output on validation data...")
                if config["Validation"]["ema"] != 0:
                    if model_name == "Video-Retina-UNETR":
                        visualize_sample(ema_model, loader=valid_loader, model_name=model_name, anchors_pos=anchors_pos)
                    else:
                        visualize_sample(ema_model, loader=valid_loader, model_name=model_name)
                else:
                    if model_name == "Video-Retina-UNETR":
                        visualize_sample(model, loader=valid_loader, model_name=model_name, anchors_pos=anchors_pos)
                    else:
                        visualize_sample(model, loader=valid_loader, model_name=model_name)
        else:
            early_stop_count += 1
            if early_stop_count == 5:
                print("Early stopping...")
                break

        print(f"Best Validation Loss: {best_val_results['Loss']:.6f}")
        print("--------------------------------------------------")

        # Log to local logger
        logger.info("--------------------------------------------------")
        logger.info(f"Epoch [{epoch+1}/{epochs}]")
        logger.info("--------------------------------------------------")
        logger.info(f"Running Results on Training Data:")
        for key, value in running_results.items():
            logger.info(f"{key}: {value/ep_train_steps:.6f}")
        logger.info("--------------------------------------------------")
        if train_pl and running_pl_results is not None:
            logger.info(f"Running PL Results on Pseudo Label Data:")
            for key, value in running_pl_results.items():
                logger.info(f"{key}: {value/len(train_pl_loader):.6f}")
            logger.info("--------------------------------------------------")
        logger.info(f"Validation Results:")
        for key, value in val_results.items():
            logger.info(f"{key}: {value:.6f}")
        logger.info("--------------------------------------------------")
        logger.info(f"Best Validation Results:")
        for key, value in best_val_results.items():
            logger.info(f"{key}: {value:.6f}")
        logger.info("--------------------------------------------------")

        # Log to wandb
        wandb_results = {}
        for key, value in running_results.items():
            wandb_results[f"Training {key}"] = value / ep_train_steps
        if train_pl and running_pl_results is not None:
            for key, value in running_pl_results.items():
                wandb_results[f"PseudoLabel/Training {key}"] = value / len(train_pl_loader)
        for key, value in val_results.items():
            wandb_results[f"Validation {key}"] = value
        wandb.log(wandb_results)

        ## Semi Supervise: Pseudo labeling
        if pl_model_thres is not None and val_results["Segmentation IoU Score"] > pl_model_thres and epoch + 1 < epochs:
            if config["Validation"]["ema"] != 0:
                ## Predict on unlabedDataset to get pseudo labels
                df = generate_pl(config, ema_model, device, unlabeled_loader, model_name, df, pl_dir)
            else:
                df = generate_pl(config, model, device, unlabeled_loader, model_name, df, pl_dir)
            print(df.tail())
            ## Save csv with img folder directory, image names, pl path and confidence
            df_csv_dir = os.path.join(pl_dir, "pl.csv")
            df.to_csv(df_csv_dir, index=False)  ## ./pseudo_label/model_1/pl.csv
            if len(df.index) > 0:
                ## Rest pseudo_dataset and loader
                if train_pl_loader is None:
                    train_transform = AugmentationImgMaskOnly(
                        resized_crop=True, color_jitter=True, horizontal_flip=True, image_size=config["Model"]["image_size"]
                    )
                    pseudo_dataset = PseudoDataset(df_csv_dir, transform=train_transform, time_window=config["Model"]["time_window"])
                    train_pl_loader = DataLoader(pseudo_dataset, batch_size=config["Train"]["batch_size"], shuffle=True, drop_last=True)
                else:
                    train_pl_loader.dataset.update_df_from_csv()
                    print(f"updated pl dataset length:{len(train_pl_loader.dataset)}")  ## check

        ## start to train pseudo labels if there is at least a batch of PL
        if pl_model_thres is not None and len(df.index) >= config["Train"]["batch_size"]:
            train_pl = True

        model.train()

    # End the logging
    print("Finished Training!")


def main(config):
    # --------------------------------------------------------------------------
    # Model Type
    # --------------------------------------------------------------------------
    retina_head = config["Train"]["retina_head"]  # with the detection head or not
    if retina_head:
        model_name = "Video-Retina-UNETR"
    else:
        model_name = "Video-UNETR"
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # System Configuration
    # --------------------------------------------------------------------------
    # Set random seed for reproducibility
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    # torch.backends.cudnn.benchmark = True

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    print(device_name)

    # Logging
    os.makedirs("./logs", exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        f"./logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{model_name}_exp_{config['Train']['experiment_id']}.log"
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Model Configuration & Initialization
    # --------------------------------------------------------------------------
    # Create the model
    if model_name == "Video-UNETR":
        model = VideoUnetr(
            img_size=config["Model"]["image_size"],
            patch_size=config["Model"]["patch_size"],
            in_chans=config["Model"]["time_window"],
            embed_dim=config["Model"]["embed_dim"],
            depth=config["Model"]["depth"],
            num_heads=config["Model"]["num_heads"],
            mlp_ratio=config["Model"]["mlp_ratio"],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            skip_chans=config["Model"]["skip_chans"],
            out_chans=1,
        )
    elif model_name == "Video-Retina-UNETR":
        model = VideoRetinaUnetr(
            img_size=config["Model"]["image_size"],
            patch_size=config["Model"]["patch_size"],
            in_chans=config["Model"]["time_window"],
            embed_dim=config["Model"]["embed_dim"],
            depth=config["Model"]["depth"],
            num_heads=config["Model"]["num_heads"],
            mlp_ratio=config["Model"]["mlp_ratio"],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            skip_chans=config["Model"]["skip_chans"],
            out_chans=1,
            det_num_classes=config["Train"]["det_num_classes"],
            det_with_aqe=config["Train"]["with_aqe"],
        )

    # load pretrained ViT weights
    vit_pretrained_weights = "MAE ImageNet 1k"
    # checkpoint download page link: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

    if vit_pretrained_weights == "MAE ImageNet 1k":
        if config["Model"]["time_window"] % 3 != 0 and config["Model"]["time_window"] != 1:
            raise ValueError("The time window must be 1 or a multiple of 3 for loading the pretrained weights.")
        elif config["Model"]["time_window"] == 3:
            checkpoint = torch.load("./mae_pretrain_vit_base_checkpoints/mae_pretrain_vit_base.pth")
            model.load_state_dict(checkpoint["model"], strict=False)
            print(f"{vit_pretrained_weights} pretrained weights loaded!")
        else:
            checkpoint = torch.load("./mae_pretrain_vit_base_checkpoints/mae_pretrain_vit_base.pth")
            # load pretrained weights layer by layer
            for key, value in checkpoint["model"].items():
                if key == "patch_embed.proj.weight":
                    # repeat the weights for the time window in patch linear projection layer
                    print(f"Copy {key} with shape {value.shape} to {model.patch_embed.proj.weight.shape}")
                    num_copy = config["Model"]["time_window"] // 3
                    model.state_dict()[key].copy_((value / num_copy).repeat(1, num_copy, 1, 1))
                else:
                    model.state_dict()[key].copy_(value)
            print(f"{vit_pretrained_weights} pretrained weights loaded!")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Dataset Construction
    # --------------------------------------------------------------------------
    train_loader, valid_loader, medium_test_loader, hard_test_loader = construct_datasets(config)

    visualize = config["Train"]["visualize"]  # Set to True to visualize the training data & model output

    # Show the first batch of training data
    if visualize:
        print("Visualizing the first batch of training data...")
        time_window = config["Model"]["time_window"]
        buffer_num_sample = config["Train"]["buffer_num_sample"]
        for buffer in train_loader:
            for t in range(buffer_num_sample):  # iterate over the buffer to get samples
                images = buffer["images"][:, t : t + time_window, :, :]  # [N, T, H, W]
                masks = buffer["masks"][:, t : t + time_window, :, :]  # [N, T, H, W]
                cals = buffer["cals"][:, t : t + time_window, :]  # [N, T, 4]
                endpoints = buffer["endpoints"][:, t : t + time_window, :]  # [N, T, 2]
                labels = buffer["labels"][:, t : t + time_window]  # [N, T]
                show_dataset_samples(images, masks, cals, endpoints, labels, figsize=(8, 8), font_size=10)
                break
            break
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Training Configuration
    # --------------------------------------------------------------------------
    # effective batch size & learning rate
    eff_batch_size = config["Train"]["accumulation_steps"] * config["Train"]["batch_size"]
    lr = config["Train"]["blr"] * eff_batch_size / 256

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=config["Train"]["epochs"])

    # loss functions
    seg_focal_loss = SegFocalLoss(alpha=0.75, gamma=2).to(device)
    seg_dice_loss = SegDiceLoss().to(device)
    seg_ft_loss = SegFocalTverskyLoss().to(device)
    seg_iou_loss = SegIoULoss().to(device)
    if model_name == "Video-Retina-UNETR":
        siou_loss = SIoULoss() if config["Train"]["siou_loss"] else None
        aqe_loss = AQELoss() if config["Train"]["with_aqe"] else None
        det_loss = DetLoss(siou_loss=siou_loss, aqe_loss=aqe_loss).to(device)
    else:
        det_loss = None

    # --------------------------------------------------------------------------
    # wandb Logging
    # --------------------------------------------------------------------------
    # Set the name of the experiment for wandb logging
    name = f"{model_name}-{config['Train']['experiment_id']}"  ## Experiment ID for logging

    # number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())

    ## pseudo label config
    if config["Semi_Supervise"]["Pseudo_label"]["model_thres"] is None:
        model_thres, pl_thres, version = None, None, None
    else:
        model_thres = config["Semi_Supervise"]["Pseudo_label"]["model_thres"]
        pl_thres = config["Semi_Supervise"]["Pseudo_label"]["mask_thres"]
        version = config["Semi_Supervise"]["Pseudo_label"]["pl_version"]

    # Start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="DLMI Final",
        # Set the experiment name
        name=name,
        # track hyperparameters and run metadata
        config={
            "Architecture": model_name,
            "Time Window": config["Model"]["time_window"],
            "Number of Samples in Buffer": config["Train"]["buffer_num_sample"],
            "ViT Pre-Trained Weights": vit_pretrained_weights,
            "Skip Connection Channels": f"{config['Model']['skip_chans']}",
            "Number of Parameters": num_params,
            "Training Dataset": "easy_2220, easy_1429, easy_1063, easy_270, easy_1295, easy_1058, easy_284, medium_1903,",
            "Validation Dataset": "easy_2097, easy_679, easy_1530, easy_1148,",
            "Image Size": config["Model"]["image_size"],
            "Device Batch Size": config["Train"]["batch_size"],
            "Effective Batch Size": eff_batch_size,
            "Epochs": config["Train"]["epochs"],
            "Base Learning Rate": config["Train"]["blr"],
            "Learning Rate Scheduler": "Linear Decay",
            "Optimizer": "AdamW",
            "Detection Needle Classes": config["Train"]["det_num_classes"],
            "Theta Regression Reference": "Horizontal Orientation",
            "Angle Quality Estimation": config["Train"]["with_aqe"],
            "Loss Functions": config["Train"]["loss"],  # dice + focal + cls_focal + reg_l1,
            "PL_model_thres": model_thres,
            "PL_thres": pl_thres,
            "PL_version": version,
            "EMA": config["Validation"]["ema"],
        },
    )
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Train the model
    # --------------------------------------------------------------------------
    # Create checkpoint directory
    if model_name == "Video-UNETR":
        checkpoint_dir = "./video_unetr_checkpoints"
    elif model_name == "Video-Retina-UNETR":
        checkpoint_dir = "./video_retina_unetr_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train(
        config=config,
        model=model,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        seg_focal_loss=seg_focal_loss,
        seg_dice_loss=seg_dice_loss,
        seg_ft_loss=seg_ft_loss,
        seg_iou_loss=seg_iou_loss,
        model_name=model_name,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        det_loss=det_loss,
        scheduler=scheduler,
        visualize=visualize,
    )

    # Finish the wandb logging
    wandb.finish()
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    main(config)
