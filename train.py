# Main script to train the Video-UNETR model

import datetime
import json
import logging
import os
import random

import numpy as np

import torch
import torch.nn as nn

import wandb

from anchors import AnchorGenerator

from functools import partial

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm

from dataset import CustomDataset, Augmentation
from evaluation import evaluate, seg_dice_score, seg_iou_score, results_dictioanry
from visualization import show_dataset_samples, show_seg_preds_only
from model import VideoUnetr, VideoRetinaUnetr
from loss import SegFocalLoss, SegDiceLoss, SIoULoss, DetLoss


# Construct the datasets
def construct_datasets(config):
    image_size = config["Model"]["image_size"]
    batch_size = config["Train"]["batch_size"]
    t = config["Model"]["time_window"]

    train_transform = Augmentation(resized_crop=True, color_jitter=True, horizontal_flip=True, image_size=image_size)
    valid_transform = Augmentation(resized_crop=False, color_jitter=False, horizontal_flip=False, image_size=image_size)

    train_dataset_list = []
    for folder_name in config["Data"]["Train_folder"].values():
        subdataset = CustomDataset(os.path.join(config["Data"]["folder_dir"], folder_name), transform=train_transform, time_window=t)
        train_dataset_list.append(subdataset)

    valid_dataset_list = []
    for folder_name in config["Data"]["Val_folder"].values():
        subdataset = CustomDataset(os.path.join(config["Data"]["folder_dir"], folder_name), transform=valid_transform, time_window=t)
        valid_dataset_list.append(subdataset)

    test_med_dataset_list = []
    for folder_name in config["Data"]["Test_folder"]["Medium"].values():
        subdataset = CustomDataset(os.path.join(config["Data"]["folder_dir"], folder_name), transform=valid_transform, time_window=t)
        test_med_dataset_list.append(subdataset)

    test_hard_dataset_list = []
    for folder_name in config["Data"]["Test_folder"]["Hard"].values():
        subdataset = CustomDataset(os.path.join(config["Data"]["folder_dir"], folder_name), transform=valid_transform, time_window=t)
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


# Training Function
def train(
    model,
    device,
    epochs,
    accumulation_steps,
    train_loader,
    valid_loader,
    optimizer,
    seg_focal_loss,
    seg_dice_loss,
    experiment_id,
    model_name,
    logger,
    checkpoint_dir,
    det_loss=None,
    scheduler=None,
    visualize=True,
):
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

        for step, samples in enumerate(tqdm(train_loader)):
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
                vis_pred_masks, classifications, regressions = model(images)
                # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4]
            else:
                vis_pred_masks = model(images)  # [N, 1, H, W]

            # Calculate loss (use the last frame as the target mask)
            masks = masks[:, -1, :, :].unsqueeze(1)  # [N, 1, H, W]
            fl = seg_focal_loss(vis_pred_masks, masks)
            dl = seg_dice_loss(vis_pred_masks, masks)
            if model_name == "Video-Retina-UNETR":  # with the detection head
                annotations = annotations[:, -1, :].unsqueeze(1)  # [N, 1, 5]
                cl, rl = det_loss(classifications, regressions, anchors_pos, annotations)

            # Calculate total loss
            loss = fl + dl
            if model_name == "Video-Retina-UNETR":  # with the detection head
                loss = loss + 0.3 * cl + rl  ## TODO: adaptively modify the weight for the detection loss

            # Calculate the segmentation Dice score & IoU score
            seg_dscore = seg_dice_score(vis_pred_masks, masks)
            seg_iscore = seg_iou_score(vis_pred_masks, masks)

            # update running loss & score
            running_results["Loss"] += fl.item() + dl.item()
            running_results["Segmentation Focal Loss"] += fl.item()
            running_results["Segmentation Dice Loss"] += dl.item()
            running_results["Segmentation Dice Score"] += seg_dscore.item()
            running_results["Segmentation IoU Score"] += seg_iscore.item()
            if model_name == "Video-Retina-UNETR":  # with the detection head
                running_results["Loss"] += cl.item() + rl.item()
                running_results["Detection Classification Loss"] += cl.item()
                running_results["Detection Regression Loss"] += rl.item()

            # Gradient accumulation normalization
            loss = loss / accumulation_steps
            loss.backward()

            # Update weights
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if scheduler is not None:
            # Adjust learning rate
            scheduler.step()

        # Evaluate the model on validation data
        if model_name == "Video-Retina-UNETR":
            val_results = evaluate(model, device, valid_loader, seg_focal_loss, seg_dice_loss, model_name, det_loss, anchors_pos)
        else:
            val_results = evaluate(model, device, valid_loader, seg_focal_loss, seg_dice_loss, model_name)

        # Print running loss on training data & loss on validation data
        print("--------------------------------------------------")
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {running_results['Loss']/len(train_loader):.6f}")
        print(f"Validation Loss: {val_results['Loss']:.6f}")

        # Save the best model & visualize the output if the model is improved
        if (
            val_results["Loss"] < best_val_results["Loss"]
            or val_results["Segmentation Dice Loss"] < best_val_results["Segmentation Dice Loss"]
            or val_results["Segmentation Needle EA Score"] > best_val_results["Segmentation Needle EA Score"]
        ):
            # Update the best validation loss & IoU score
            best_val_results = val_results

            # Reset early stopping count
            early_stop_count = 0

            # Save the best model
            if model_name == "Video-Retina-UNETR":
                save_path = os.path.join(checkpoint_dir, f"video_retina_unetr_{experiment_id}.pth")
            else:
                save_path = os.path.join(checkpoint_dir, f"video_unetr_{experiment_id}.pth")
            torch.save(model.state_dict(), save_path)
            print("The best model is saved!")

            if visualize:
                # Visualize the output on training data (show the first batch)
                print("Visualizing the output on training data...")
                model.eval()
                with torch.no_grad():
                    for _, vis_samples in enumerate(train_loader):
                        # Move to device & forward pass
                        vis_images = vis_samples["images"].to(device)
                        vis_masks = vis_samples["masks"]

                        # Forward pass
                        if model_name == "Video-Retina-UNETR":
                            vis_pred_masks, classifications, regressions = model(vis_images)
                            # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4]
                        else:
                            vis_pred_masks = model(vis_images)  # [N, 1, H, W]

                        # Detach and move to CPU
                        vis_images = vis_images.detach().cpu()
                        vis_pred_masks = vis_pred_masks.detach().cpu()

                        # Visualize the output
                        show_seg_preds_only(consec_images=vis_images, consec_masks=vis_masks, preds=vis_pred_masks)
                        break

                # Visualize the output on validation data (show the first batch)
                print("Visualizing the output on validation data...")
                with torch.no_grad():
                    for _, vis_samples in enumerate(valid_loader):
                        # Move to device & forward pass
                        vis_images = vis_samples["images"].to(device)
                        vis_masks = vis_samples["masks"]

                        # Forward pass
                        if model_name == "Video-Retina-UNETR":
                            vis_pred_masks, classifications, regressions = model(vis_images)
                            # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4]
                        else:
                            vis_pred_masks = model(vis_images)  # [N, 1, H, W]

                        # Detach and move to CPU
                        vis_images = vis_images.detach().cpu()
                        vis_pred_masks = vis_pred_masks.detach().cpu()

                        # Visualize the output
                        show_seg_preds_only(consec_images=vis_images, consec_masks=vis_masks, preds=vis_pred_masks)
                        break
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
            logger.info(f"{key}: {value/len(train_loader):.6f}")
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
            wandb_results[f"Training {key}"] = value / len(train_loader)
        for key, value in val_results.items():
            wandb_results[f"Validation {key}"] = value
        wandb.log(wandb_results)

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
            embed_dim=config["Model"]["embed_dim"],
            depth=config["Model"]["depth"],
            num_heads=config["Model"]["num_heads"],
            mlp_ratio=config["Model"]["mlp_ratio"],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            skip_chans=config["Model"]["skip_chans"],
            out_chans=1,
        )

    # load pretrained ViT weights
    vit_pretrained_weights = "MAE ImageNet 1k"

    if vit_pretrained_weights == "MAE ImageNet 1k":
        checkpoint = torch.load(
            "./mae_pretrain_vit_base_checkpoints/mae_pretrain_vit_base.pth"
        )  # download page link: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
        model.load_state_dict(checkpoint["model"], strict=False)
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
        for _, samples in enumerate(train_loader):
            show_dataset_samples(
                samples["images"],
                samples["masks"],
                samples["cals"],
                samples["endpoints"],
                samples["labels"],
                figsize=(8, 8),
                font_size=10,
            )
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
    if model_name == "Video-Retina-UNETR":
        det_loss = DetLoss(alpha=0.25, gamma=2.0, siou_loss=SIoULoss()).to(device)
    else:
        det_loss = None

    # --------------------------------------------------------------------------
    # wandb Logging
    # --------------------------------------------------------------------------
    # Set the name of the experiment for wandb logging
    name = f"{model_name}-{config['Train']['experiment_id']}"  ## Experiment ID for logging

    # number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())

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
        model=model,
        device=device,
        epochs=config["Train"]["epochs"],
        accumulation_steps=config["Train"]["accumulation_steps"],
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        seg_focal_loss=seg_focal_loss,
        seg_dice_loss=seg_dice_loss,
        experiment_id=config["Train"]["experiment_id"],
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
