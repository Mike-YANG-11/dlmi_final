# Main script to train the Video-UNETR model

import datetime
import logging
import os
import random

import numpy as np

import torch
import torch.nn as nn

import wandb

from functools import partial

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm

from dataset import CustomDataset, Augmentation
from evaluation import evaluate, dice_score, iou_score, recall_score, precision_score
from visualization import show_image_pairs
from video_unetr import VideoUnetr
from loss import SegFocalLoss, SegDiceLoss

import json


# Construct the datasets
def construct_datasets(config):
    image_size = config["Model"]["image_size"]
    batch_size = config["Train"]["batch_size"]
    t = config["Model"]["time_window"]

    train_transform = Augmentation(color_jitter=True, resized_crop=True, horizontal_flip=True, image_size=image_size)
    valid_transform = Augmentation(color_jitter=False, resized_crop=False, horizontal_flip=False, image_size=image_size)
    
    train_dataset_list = []
    for folder_name in config["Data"]["Train_folder"].values():
        subdataset = CustomDataset(os.path.join(config["Data"]["folder_dir"],folder_name), transform=train_transform, time_window=t)
        train_dataset_list.append(subdataset)
    
    valid_dataset_list = []
    for folder_name in config["Data"]["Val_folder"].values():
        subdataset = CustomDataset(os.path.join(config["Data"]["folder_dir"],folder_name), transform=valid_transform, time_window=t)
        valid_dataset_list.append(subdataset)

    test_med_dataset_list = []
    for folder_name in config["Data"]["Test_folder"]["Medium"].values():
        subdataset = CustomDataset(os.path.join(config["Data"]["folder_dir"],folder_name), transform=valid_transform, time_window=t)
        test_med_dataset_list.append(subdataset)
    
    test_hard_dataset_list = []
    for folder_name in config["Data"]["Test_folder"]["Hard"].values():
        subdataset = CustomDataset(os.path.join(config["Data"]["folder_dir"],folder_name), transform=valid_transform, time_window=t)
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
    experiment_id,
    model,
    train_loader,
    valid_loader,
    optimizer,
    focal_loss,
    dice_loss,
    device,
    epochs,
    logger,
    accumulation_steps=8,
    scheduler=None,
    visualize=True,
):
    model.train()
    model.to(device)

    logger.info(f"Experiment ID: {experiment_id}")

    # Initialize variables to store the best validation loss & score
    best_val_results = {
        "Loss": float("inf"),
        "Focal Loss": float("inf"),
        "Dice Loss": float("inf"),
        "Dice Score": 0.0,
        "IoU Score": 0.0,
        "Recall Score": 0.0,
        "Precision Score": 0.0,
        "EA Score": 0.0,
        "Line Recall Score": 0.0,
        "Line Precision Score": 0.0,
        "Line Specificity Score": 0.0,
    }

    early_stop_count = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Reset running loss & score
        running_results = {
            "Loss": 0.0,
            "Focal Loss": 0.0,
            "Dice Loss": 0.0,
            "Dice Score": 0.0,
            "IoU Score": 0.0,
            "Recall Score": 0.0,
            "Precision Score": 0.0,
        }

        for step, samples in enumerate(tqdm(train_loader)):
            images = samples["images"].to(device)
            masks = samples["masks"].to(device)

            # Forward pass
            preds = model(images)  # [N, 1, H, W]

            # Use the last frame as the target mask
            masks = masks[:, -1, :, :].unsqueeze(1)  # [N, 1, H, W]

            # Calculate loss & scores
            fl = focal_loss(preds, masks)
            dl = dice_loss(preds, masks)
            loss = fl + dl
            dscore = dice_score(preds, masks)
            iscore = iou_score(preds, masks)
            rscore = recall_score(preds, masks)
            pscore = precision_score(preds, masks)

            # update running loss & score
            running_results["Loss"] += loss.item()
            running_results["Focal Loss"] += fl.item()
            running_results["Dice Loss"] += dl.item()
            running_results["Dice Score"] += dscore.item()
            running_results["IoU Score"] += iscore.item()
            running_results["Recall Score"] += rscore.item()
            running_results["Precision Score"] += pscore.item()

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
        val_results = evaluate(model, valid_loader, focal_loss, dice_loss, device)

        # Print running loss on training data & loss on validation data
        print("--------------------------------------------------")
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {running_results['Loss']/len(train_loader):.6f}")
        print(f"Validation Loss: {val_results['Loss']:.6f}")

        # Save the best model & visualize the output if the model is improved
        if val_results["Loss"] < best_val_results["Loss"]:
            # Update the best validation loss & IoU score
            best_val_results = val_results

            # Reset early stopping count
            early_stop_count = 0

            # Save the best model
            torch.save(model.state_dict(), f"./video_unetr_checkpoints/video_unetr_{experiment_id}.pth")
            print("The best model is saved!")

            if visualize:
                # Visualize the output on training data (show the first batch)
                print("Visualizing the output on training data...")
                model.eval()
                with torch.no_grad():
                    for _, vis_samples in enumerate(train_loader):
                        # Move to device & forward pass
                        train_images = vis_samples["images"].to(device)
                        train_masks = vis_samples["masks"]
                        train_preds = model(train_images)

                        # Detach and move to CPU
                        train_images = train_images.detach().cpu()
                        train_preds = train_preds.detach().cpu()

                        # Visualize the output
                        show_image_pairs(consec_images=train_images, consec_masks=train_masks, preds=train_preds)
                        break

                # Visualize the output on validation data (show the first batch)
                print("Visualizing the output on validation data...")
                model.eval()
                with torch.no_grad():
                    for _, val_samples in enumerate(valid_loader):
                        # Move to device & forward pass
                        val_images = val_samples["images"].to(device)
                        val_masks = val_samples["masks"]
                        val_preds = model(val_images)

                        # Detach and move to CPU
                        val_images = val_images.detach().cpu()
                        val_preds = val_preds.detach().cpu()

                        # Visualize the output
                        show_image_pairs(consec_images=val_images, consec_masks=val_masks, preds=val_preds)
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

    visualize = config["Train"]["visualize"]  # Set to True to visualize the training data & model output

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
        f"./logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_exp_{config['Train']['experiment_id']}.log"
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Construct the datasets
    train_loader, valid_loader, medium_test_loader, hard_test_loader = construct_datasets(config)

    # Show the first batch of training data
    if visualize:
        print("Visualizing the first batch of training data...")
        for _, samples in enumerate(train_loader):
            show_image_pairs(consec_images=samples["images"], consec_masks=samples["masks"])
            break

    # Create the model
    video_unetr = VideoUnetr(
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
        video_unetr.load_state_dict(checkpoint["model"], strict=False)
        print(f"{vit_pretrained_weights} pretrained weights loaded!")

    # Create checkpoint directory
    os.makedirs("./video_unetr_checkpoints", exist_ok=True)

    # number of parameters in the model
    num_params = sum(p.numel() for p in video_unetr.parameters())

    eff_batch_size = config["Train"]["accumulation_steps"] * config["Train"]["batch_size"]
    lr = config["Train"]["blr"] * eff_batch_size / 256

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(video_unetr.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=config["Train"]["epochs"])

    # loss functions
    seg_focal_loss = SegFocalLoss(alpha=0.75, gamma=2).to(device)
    seg_dice_loss = SegDiceLoss().to(device)

    # Set the name of the experiment for wandb logging
    name = f"Video-UNETR-{config['Train']['experiment_id']}"  ## Experiment ID for logging

    # Start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="DLMI Final",
        # Set the experiment name
        name=name,
        # track hyperparameters and run metadata
        config={
            "Architecture": "Video-UNETR",
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
            "Loss Functions": "Focal Loss + Dice Loss",
        },
    )

    # Train the model
    train(
        config["Train"]["experiment_id"],
        video_unetr,
        train_loader,
        valid_loader,
        optimizer,
        seg_focal_loss,
        seg_dice_loss,
        device,
        config["Train"]["epochs"],
        logger,
        accumulation_steps=config["Train"]["accumulation_steps"],
        scheduler=scheduler,
        visualize=visualize,
    )

    # Finish the wandb logging
    wandb.finish()


if __name__ == "__main__":
    with open('config.json', 'r', encoding="utf-8") as f:
        config = json.load(f)

    main(config)
