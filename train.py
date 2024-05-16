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
from loss import FocalLoss, DiceLoss


# Construct the datasets
def construct_datasets(image_size, batch_size, t):
    train_transform = Augmentation(color_jitter=True, resized_crop=True, horizontal_flip=True, image_size=image_size)
    valid_transform = Augmentation(color_jitter=False, resized_crop=False, horizontal_flip=True, image_size=image_size)

    """ easy datasets """
    # 20231017_08
    easy_2220 = CustomDataset("./data/Sonosite_20231017_0833_2220frames_abc", transform=train_transform, time_window=t)
    # easy_1941 = CustomDataset("./data/Sonosite_20231017_0836_1941frames_abc")

    # 20231107_14
    easy_2097 = CustomDataset(
        "./data/Sonosite_20231107_1456_2097frames_abc", transform=valid_transform, time_window=t
    )  # validation set

    # 20231128_12
    easy_1429 = CustomDataset("./data/Sonosite_20231128_1220_1429frames_abc", transform=train_transform, time_window=t)
    easy_1063 = CustomDataset("./data/Sonosite_20231128_1259_1063frames_abc", transform=train_transform, time_window=t)

    # 20231128_14
    easy_270 = CustomDataset("./data/Sonosite_20231128_1423-1_270frames_abc", transform=train_transform, time_window=t)
    easy_1295 = CustomDataset(
        "./data/Sonosite_20231128_1423-2_1295frames_abc", transform=train_transform, time_window=t
    )
    easy_1058 = CustomDataset("./data/Sonosite_20231128_1429_1058frams_abc", transform=train_transform, time_window=t)

    # 20231201_09
    easy_679 = CustomDataset(
        "./data/Sonosite_20231201_0910_679frams_abc", transform=valid_transform, time_window=t
    )  # validation set
    easy_1530 = CustomDataset(
        "./data/Sonosite_20231201_0913_1530frames_abc", transform=valid_transform, time_window=t
    )  # validation set
    easy_1148 = CustomDataset(
        "./data/Sonosite_20231201_0921_1148frames_abc", transform=valid_transform, time_window=t
    )  # validation set

    """ easy & medium datasets """
    # 20231110_09
    easy_284 = CustomDataset("./data/Sonosite_20231110_0903-2_284frames_abc", transform=train_transform, time_window=t)
    medium_1903 = CustomDataset(
        "./data/Sonosite_20231110_0910_1903frames_abc", transform=train_transform, time_window=t
    )

    """ medium datasets """
    # 20231003_09
    medium_1120 = CustomDataset(
        "./data/Sonosite_20231003_0945_1120frames_abc", transform=valid_transform, time_window=t
    )  # test set

    # 20231003_10
    medium_755 = CustomDataset(
        "./data/Sonosite_20231003_1028_755frames_abc", transform=valid_transform, time_window=t
    )  # test set

    # 20231024_09
    medium_990 = CustomDataset(
        "./data/Sonosite_20231024_0909_990frames_abc", transform=valid_transform, time_window=t
    )  # test set

    # 20231024_12
    medium_1704 = CustomDataset(
        "./data/Sonosite_20231024_1228_1704frames_abc", transform=valid_transform, time_window=t
    )  # test set

    """ hard datasets """
    # 20230613_10
    hard_574 = CustomDataset(
        "./data/Sonosite_20230613_1033_574frames_abc", transform=valid_transform, time_window=t
    )  # test set

    # 20230613_13
    hard_549 = CustomDataset(
        "./data/Sonosite_20230613_1329_549frames_abc", transform=valid_transform, time_window=t
    )  # test set

    # 20231024_15
    hard_412 = CustomDataset(
        "./data/Sonosite_20231024_1520_412frames_abc", transform=valid_transform, time_window=t
    )  # test set

    # 20231031_13
    hard_950 = CustomDataset(
        "./data/Sonosite_20231031_1322_950frames_abc", transform=valid_transform, time_window=t
    )  # test set
    hard_1320 = CustomDataset(
        "./data/Sonosite_20231031_1345_1320frames_abc", transform=valid_transform, time_window=t
    )  # test set

    """ training dataset """
    train_dataset = ConcatDataset(
        [
            easy_2220,
            easy_1429,
            easy_1063,
            easy_270,
            easy_1295,
            easy_1058,
            easy_284,
            medium_1903,
        ]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    """ validation dataset """
    valid_dataset = ConcatDataset(
        [
            easy_2097,
            easy_679,
            easy_1530,
            easy_1148,
        ]
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    """ medium test dataset """
    medium_test_dataset = ConcatDataset(
        [
            medium_1120,
            medium_755,
            medium_990,
            medium_1704,
        ]
    )
    medium_test_loader = DataLoader(medium_test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    """ hard test dataset """
    hard_test_dataset = ConcatDataset(
        [
            hard_574,
            hard_549,
            hard_412,
            hard_950,
            hard_1320,
        ]
    )
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

        for step, (images, masks) in enumerate(tqdm(train_loader)):
            images, masks = images.to(device), masks.to(device)

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
                    for _, (train_images, train_masks) in enumerate(train_loader):
                        # Move to device & forward pass
                        train_images = train_images.to(device)
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
                    for _, (val_images, val_masks) in enumerate(valid_loader):
                        # Move to device & forward pass
                        val_images = val_images.to(device)
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


def main():
    # Set Experiment ID for logging
    experiment_id = 0

    # Basic settings
    image_size = 224
    batch_size = 8
    t = 3  # Time window size
    visualize = True  # Set to True to visualize the training data & model output

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
        f"./logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_exp_{experiment_id}.log"
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Construct the datasets
    train_loader, valid_loader, medium_test_loader, hard_test_loader = construct_datasets(image_size, batch_size, t)

    # Show the first batch of training data
    if visualize:
        print("Visualizing the first batch of training data...")
        for _, (images, masks) in enumerate(train_loader):
            show_image_pairs(consec_images=images, consec_masks=masks)
            break

    # Set the name of the experiment for wandb logging
    name = f"Video-UNETR-{experiment_id}"

    # Define the model parameters
    patch_size = 16
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4
    skip_chans = [64, 128, 256]

    # Create the model
    video_unetr = VideoUnetr(
        img_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        skip_chans=skip_chans,
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

    # training epochs
    epochs = 20

    # learning rate & optimizer
    blr = 1.0e-4
    accumulation_steps = 8
    eff_batch_size = accumulation_steps * batch_size
    lr = blr * eff_batch_size / 256

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(video_unetr.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=epochs)

    # loss functions
    focal_loss = FocalLoss(alpha=0.75, gamma=2).to(device)
    dice_loss = DiceLoss().to(device)

    # Start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="DLMI Final",
        # Set the experiment name
        name=name,
        # track hyperparameters and run metadata
        config={
            "Architecture": "Video-UNETR",
            "Time Window": t,
            "ViT Pre-Trained Weights": vit_pretrained_weights,
            "Skip Connection Channels": f"{skip_chans}",
            "Number of Parameters": num_params,
            "Training Dataset": "easy_2220, easy_1429, easy_1063, easy_270, easy_1295, easy_1058, easy_284, medium_1903,",
            "Validation Dataset": "easy_2097, easy_679, easy_1530, easy_1148,",
            "Image Size": image_size,
            "Device Batch Size": batch_size,
            "Effective Batch Size": eff_batch_size,
            "Epochs": epochs,
            "Base Learning Rate": blr,
            "Learning Rate Scheduler": "Linear Decay",
            "Optimizer": "AdamW",
            "Loss Functions": "Focal Loss + Dice Loss",
        },
    )

    # Train the model
    train(
        experiment_id,
        video_unetr,
        train_loader,
        valid_loader,
        optimizer,
        focal_loss,
        dice_loss,
        device,
        epochs,
        logger,
        accumulation_steps=accumulation_steps,
        scheduler=scheduler,
        visualize=visualize,
    )

    # Finish the wandb logging
    wandb.finish()


if __name__ == "__main__":
    main()
