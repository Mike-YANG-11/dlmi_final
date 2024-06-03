import json
import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import wandb

from anchors import AnchorGenerator

from functools import partial

from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm

from dataset import CustomDataset, Augmentation
from evaluation import evaluate_test, seg_dice_score, seg_iou_score
from visualization import show_dataset_samples, show_seg_preds_only, show_preds_with_det_head
from model import VideoUnetr, VideoRetinaUnetr
from loss import SegFocalLoss, SegDiceLoss


# Construct the test datasets
def construct_datasets(config, run):
    image_size = run.config["Image Size"]
    batch_size = run.config["Device Batch Size"]
    time_window = run.config["Time Window"]
    buffer_num_sample = run.config["Number of Samples in Buffer"]
    line_width = config["Data"]["line_width"]
    det_num_classes = run.config["Detection Needle Classes"]

    valid_transform = Augmentation(resized_crop=False, color_jitter=False, horizontal_flip=False, image_size=image_size)

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

    """ medium test dataset """
    medium_test_dataset = ConcatDataset(test_med_dataset_list)
    medium_test_loader = DataLoader(medium_test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    """ hard test dataset """
    hard_test_dataset = ConcatDataset(test_hard_dataset_list)
    hard_test_loader = DataLoader(hard_test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print(f"Number of medium test samples: {len(medium_test_dataset)}")
    print(f"Number of hard test samples: {len(hard_test_dataset)}")

    return medium_test_loader, hard_test_loader

def visualize_sample(run, model, loader, model_name, device, anchors_pos=None):
    time_window = run.config["Time Window"]
    buffer_num_sample = run.config["Number of Samples in Buffer"]
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
                        with_aqe=run.config["Angle Quality Estimation"],
                    )
                    anchors_pos = anchors_pos.cuda()
                else:
                    show_seg_preds_only(consec_images=vis_images, consec_masks=vis_masks, pred_masks=vis_pred_masks)

                break
            break
           

def main(config, args):
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

    # --------------------------------------------------------------------------
    # wandb Logging
    # --------------------------------------------------------------------------
    api = wandb.Api()
    run_id = args.run_id
    run = api.run(f"mikeyang900819/DLMI Final/{run_id}")
    print(f"Testing {run.name}")

    model_name = run.config["Architecture"]  # with the detection head or not

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
        if run.config["Time Window"] % 3 != 0 and run.config["Time Window"] != 1:
            raise ValueError("The time window must be 1 or a multiple of 3 for loading the pretrained weights.")
        elif run.config["Time Window"] == 3:
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
                    num_copy = run.config["Time Window"] // 3
                    model.patch_embed.proj.weight.data.copy_((value / num_copy).repeat(1, num_copy, 1, 1))
                else:
                    model.state_dict()[key].copy_(value)
            print(f"{vit_pretrained_weights} pretrained weights loaded!")
    
    ## key names in ema_model & model are slightly different
    if run.config["EMA"] != 0:
        ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config["Validation"]["ema"]))
        model = ema_model
    
    if model_name == "Video-Retina-UNETR":
        image_size = run.config["Image Size"]
        anchor_generator = AnchorGenerator()
        anchors_pos = anchor_generator([image_size, image_size])
    else:
        anchors_pos = None


    # --------------------------------------------------------------------------
    # Read ckpt
    # --------------------------------------------------------------------------
    model.load_state_dict(torch.load(args.ckpt_path))
    print(f'Ckpt loaded {args.ckpt_path}')
    
    # --------------------------------------------------------------------------
    # Dataset Construction
    # --------------------------------------------------------------------------
    medium_test_loader, hard_test_loader = construct_datasets(config, run)
    
    # --------------------------------------------------------------------------
    # Test
    # --------------------------------------------------------------------------
    test_med_result = evaluate_test(run, model, device, medium_test_loader, model_name, anchors_pos=anchors_pos, with_aqe=False, refined_mask=False)
    test_hard_result = evaluate_test(run, model, device, hard_test_loader, model_name, anchors_pos=anchors_pos, with_aqe=False, refined_mask=False)

    # --------------------------------------------------------------------------
    # Show result
    # --------------------------------------------------------------------------    
    
    ## Record result to wandb
    for metric in test_med_result:
        run.summary[f"Test Med {metric}"] = test_med_result[metric]
        run.summary[f"Test Hard {metric}"] = test_hard_result[metric]
    run.update()

    df = pd.DataFrame([test_med_result, test_hard_result]).T
    df.rename(columns={0: "Medium", 1: "Hard"}, inplace=True)
    print(df)

    visualize_sample(run, model, medium_test_loader, model_name, device, anchors_pos)
    visualize_sample(run, model, hard_test_loader, model_name, device, anchors_pos)

    wandb.finish()


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str)
    parser.add_argument("ckpt_path", type=str)
    args = parser.parse_args()
    main(config, args)

## python test.py "yryi2bdx" "./video_unetr_checkpoints/video_unetr_200new.pth"