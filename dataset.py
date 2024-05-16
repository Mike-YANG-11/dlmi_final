# Custom Dataset Class

import os
import random

import cv2

import torch
import torch.nn as nn
import torchvision.transforms as tf

from torch.utils.data import Dataset


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dir_path, transform=None, time_window=3):
        """
        Args:
            root (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir_path = dir_path
        self.time_window = time_window
        self.file_names = sorted(os.listdir(self.dir_path))
        self.image_names = [
            f for f in self.file_names if f[0] == "a" and f.endswith(".jpg")
        ]  # ["a0001.jpg", "a0002.jpg", ...]
        self.mask_names = [
            "m" + f[1:-4] + "_lw_20.png" for f in self.image_names
        ]  # ["m0001_lw_20.png", "m0002_lw_20.png", ...]

        # creat T consecutive image & mask names list
        self.consec_images_names = [
            self.image_names[i : i + self.time_window] for i in range(0, len(self.image_names) - self.time_window + 1)
        ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg"], ["a0002.jpg", "a0003.jpg", "a0004.jpg"], ...]
        self.consec_masks_names = [
            self.mask_names[i : i + self.time_window] for i in range(0, len(self.mask_names) - self.time_window + 1)
        ]  # [["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png"], ["m0002_lw_20.png", "m0003_lw_20.png", "m0004_lw_20.png"], ...]

        self.transform = transform

    def __len__(self):
        return len(self.consec_images_names)

    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # ["a0001.jpg", "a0002.jpg", "a0003.jpg"]
        consec_images = [
            cv2.imread(os.path.join(self.dir_path, image_name), cv2.IMREAD_GRAYSCALE)
            for image_name in consec_images_name
        ]

        consec_masks_name = self.consec_masks_names[idx]  # ["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png"]
        consec_masks = [
            cv2.imread(os.path.join(self.dir_path, mask_name), cv2.IMREAD_GRAYSCALE) for mask_name in consec_masks_name
        ]

        # To tensor
        for i in range(self.time_window):
            consec_images[i] = torch.tensor(consec_images[i], dtype=torch.float32)
            consec_masks[i] = torch.tensor(consec_masks[i], dtype=torch.float32)

        # Stack
        consec_images = torch.stack(consec_images, dim=0)  # [T, H, W]
        consec_masks = torch.stack(consec_masks, dim=0)  # [T, H, W]

        # Unsqueeze
        consec_images = consec_images.unsqueeze(1)  # [T, 1, H, W]
        consec_masks = consec_masks.unsqueeze(1)  # [T, 1, H, W]

        # Apply transform
        if self.transform:
            consec_images, consec_masks = self.transform(consec_images, consec_masks)

        # Sqeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]
        consec_masks = consec_masks.squeeze(1)  # [T, H, W]

        return consec_images, consec_masks


# Augmentation Class
class Augmentation(nn.Module):
    def __init__(self, color_jitter=True, resized_crop=True, horizontal_flip=True, image_size=224):
        self.color_jitter = color_jitter
        self.resized_crop = resized_crop
        self.horizontal_flip = horizontal_flip
        self.image_size = image_size

    def __call__(self, images, masks):
        """Resize"""
        images = tf.functional.resize(
            images, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR, antialias=True
        )
        masks = tf.functional.resize(
            masks, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.NEAREST
        )

        """ Normalization """
        images = images / 255.0
        masks = masks / 255.0

        """ Random Color Jitter"""
        if self.color_jitter and random.random() < 0.5:
            color_transform = tf.Compose(
                [
                    tf.ColorJitter(brightness=0.5, contrast=0.3),
                ]
            )
            images = color_transform(images)

        """ Random Resized Crop """
        if self.resized_crop and random.random() < 0.5:
            # Get parameters for RandomResizedCrop
            top, left, height, width = tf.RandomResizedCrop.get_params(images, scale=(0.7, 1.0), ratio=(0.8, 1.2))

            # Apply RandomResizedCrop
            images = tf.functional.resized_crop(
                images,
                top,
                left,
                height,
                width,
                (self.image_size, self.image_size),
                interpolation=tf.InterpolationMode.BILINEAR,
                antialias=True,
            )
            masks = tf.functional.resized_crop(
                masks,
                top,
                left,
                height,
                width,
                (self.image_size, self.image_size),
                interpolation=tf.InterpolationMode.NEAREST,
            )

        """ Random Horizontal Flip """
        if self.horizontal_flip and random.random() < 0.5:
            images = tf.functional.hflip(images)
            masks = tf.functional.hflip(masks)

        return images, masks
