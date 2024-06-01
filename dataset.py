# Custom Dataset Class

import os
import random

import cv2
import math

import torch
import torch.nn as nn
import torchvision.transforms as tf

from torch.utils.data import Dataset
from PIL import Image
import json

from torchvision.transforms import v2
from torchvision import tv_tensors

import pandas as pd

# from functools import lru_cache


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dir_path, transform=None, time_window=3, buffer_num_sample=8, line_width=20, b_thres=False, det_num_classes=1):
        """
        Args:
            root (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if line_width == 20 and b_thres:
            mask_name_suffix = "_bt_lw_20.png"
        elif line_width == 20:
            mask_name_suffix = "_lw_20.png"
        else:
            mask_name_suffix = ".png"

        self.dir_path = dir_path
        self.time_window = time_window
        self.buffer_num_sample = buffer_num_sample
        self.buffer_length = time_window + buffer_num_sample - 1
        self.file_names = sorted(os.listdir(self.dir_path))
        self.image_names = [f for f in self.file_names if f[0] == "a" and f.endswith(".jpg")]  # ["a0001.jpg", "a0002.jpg", ...]
        self.mask_names = ["m" + f[1:-4] + mask_name_suffix for f in self.image_names]  # ["m0001_lw_20.png", "m0002_lw_20.png", ...]
        self.json_names = [f.replace(".jpg", ".json") for f in self.image_names]  # ["a0001.json", "a0002.json", ...]

        # creat T consecutive image & mask names list
        if time_window == 1:  ## duplicate
            self.consec_images_names = [
                [self.image_names[i]] * 3 for i in range(0, len(self.image_names) - self.time_window + 1)
            ]  # [["a0001.jpg", "a0001.jpg", "a0001.jpg"], ["a0002.jpg", "a0002.jpg", "a0002.jpg"], ...]
            self.consec_masks_names = [
                [self.mask_names[i]] * 3 for i in range(0, len(self.mask_names) - self.time_window + 1)
            ]  # [["m0001_lw_20.png", "m0001_lw_20.png", "m0001_lw_20.png"], ["m0002_lw_20.png", "m0002_lw_20.png", "m0002_lw_20.png"], ...]
            self.consec_json_names = [[self.json_names[i]] * 3 for i in range(0, len(self.json_names) - self.time_window + 1)]
        # -------------------------------------------------------------------------------
        # buffer for speed up
        # -------------------------------------------------------------------------------
        else:
            total_num_buffer = 1 + math.ceil((len(self.image_names) - self.buffer_length) / self.buffer_num_sample)
            self.consec_images_names = [
                self.image_names[i * self.buffer_num_sample : i * self.buffer_num_sample + self.buffer_length] for i in range(0, total_num_buffer)
            ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg", ... ], ["a0003.jpg", "a0004.jpg", "a0005.jpg", ... ], ...]
            self.consec_masks_names = [
                self.mask_names[i * self.buffer_num_sample : i * self.buffer_num_sample + self.buffer_length] for i in range(0, total_num_buffer)
            ]  # [["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png", ... ], ["m0003_lw_20.png", "m0004_lw_20.png", "m0005_lw_20.png", ... ], ...]
            self.consec_json_names = [
                self.json_names[i * self.buffer_num_sample : i * self.buffer_num_sample + self.buffer_length] for i in range(0, total_num_buffer)
            ]  # [["a0001.json", "a0002.json", "a0003.json", ... ], ["a0003.json", "a0004.json", "a0005.json", ... ], ...]

            # drop the last buffer if the length is less than buffer_length (therefore the buffer_num_sample should be too large to avoid significant dataset drop)
            if len(self.consec_images_names[-1]) < self.buffer_length:
                self.consec_images_names = self.consec_images_names[:-1]
                self.consec_masks_names = self.consec_masks_names[:-1]
                self.consec_json_names = self.consec_json_names[:-1]
        # -------------------------------------------------------------------------------
        # original version
        # -------------------------------------------------------------------------------
        # else:
        #     self.consec_images_names = [
        #         self.image_names[i : i + self.time_window] for i in range(0, len(self.image_names) - self.time_window + 1)
        #     ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg"], ["a0002.jpg", "a0003.jpg", "a0004.jpg"], ...]
        #     self.consec_masks_names = [
        #         self.mask_names[i : i + self.time_window] for i in range(0, len(self.mask_names) - self.time_window + 1)
        #     ]  # [["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png"], ["m0002_lw_20.png", "m0003_lw_20.png", "m0004_lw_20.png"], ...]
        #     self.consec_json_names = [self.json_names[i : i + self.time_window] for i in range(0, len(self.json_names) - self.time_window + 1)]
        # -------------------------------------------------------------------------------

        self.transform = transform
        self.trans_totensor = tf.Compose([tf.ToTensor()])
        self.det_num_classes = det_num_classes

    def __len__(self):
        return len(self.consec_images_names)

    # @lru_cache(256)
    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # ["a0001.jpg", "a0002.jpg", "a0003.jpg"]
        consec_images = []
        fname_list = []
        for f_name in consec_images_name:
            img = Image.open(os.path.join(self.dir_path, f_name)).convert("L")
            img_tensor = self.trans_totensor(img)
            consec_images.append(img_tensor)
            fname_list.append(os.path.join(self.dir_path, f_name))
            img.close()
        consec_images = torch.cat(consec_images, dim=0)  ## [T, H, W]

        consec_masks_name = self.consec_masks_names[idx]  # ["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png"]
        consec_masks = []
        for f_name in consec_masks_name:
            img = Image.open(os.path.join(self.dir_path, f_name)).convert("L")
            img_tensor = self.trans_totensor(img)
            # img_tensor.unsqueeze_(0)
            consec_masks.append(img_tensor)
            img.close()
        consec_masks = torch.cat(consec_masks, dim=0)  ## [T, H, W]

        ## Center, Angle, Length (cal) and Cls
        consec_json_name = self.consec_json_names[idx]  # ["a0001.json", "a0002.json", "a0003.json"]
        consec_cals = []
        consec_endpoints = []
        consec_labels = []
        for f_name in consec_json_name:
            with open(os.path.join(self.dir_path, f_name), "r") as f:
                js = json.load(f)
                # print(js)
            if len(js["shapes"]) >= 1:  ## annotated with upper needle
                cal = [js["shapes"][1]["center"][0], js["shapes"][1]["center"][1], js["shapes"][1]["theta"], js["shapes"][1]["length"]]
                endpoint = [
                    js["shapes"][1]["points"][0][0],
                    js["shapes"][1]["points"][0][1],
                    js["shapes"][1]["points"][1][0],
                    js["shapes"][1]["points"][1][1],
                ]
                # print('bbox', bbox, 'end', endpoint)
                # label = 0  ## TODO: other cls?
            elif len(js["shapes"]) == 1:  ## annotated with needle
                cal = [js["shapes"][0]["center"][0], js["shapes"][0]["center"][1], js["shapes"][0]["theta"], js["shapes"][0]["length"]]
                endpoint = [
                    js["shapes"][0]["points"][0][0],
                    js["shapes"][0]["points"][0][1],
                    js["shapes"][0]["points"][1][0],
                    js["shapes"][0]["points"][1][1],
                ]
                # label = 0
            else:  ## no needle
                cal = [0, 0, 0, 0]
                endpoint = [0, 0, 0, 0]
                # label = -1
            consec_cals.append(torch.as_tensor(cal, dtype=torch.float32))
            # consec_endpoints.append(tv_tensors.BoundingBoxes(endpoint, format="XYXY", canvas_size=[1758,1758]) )
            consec_endpoints.append(torch.as_tensor(endpoint, dtype=torch.float32))
            # consec_labels.append(torch.as_tensor(label, dtype=torch.float32))
            f.close()
        consec_cals = torch.stack(consec_cals, dim=0)  ## [T, 4]
        consec_endpoints = torch.stack(consec_endpoints, dim=0)  ## [T, 4]
        # consec_labels = torch.stack(consec_labels, dim=0).long()  ## [T,]

        # Unsqueeze
        consec_images = consec_images.unsqueeze(1)  # [T, 1, H, W]
        consec_masks = consec_masks.unsqueeze(1)  # [T, 1, H, W]

        # Apply transform
        if self.transform:
            consec_images, consec_masks, consec_endpoints, consec_cals = self.transform(consec_images, consec_masks, consec_endpoints, consec_cals)

        # Assign labels based on the orientation of the needle
        for t in range(consec_endpoints.shape[0]):
            if consec_endpoints[t, :].sum() == 0:
                consec_labels.append(torch.as_tensor(-1, dtype=torch.float32))  ## cls_id = -1: no needle
            else:
                if self.det_num_classes == 1:  ## 1 class for with needle or not
                    consec_labels.append(torch.as_tensor(0, dtype=torch.float32))  ## cls_id = 0: with needle
                else:  ## 2 classes for needles with different directions
                    if torch.sign(consec_endpoints[t][0] - consec_endpoints[t][2]) == torch.sign(consec_endpoints[t][1] - consec_endpoints[t][3]):
                        consec_labels.append(torch.as_tensor(0, dtype=torch.float32))  ## cls_id = 0: left-top to right-bottom needle
                    else:
                        consec_labels.append(torch.as_tensor(1, dtype=torch.float32))  ## cls_id = 1: right-top to left-bottom needle
        consec_labels = torch.stack(consec_labels, dim=0).long()  ## [T,]

        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]
        consec_masks = consec_masks.squeeze(1)  # [T, H, W]

        sample = {
            "images": consec_images,
            "masks": consec_masks,
            "cals": consec_cals,  ## center, angle, length (x2, y2, angle, length)
            "endpoints": consec_endpoints,  ## (x1, y1, x3, y3) tensor, not tv_tensors.BoundingBoxes)
            "labels": consec_labels,  ## cls_id = -1: no needle, 0: left-top to right-bottom, 1: right-top to left-bottom
            "img_path": fname_list,  ## (path_t1, path_t2, path_t3)
        }
        return sample


# Augmentation Class
class Augmentation(nn.Module):
    def __init__(self, resized_crop=True, color_jitter=True, horizontal_flip=True, image_size=224):
        self.resized_crop = resized_crop
        self.color_jitter = color_jitter
        self.horizontal_flip = horizontal_flip
        self.image_size = image_size

    def flip_endpoints(self, endpoints):  ## v2.functional.horizontal_flip gives wrong direction
        if endpoints.dim() == 1:
            new_x1 = self.image_size - endpoints[0]
            new_x3 = self.image_size - endpoints[2]
            new_coords = torch.tensor([new_x1, endpoints[1], new_x3, endpoints[3]])
        else:
            new_x1 = self.image_size - endpoints[:, 0]
            new_x3 = self.image_size - endpoints[:, 2]
            new_coords = torch.stack([new_x1, endpoints[:, 1], new_x3, endpoints[:, 3]], dim=1)
        return new_coords

    def __call__(self, images, masks, endpoints, cals):  ## endpoints type: tv_tensors.BoundingBoxes

        endpoints = tv_tensors.BoundingBoxes(endpoints, format="XYXY", canvas_size=[1758, 1758])  ## [T, 4]

        """ Random Resized Crop """
        if self.resized_crop and random.random() < 0.5:

            # Get parameters for RandomResizedCrop
            ### Note: Due to bias in mask and upper points
            ### large ratio may provide more vertical bias; while small ratio may provide horizontal bias
            top, left, height, width = tf.RandomResizedCrop.get_params(images, scale=(0.7, 1.0), ratio=(0.9, 1.1))  ## ratio:w/h

            # Apply Crop
            images = images[:, :, top : top + height, left : left + width]  ## (... ,Y1:Y2 , X1:X2)
            masks = masks[:, :, top : top + height, left : left + width]

            for r in range(endpoints.shape[0]):
                newpoints = 按斜率滑動到裁剪範圍內(endpoints[r].tolist(), left, top, left + width, top + height)
                newpoints = 轉换到裁剪後座標系(newpoints, left, top)
                endpoints[r][0], endpoints[r][1], endpoints[r][2], endpoints[r][3] = (
                    newpoints[0][0],
                    newpoints[0][1],
                    newpoints[1][0],
                    newpoints[1][1],
                )

            endpoints.canvas_size = (height, width)  ## reset the world size of bbox
            # print(endpoints)

        """Resize"""  ## do not resize at the begining to avoid distortion
        images = v2.functional.resize(images, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR, antialias=True)
        masks = v2.functional.resize(masks, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.NEAREST)
        endpoints = v2.functional.resize(endpoints, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR)

        """ Random Color Jitter"""
        if self.color_jitter and random.random() < 0.5:
            color_transform = tf.Compose(
                [
                    tf.ColorJitter(brightness=0.5, contrast=0.3),
                ]
            )
            images = color_transform(images)

        """ Random Horizontal Flip """
        if self.horizontal_flip and random.random() < 0.5:
            images = tf.functional.hflip(images)
            masks = tf.functional.hflip(masks)
            endpoints = self.flip_endpoints(endpoints)  ## tensor

        ## endpoints (0,0,0,0) may be scaled to (224,0,224,0)
        ## check if endpoints should not exists, reset to (0,0,0,0)
        for r in range(endpoints.shape[0]):
            if endpoints[r][0] == endpoints[r][2] and endpoints[r][1] == endpoints[r][3]:
                endpoints[r][0], endpoints[r][1], endpoints[r][2], endpoints[r][3] = 0.0, 0.0, 0.0, 0.0

        ## update bbox (center x, center y, theta, len)
        cals = get_center_angle_length(endpoints)

        return images, masks, endpoints, cals


# Augmentation Class if no detection head
class AugmentationImgMaskOnly(nn.Module):
    def __init__(self, resized_crop=True, color_jitter=True, horizontal_flip=True, image_size=224):
        self.resized_crop = resized_crop
        self.color_jitter = color_jitter
        self.horizontal_flip = horizontal_flip
        self.image_size = image_size

    def __call__(self, images, masks):

        masks = v2.functional.resize(masks, (images.shape[-2], images.shape[-1]), interpolation=tf.InterpolationMode.NEAREST)
        """ Random Resized Crop """
        if self.resized_crop and random.random() < 0.5:

            # Get parameters for RandomResizedCrop
            ### Note: Due to bias in mask and upper points
            ### large ratio may provide more vertical bias; while small ratio may provide horizontal bias
            top, left, height, width = tf.RandomResizedCrop.get_params(images, scale=(0.7, 1.0), ratio=(0.9, 1.1))  ## ratio:w/h

            # Apply Crop
            images = images[:, :, top : top + height, left : left + width]  ## (... ,Y1:Y2 , X1:X2)
            masks = masks[:, :, top : top + height, left : left + width]

        """Resize"""  ## do not resize at the begining to avoid distortion
        images = v2.functional.resize(images, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR, antialias=True)
        masks = v2.functional.resize(masks, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.NEAREST)

        """ Random Color Jitter"""
        if self.color_jitter and random.random() < 0.5:
            color_transform = tf.Compose(
                [
                    tf.ColorJitter(brightness=0.5, contrast=0.3),
                ]
            )
            images = color_transform(images)

        """ Random Horizontal Flip """
        if self.horizontal_flip and random.random() < 0.5:
            images = tf.functional.hflip(images)
            masks = tf.functional.hflip(masks)

        return images, masks


"""For Semi Supervise Pseudo Labeling"""


class UnlabeledDataset(Dataset):
    def __init__(self, dir_path, transform=None, time_window=3):
        """
        Args:
            dir_path (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir_path = dir_path
        self.time_window = time_window
        self.file_names = sorted(os.listdir(self.dir_path))
        self.image_names = [f for f in self.file_names if f[0] == "a" and f.endswith(".jpg")]  # ["a0001.jpg", "a0002.jpg", ...]

        # creat T consecutive image & mask names list
        if time_window == 1:  ## duplicate
            self.consec_images_names = [
                [self.image_names[i]] * 3 for i in range(0, len(self.image_names) - self.time_window + 1)
            ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg"], ["a0002.jpg", "a0003.jpg", "a0004.jpg"], ...]
        else:
            self.consec_images_names = [
                self.image_names[i : i + self.time_window] for i in range(0, len(self.image_names) - self.time_window + 1)
            ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg"], ["a0002.jpg", "a0003.jpg", "a0004.jpg"], ...]
        self.transform = transform
        self.trans_totensor = tf.Compose([tf.ToTensor()])

    def __len__(self):
        return len(self.consec_images_names)

    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # ["a0001.jpg", "a0002.jpg", "a0003.jpg"]
        consec_images = []
        for f_name in consec_images_name:
            img = Image.open(os.path.join(self.dir_path, f_name)).convert("L")
            img_tensor = self.trans_totensor(img)
            consec_images.append(img_tensor)
            img.close()
        consec_images = torch.cat(consec_images, dim=0)  ## [T, H, W]

        # Unsqueeze
        consec_images = consec_images.unsqueeze(1)  # [T, 1, H, W]

        # Apply transform
        consec_masks = torch.zeros_like(consec_images)
        consec_endpoints, consec_bboxes = torch.zeros([3, 4]), torch.zeros([3, 4])
        if self.transform:
            consec_images, _, _, _ = self.transform(consec_images, consec_masks, consec_endpoints, consec_bboxes)

        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]

        sample = {
            "images": consec_images,
            "img_names": " ".join(consec_images_name),  ## "a0000.jpg a0001.jpg a0002.jpg"
            "img_folder_dir": self.dir_path,  ## folder directory of images
        }
        return sample


class PseudoDataset(Dataset):
    def __init__(self, csv_dir=None, transform=None, time_window=3):
        """
        Args:
            csv_dir (str): The csv that updates image, mask path and confidence,
                            with columns <img_root> <img_names> <mask_path> <confidence>.
            transform (callable, optional): Optional transform to be applied on a sample.

            add <pl> in mask or json name for pseudo label
        """
        self.csv_dir = csv_dir
        self.time_window = time_window

        ## initialize consec names and confidence
        self.update_df_from_csv()

        self.transform = transform
        self.trans_totensor = tf.Compose([tf.ToTensor()])

    def __len__(self):
        return len(self.consec_images_names)

    def update_df_from_csv(self):
        self.df = pd.read_csv(self.csv_dir)
        self.img_roots = self.df["img_root"].tolist()
        self.image_names = self.df["img_names"].tolist()
        self.mask_paths = self.df["mask_path"].tolist()
        # self.json_names = [f.replace(".png", ".json") for f in self.mask_names]   ### Note: assume same name with mask
        self.pl_confidence = self.df["confidence"].tolist()

        # creat T consecutive image & mask & json names list
        self.consec_images_names = [names.split(" ") for names in self.image_names]
        # [["a0001.jpg", "a0002.jpg", "a0003.jpg"], ["a0002.jpg", "a0003.jpg", "a0004.jpg"], ...]
        self.consec_masks_names = [
            [self.mask_paths[i]] * 3 for i in range(0, len(self.mask_paths))
        ]  # [["root/m0001_pl.png", "root/m0002_pl.png", "root/m0003_pl.png"], ["root/m0002_pl.png", "root/m0003_pl.png", "root/m0004_pl.png"], ...]
        # self.consec_json_names = [[self.json_names[i]] * 3 for i in range(0, len(self.json_names))]
        #   ## [["m0001_pl.json", "m0002_pl.json", "m0003_pl.json"], ["m0002_pl.json", "m0003_pl.json", "m0004_pl.json"], ...]

        return

    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # ["a0001.jpg", "a0002.jpg", "a0003.jpg"]
        consec_images = []
        fname_list = []
        for f_name in consec_images_name:
            img = Image.open(os.path.join(self.img_roots[idx], f_name)).convert("L")
            img_tensor = self.trans_totensor(img)
            consec_images.append(img_tensor)
            fname_list.append(os.path.join(self.img_roots[idx], f_name))
            img.close()
        consec_images = torch.cat(consec_images, dim=0)  ## [T, H, W]

        consec_masks_name = self.consec_masks_names[idx]  # ["root/m0001_pl.png", "root/m0001_pl.png", "root/m0001_pl.png"]
        consec_masks = []
        for f_name in consec_masks_name:
            img = Image.open(f_name).convert("L")
            img_tensor = self.trans_totensor(img)
            consec_masks.append(img_tensor)
            img.close()
        consec_masks = torch.cat(consec_masks, dim=0)  ## [T, H, W]

        # ## TODO: pseudo label only train on seg branch?
        # ## Center, Angle, Length (cal) and Cls
        # consec_json_name = self.consec_json_names[idx]  # ["m0001_pl.json", "m0002_pl.json", "m0003_pl.json"]
        # consec_cals = []
        # consec_endpoints = []
        # consec_labels = []
        # for f_name in consec_json_name:
        #     with open(os.path.join(self.pl_dir, f_name), "r") as f:
        #         js = json.load(f)
        #         # print(js)
        #     if "shapes" in js and len(js["shapes"]) >= 0:
        #         cal = [js["shapes"]["center"][0], js["shapes"]["center"][1], js["shapes"]["theta"], js["shapes"]["length"]]
        #         endpoint = [
        #             js["shapes"]["points"][0][0],
        #             js["shapes"]["points"][0][1],
        #             js["shapes"]["points"][1][0],
        #             js["shapes"]["points"][1][1],
        #         ]
        #     else:
        #         cal = [0, 0, 0, 0]
        #         endpoint = [0, 0, 0, 0]
        #         # label = -1
        #     consec_cals.append(torch.as_tensor(cal, dtype=torch.float32))
        #     consec_endpoints.append(torch.as_tensor(endpoint, dtype=torch.float32))
        #     # consec_labels.append(torch.as_tensor(label, dtype=torch.float32))
        #     f.close()
        # consec_cals = torch.stack(consec_cals, dim=0)  ## [T, 4]
        # consec_endpoints = torch.stack(consec_endpoints, dim=0)  ## [T, 4]
        # # consec_labels = torch.stack(consec_labels, dim=0).long()  ## [T,]

        # Unsqueeze
        consec_images = consec_images.unsqueeze(1)  # [T, 1, H, W]
        consec_masks = consec_masks.unsqueeze(1)  # [T, 1, H, W]

        # consec_endpoints = torch.zeros([self.time_window, 4])
        # consec_cals = torch.zeros([self.time_window, 4])

        # Apply transform
        if self.transform:
            consec_images, consec_masks = self.transform(consec_images, consec_masks)

        # # Assign labels based on the orientation of the needle
        # for t in range(consec_endpoints.shape[0]):
        #     if consec_endpoints[t, :].sum() == 0:
        #         consec_labels.append(torch.as_tensor(-1, dtype=torch.float32))  ## cls_id = -1: no needle
        #     elif torch.sign(consec_endpoints[t][0] - consec_endpoints[t][2]) == torch.sign(consec_endpoints[t][1] - consec_endpoints[t][3]):
        #         consec_labels.append(torch.as_tensor(0, dtype=torch.float32))  ## cls_id = 0: left-top to right-bottom
        #     else:
        #         consec_labels.append(torch.as_tensor(1, dtype=torch.float32))  ## cls_id = 1: right-top to left-bottom
        # consec_labels = torch.stack(consec_labels, dim=0).long()  ## [T,]

        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]
        consec_masks = consec_masks.squeeze(1)  # [T, H, W]

        sample = {
            "images": consec_images,
            "masks": consec_masks,
            "cals": torch.tensor([-1, -1, -1, -1]),  ## center, angle, length (x2, y2, angle, length)
            "endpoints": torch.tensor([-1, -1, -1, -1]),  ## (x1, y1, x3, y3) tensor, not tv_tensors.BoundingBoxes)
            "labels": -2,  ## cls_id = -1: no needle, 0: left-top to right-bottom, 1: right-top to left-bottom
            "img_path": fname_list,  ## (path_t1, path_t2, path_t3)
        }
        return sample


def 按斜率滑動到裁剪範圍內(points, X1, Y1, X2, Y2):  # points「按斜率滑動」到crop範圍內
    x1, y1 = points[0], points[1]
    x2, y2 = points[2], points[3]
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float("inf")

    if x1 < X1:
        y1 += slope * (X1 - x1)
        x1 = X1
    if x1 > X2:
        y1 += slope * (X2 - x1)
        x1 = X2
    if y1 < Y1:
        x1 += (Y1 - y1) / slope if slope != float("inf") else 0
        y1 = Y1
    if y1 > Y2:
        x1 += (Y2 - y1) / slope if slope != float("inf") else 0
        y1 = Y2
    if x2 < X1:
        y2 -= slope * (x2 - X1)
        x2 = X1
    if x2 > X2:
        y2 -= slope * (x2 - X2)
        x2 = X2
    if y2 < Y1:
        x2 -= (y2 - Y1) / slope if slope != float("inf") else 0
        y2 = Y1
    if y2 > Y2:
        x2 -= (y2 - Y2) / slope if slope != float("inf") else 0
        y2 = Y2

    return [[max(min(x1, X2), X1), max(min(y1, Y2), Y1)], [max(min(x2, X2), X1), max(min(y2, Y2), Y1)]]


def 轉换到裁剪後座標系(points, X1, Y1):
    # 用新的原點描述滑動好的points 在前面輸入裁剪範圍時就排序大小X1<X2, Y1<Y2, (X1,Y1)top-left corner of the crop area becomes the new origin (0, 0)
    return [[x - X1, y - Y1] for x, y in points]


def 計算中心點和角度和長度(points):
    x1, y1 = points[0], points[1]
    x2, y2 = points[2], points[3]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    if x2 == x1:  # 確保不會除以零
        theta = math.pi / 2 if y2 > y1 else -math.pi / 2
    else:
        theta = math.atan((y2 - y1) / (x2 - x1))
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return torch.tensor([x_center, y_center, theta, length])


def get_center_angle_length(points):
    cals = torch.zeros_like(points, dtype=torch.float32)  ## set dtype to float to avoid error
    x1, y1 = points[:, 0], points[:, 1]
    x2, y2 = points[:, 2], points[:, 3]
    cals[:, 0] = (x1 + x2) / 2  ## center x
    cals[:, 1] = (y1 + y2) / 2  ## center y
    # bboxs[:,2] = torch.where(x1 == x2 , torch.sign(y2 - y1) * math.pi / 2, torch.atan2(y2 - y1 , x2 - x1))
    cals[:, 2] = torch.where(x1 == x2, torch.sign(y2 - y1) * math.pi / 2, torch.atan((y2 - y1) / (x2 - x1)))
    cals[:, 3] = torch.sqrt(torch.pow(x2 - x1, 2) + torch.pow(y2 - y1, 2))  ## length
    return cals
