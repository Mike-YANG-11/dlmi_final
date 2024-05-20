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
        self.json_names = [
            f.replace(".jpg", ".json") for f in self.image_names
        ]

        # creat T consecutive image & mask names list
        self.consec_images_names = [
            self.image_names[i : i + self.time_window] for i in range(0, len(self.image_names) - self.time_window + 1)
        ]  # [["a0001.jpg", "a0002.jpg", "a0003.jpg"], ["a0002.jpg", "a0003.jpg", "a0004.jpg"], ...]
        self.consec_masks_names = [
            self.mask_names[i : i + self.time_window] for i in range(0, len(self.mask_names) - self.time_window + 1)
        ]  # [["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png"], ["m0002_lw_20.png", "m0003_lw_20.png", "m0004_lw_20.png"], ...]
        self.consec_json_names = [
            self.json_names[i : i + self.time_window] for i in range(0, len(self.json_names) - self.time_window + 1)
        ]
        self.transform = transform
        self.trans_totensor = tf.Compose([ tf.ToTensor() ])

    def __len__(self):
        return len(self.consec_images_names)

    def __getitem__(self, idx):
        consec_images_name = self.consec_images_names[idx]  # ["a0001.jpg", "a0002.jpg", "a0003.jpg"]
        consec_images = []
        fname_list = []
        for f_name in consec_images_name:
            img = Image.open(os.path.join(self.dir_path, f_name)).convert('L')
            img_tensor = self.trans_totensor(img)
            consec_images.append(img_tensor)
            fname_list.append(os.path.join(self.dir_path, f_name))
            img.close()
        consec_images = torch.cat(consec_images, dim = 0) ## [T, H, W]

        consec_masks_name = self.consec_masks_names[idx]  # ["m0001_lw_20.png", "m0002_lw_20.png", "m0003_lw_20.png"]
        consec_masks = []
        for f_name in consec_masks_name:
            img = Image.open(os.path.join(self.dir_path, f_name)).convert('L')
            img_tensor = self.trans_totensor(img)
            # img_tensor.unsqueeze_(0)
            consec_masks.append(img_tensor)
            img.close()
        consec_masks = torch.cat(consec_masks, dim = 0) ## [T, H, W]
        
        ## BBox and Cls
        consec_json_name = self.consec_json_names[idx]  # ["a0001.json", "a0002.json", "a0003.json"]
        consec_bboxes = []
        consec_endpoints = []
        consec_labels = []
        for f_name in consec_json_name:
            with open(os.path.join(self.dir_path, f_name), 'r') as f:
                js = json.load(f)
                # print(js)
            if len(js["shapes"]) >= 1:  ## annotated with upper needle
                bbox = [js["shapes"][1]["center"][0], 
                        js["shapes"][1]["center"][1],
                        js["shapes"][1]["theta"],
                        js["shapes"][1]["length"]]
                endpoint = [js["shapes"][1]["points"][0][0], 
                        js["shapes"][1]["points"][0][1],
                        js["shapes"][1]["points"][1][0],
                        js["shapes"][1]["points"][1][1]]
                print('bbox', bbox, 'end', endpoint)
                label = 1  ## TODO: other cls?
            elif len(js["shapes"]) == 1:  ## annotated with needle
                bbox = [js["shapes"][0]["center"][0], 
                        js["shapes"][0]["center"][1],
                        js["shapes"][0]["theta"],
                        js["shapes"][0]["length"]]
                endpoint = [js["shapes"][0]["points"][0][0], 
                        js["shapes"][0]["points"][0][1],
                        js["shapes"][0]["points"][1][0],
                        js["shapes"][0]["points"][1][1]]
                label = 1 
            else:
                bbox = [0,0,0,0]
                endpoint = [0,0,0,0]
                label = 0
            consec_bboxes.append(torch.as_tensor(bbox))
            # consec_endpoints.append(tv_tensors.BoundingBoxes(endpoint, format="XYXY", canvas_size=[1758,1758]) )
            consec_endpoints.append(torch.as_tensor(endpoint))
            consec_labels.append(torch.as_tensor(label))
            f.close()
        consec_bboxes = torch.stack(consec_bboxes, dim = 0) ## [T, 4]
        consec_endpoints = torch.stack(consec_endpoints, dim = 0) ## [T, 4]
        consec_labels = torch.stack(consec_labels, dim = 0).long() ## [T,]

        # Unsqueeze
        consec_images = consec_images.unsqueeze(1)  # [T, 1, H, W]
        consec_masks = consec_masks.unsqueeze(1)  # [T, 1, H, W]

        # Apply transform 
        if self.transform:
            consec_images, consec_masks, consec_endpoints, consec_bboxes = self.transform(consec_images, consec_masks, consec_endpoints, consec_bboxes)
        
        # Squeeze
        consec_images = consec_images.squeeze(1)  # [T, H, W]
        consec_masks = consec_masks.squeeze(1)  # [T, H, W]

        sample = {
            "images": consec_images,
            "masks" : consec_masks,
            "bboxes": consec_bboxes,       ## (x2, y2, angle, length)
            "endpoints": consec_endpoints, ## (x1, y1, x3, y3) tensor, not tv_tensors.BoundingBoxes)
            "labels": consec_labels,
            "img_path": fname_list         ## (path_t1, path_t2, path_t3)
        }
        return sample


# Augmentation Class
class Augmentation(nn.Module):
    def __init__(self, color_jitter=True, resized_crop=True, horizontal_flip=True, image_size=224):
        self.color_jitter = color_jitter
        self.resized_crop = resized_crop
        self.horizontal_flip = horizontal_flip
        self.image_size = image_size

    def flip_endpoints(self, endpoints):  ## v2.functional.horizontal_flip gives wrong direction
        if endpoints.dim() == 1:
            new_x1 = self.image_size - endpoints[0]
            new_x3 = self.image_size  - endpoints[2]
            new_coords = torch.tensor([new_x1, endpoints[1], new_x3, endpoints[3]])            
        else:
            new_x1 = self.image_size  - endpoints[:, 0]
            new_x3 = self.image_size  - endpoints[:, 2]
            new_coords = torch.stack([new_x1, endpoints[:, 1], new_x3, endpoints[:,3]], dim=1)
        return new_coords

    def __call__(self, images, masks, endpoints, bboxes):  ## endpoints type: tv_tensors.BoundingBoxes
        
        endpoints = tv_tensors.BoundingBoxes(endpoints, format="XYXY", canvas_size=[1758,1758]) ## [T, 4]
        
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
            ### Note: Due to bias in mask and upper points
            ### large ratio may provide more vertical bias; while small ratio may provide horizontal bias
            top, left, height, width = tf.RandomResizedCrop.get_params(images, scale=(0.7, 1.0), ratio=(0.9, 1.1))  ## ratio:w/h
            
            # Apply Crop
            images = images[:, :, top: top + height , left:left + width] ## (... ,Y1:Y2 , X1:X2)
            masks  = masks[:, :, top: top + height , left:left + width]

            for r in range(endpoints.shape[0]): 
                newpoints = 按斜率滑動到裁剪範圍內(endpoints[r].tolist(), left, top, left + width, top + height)
                newpoints = 轉换到裁剪後座標系(newpoints, left, top)
                endpoints[r][0], endpoints[r][1], endpoints[r][2], endpoints[r][3] = newpoints[0][0], newpoints[0][1],  newpoints[1][0], newpoints[1][1]
                
            endpoints.canvas_size = (height, width)  ## reset the world size of bbox
            # print(endpoints)

        """Resize"""  ## do not resize at the begining to avoid distortion
        images = v2.functional.resize(
                images, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR, antialias=True
            )
        masks = v2.functional.resize(
                masks, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.NEAREST
            )
        endpoints = v2.functional.resize(
            endpoints, (self.image_size, self.image_size), interpolation=tf.InterpolationMode.BILINEAR
        )
        
        """ Random Horizontal Flip """
        if self.horizontal_flip and random.random() < 0.5:
            images = tf.functional.hflip(images)
            masks = tf.functional.hflip(masks)
            endpoints = self.flip_endpoints(endpoints) ## tensor

        ## endpoints (0,0,0,0) may be scaled to (224,0,224,0)
        ## check if endpoints should not exists, reset to (0,0,0,0)
        for r in range(endpoints.shape[0]):
            if endpoints[r][0] == endpoints[r][2] and endpoints[r][1] == endpoints[r][3]:
                endpoints[r][0], endpoints[r][1], endpoints[r][2], endpoints[r][3] = 0.,0.,0.,0.
        
        ## update bbox (center x, center y, theta, len)
        bboxes = get_center_angle_length(endpoints)
        
        return images, masks, endpoints, bboxes


def 按斜率滑動到裁剪範圍內(points, X1, Y1, X2, Y2): #points「按斜率滑動」到crop範圍內
    x1, y1 = points[0], points[1]
    x2, y2 = points[2], points[3]
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

    if x1 < X1: y1 += slope * (X1 - x1); x1 = X1
    if x1 > X2: y1 += slope * (X2 - x1); x1 = X2
    if y1 < Y1: x1 += (Y1 - y1) / slope if slope != float('inf') else 0; y1 = Y1
    if y1 > Y2: x1 += (Y2 - y1) / slope if slope != float('inf') else 0; y1 = Y2
    if x2 < X1: y2 -= slope * (x2 - X1); x2 = X1
    if x2 > X2: y2 -= slope * (x2 - X2); x2 = X2
    if y2 < Y1: x2 -= (y2 - Y1) / slope if slope != float('inf') else 0; y2 = Y1
    if y2 > Y2: x2 -= (y2 - Y2) / slope if slope != float('inf') else 0; y2 = Y2

    return [[max(min(x1, X2), X1), max(min(y1, Y2), Y1)], [max(min(x2, X2), X1), max(min(y2, Y2), Y1)]]

def 轉换到裁剪後座標系(points, X1, Y1): #用新的原點描述滑動好的points     在前面輸入裁剪範圍時就排序大小X1<X2, Y1<Y2, (X1,Y1)top-left corner of the crop area becomes the new origin (0, 0)
    return [[x - X1, y - Y1] for x, y in points]

def 計算中心點和角度和長度(points):
    x1, y1 = points[0], points[1]
    x2, y2 = points[2], points[3]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    if x2 == x1:       # 確保不會除以零
        theta = math.pi / 2 if y2 > y1 else -math.pi / 2
        theta_y= -math.pi / 2 if y2 > y1 else math.pi / 2 # 反轉 Y 軸，符合傳統座標系Y軸朝上
    else:
        theta = math.atan((y2 - y1) / (x2 - x1))
        theta_y = math.atan((y1 - y2) / (x2 - x1))  # 反轉 Y 軸，符合傳統座標系Y軸朝上
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return torch.tensor([x_center, y_center, theta, length])

def get_center_angle_length(points):
    bboxs = torch.zeros_like(points)
    x1, y1 = points[:,0], points[:,1]
    x2, y2 = points[:,2], points[:,3]
    bboxs[:,0] = (x1 + x2) / 2  ## center x
    bboxs[:,1] = (y1 + y2) / 2  ## center y
    # bboxs[:,2] = torch.where(x1 == x2 , torch.sign(y2 - y1) * math.pi / 2, torch.atan2(y2 - y1 , x2 - x1))
    bboxs[:,2] = torch.where(x1 == x2 , torch.sign(y2 - y1) * math.pi / 2, torch.atan((y2 - y1 )/(x2 - x1)))
    bboxs[:,3] = torch.sqrt(torch.pow(x2 - x1, 2) + torch.pow(y2 - y1, 2))  ## length
    return bboxs
