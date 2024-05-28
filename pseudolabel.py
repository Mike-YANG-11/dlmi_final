import numpy as np
import torch
from torchvision.utils import save_image

from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import ntpath

## https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format/8384788#8384788
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

## Evaluate the prediction on unlabeled data to get pseudo labels
def generate_pl(config, model, device, loader, model_name, df, pl_dir):
    print("Generating Pseudo Labels...")
    model.eval()
    # model.to(device)

    ## if mask is confident enough, then add to pseudo dataset
    pl_mask_thres = config["Semi_Supervise"]["Pseudo_label"]["mask_thres"]
    pl_version = config["Semi_Supervise"]["Pseudo_label"]["pl_version"]
    pl_count = 0
    
    if not os.path.exists(pl_dir):
        os.makedirs(pl_dir)

    with torch.no_grad():
        for step, samples in enumerate(tqdm(loader)):
            # Image data
            images = samples["images"].to(device)  # [N, T, H, W]
            img_names = samples["img_names"]       # [N] list   ## ['a0275.jpg a0278.jpg a0280.jpg',  ...]
            img_folder_dir = samples["img_folder_dir"]  # [N] list

            # Forward pass
            if model_name == "Video-Retina-UNETR":
                pred_masks, classifications, regressions = model(images)
                # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4]
            else:
                pred_masks = model(images)  # [N, 1, H, W]
            
            ## TODO confidence for detection head?? (future work)
            preds_qualification, preds_confidence, binary_masks = get_confidence(pred_masks.squeeze_(), pl_mask_thres, pl_version)
            for i in range(pred_masks.shape[0]):
                if preds_qualification[i] == 1:
                    ## get mask pl name
                    last_img_folder = path_leaf(img_folder_dir[i])  ##  e.g. 20240402_1542_4074x3154_abc
                    pl_subfolder = os.path.join(pl_dir,last_img_folder).replace("\\","/")   ## ./pseudo_label/model_1/20240402_1542_4074x3154_abc
                    if not os.path.exists(pl_subfolder):
                        os.makedirs(pl_subfolder)
                    last_img_name = img_names[i].split(" ")[-1]
                    mask_name = "m" + last_img_name[1:].replace('.jpg', '_pl.png')  ## mXXXX_pl.png
                    mask_path = os.path.join(pl_subfolder, mask_name).replace("\\","/")   ## ./pseudo_label/model_1/20240402_1542_4074x3154_abc/mXXXX_pl.png
                    
                    ## save mask
                    save_image(tensor=binary_masks[0], fp=mask_path)  ### TODO: compare with past pred confidence
                    
                    ## add row to df
                    df.loc[len(df.index)] = [img_folder_dir[i].replace("\\","/"), img_names[i], mask_path, preds_confidence[i].item()]
                    pl_count += 1
    
    
    print(f"\t! Generate {pl_count} pseudo labels")
    return df


def get_confidence(pred_masks, pl_mask_thres, version=1):
    """input pred_masks [N, H, W]
       output [N,] (qualification for each sample)
              [N,] (confidence for each sample)
              [N, H, W] (pl masks)""" 
    smooth = 1e-6
    ## area of 0 or 1 in each binary mask
    zero_area = (pred_masks <= 0.5).sum(dim=(1,2))
    one_area  = (pred_masks > 0.5).sum(dim=(1,2))

    ## area of 0 or 1 with high confidence in each binary mask
    confident_zero_area = (pred_masks <= 0.2).sum(dim=(1,2)) ## TODO check if too serious
    confident_one_area  = (pred_masks >= 0.8).sum(dim=(1,2))

    ## area portion of 0 or 1 with high confidence in each binary mask
    confident_zero_portion = torch.divide(confident_zero_area, torch.add(zero_area, smooth))
    confident_one_portion = torch.divide(confident_one_area, torch.add(one_area, smooth))
    confidence_score = torch.multiply(confident_zero_portion, confident_one_portion)

    qualification = torch.where((confident_zero_portion >= pl_mask_thres) & (confident_one_portion >= pl_mask_thres), 1,0)
    if version == 1:
        pl_masks = torch.where(pred_masks > 0.5, 1., 0.).detach().cpu()
    elif version == 2:
        pl_masks = torch.where(pred_masks >= 0.8, 1., pred_masks).detach().cpu()
        pl_masks = torch.where(pl_masks <= 0.2, 0., pl_masks)
        pl_masks = torch.where((pl_masks <= 0.2) | (pl_masks >= 0.8), pl_masks, 2)
    return qualification, confidence_score.detach().cpu(), pl_masks