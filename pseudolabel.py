import numpy as np
import torch
from torchvision.utils import save_image

from sklearn.decomposition import PCA
from tqdm import tqdm
import os

## Evaluate the prediction on unlabeled data to get pseudo labels
def generate_pl(model, device, loader, model_name, df, pl_mask_thres, pl_dir):
    print("Generating Pseudo Labels...")
    model.eval()
    # model.to(device)

    if not os.path.exists(pl_dir):
        os.makedirs(pl_dir)

    with torch.no_grad():
        for step, samples in enumerate(tqdm(loader)):
            # Image data
            images = samples["images"].to(device)  # [N, T, H, W]
            img_paths = samples["img_path"]  # (N, 3)

            # Forward pass
            if model_name == "Video-Retina-UNETR":
                pred_masks, classifications, regressions = model(images)
                # [N, 1, H, W], [N, num_total_anchors, num_classes], [N, num_total_anchors, 4]
            else:
                pred_masks = model(images)  # [N, 1, H, W]
            
            ## TODO confidence for detection head??
            preds_qualification, preds_confidence, binary_masks = get_confidence(pred_masks.squeeze_(), pl_mask_thres)
            for i in range(pred_masks.shape[0]):
                if preds_qualification[i] == 1:
                    ## get mask pl name
                    last_img_path = img_paths[i][-1]
                    last_img_folder = os.path.basename(os.path.dirname(last_img_path))  ##  e.g. 20240402_1542_4074x3154_abc
                    pl_subfolder = os.path.join(pl_dir,last_img_folder)         ## ./pseudo_label/model_1/20240402_1542_4074x3154_abc
                    if not os.path.exists(pl_subfolder):
                        os.makedirs(pl_subfolder)
                    last_img_name = os.path.basename(last_img_path)
                    mask_name = "m" + last_img_name[1:].replace('.jpg', '_pl.png')   ## ./pseudo_label/model_1/20240402_1542_4074x3154_abc/mXXXX_pl.png
                    mask_path = os.path.join(pl_subfolder, mask_name)
                    
                    ## save mask
                    save_image(tensor=binary_masks[0], fp=mask_path)  ### TODO: compare with past pred confidence
                    
                    ## add row to df
                    img_paths_str = " ".join(img_paths[i])
                    df.loc[len(df.index)] = [img_paths_str, mask_path, preds_confidence[i]]
    df.to_csv(os.path.join(pl_dir,'pl.csv'))  ## ./pseudo_label/model_1/pl.csv
    return df


def get_confidence(pred_masks, pl_mask_thres):
    """input [N, H, W]
       output [N,] (qualification for each sample)
              [N, H, W] (binary masks)""" 
    smooth = 1e-6
    binary_masks = torch.where(pred_masks > 0.5, 1., 0.)
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
    return qualification, confidence_score, binary_masks.detach().cpu()