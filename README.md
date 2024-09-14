# Retina-UNETR for Ultrasound Video Needle Localization
- Report: <https://drive.google.com/file/d/10ABVJSY6566dyJyar201z6DHuYUny7MM/view?usp=drive_link>
## Video Demo
***
<video src="https://github.com/user-attachments/assets/7656418a-f9ca-4373-9d00-d50d656d399b"></video>
***
## Architecture
![model](https://github.com/user-attachments/assets/7895920a-4948-4417-9ee5-80a40a6cff72)
***
## Code Directory Layout
```
dlmi_final/
    ├── README.md
    ├── environment.yml
    ├── config.json
    ├── dataset.py
    ├── model.py
    ├── loss.py
    ├── evaluation.py
    ├── visualization.py
    ├── train.py
    ├── test.py
    ├── pseudo_label.py
    ├── post_processing.py
    ├── anchors.py
    ├── logs/
    │   └── ...
    ├── pseudo_label/
    │   └── ...
    ├── video_unetr_checkpoints/
    │   └── ...
    └── mae_pretrain_vit_base_checkpoints/
        └── mae_pretrain_vit_base.pth
```
## Set up
```
conda env create -f environment.yml
conda activate dlmi_final
```
## Implementation
```
# trainig
python train.py
# testing
python test.py "<run id in wandb>" "<checkpoint path>"
```
## File Functionalities
- `config.json`
  - Training hyperparameters and model architecture designs can be modified in this file.
- `dataset.py`
  - Contains the custom dataset and augmentation classes.
  - Image reading can be speeded up with larger number of samples in a buffer. 
  - The masks are redundant in the current implementation, as only the mask from the last frame are used in the training process.
- `model.py`
  - Video-UNETR model architecture.
  - Video-Retina-UNETR model architecture.
  - The model supports Tx224x224 input size, where T is an arbitrary number multiple of 3.
- `loss.py`
  - Loss functions for segmentation and detection tasks.
- `evaluation.py`
  - Evaluation functions based on the segmentation and detection results.
- `visualization.py`
  - Visualization function for showing the training data before training & prediction results during training.
  - The visualization function can be disabled by setting `visualize = False` in `config.json`.
- `train.py`
  - The main training script. Run `python train.py` to train the model.
  - Hyperparameters can be modified in `config.json`.
  - Pseudo label training is only activated if the mask threshold is set to non `null` value in `config.json`.
- `test.py`
  - Pass `run id` in wandb and checkpoint path when running, so that the model follows the setting during training.
  - Inference medium and hard test folders in `config.json`
  - Record results into `run.summary` in wandb.
- `pseudo_label.py`
  - Evaluate the predicted masks of unlabeled dataset.
  - If the confidence of the mask is high enough, then `mask_XXXX_pl.png` is saved and recorded to `pl.csv` in the `pseudo_label` folder.
- `post_processing.py`
  - Functions for detection head output post-processing.
- `anchors.py`
  - Functions for anchor generation in detection head.
- `logs/`
  - Contains the logs for the training process.
- `video_unetr_checkpoints/`
  - Directory to save the trained model checkpoints.
- `video_retina-unetr_checkpoints/`
  - Directory to save the trained model checkpoints.
- `mae_pretrain_vit_base_checkpoints/`
  - Directory to the MAE ImageNet-1K pre-trained model checkpoints.
  - The checkpoint file is not included in the repository and should be downloaded from [here](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth).
  - Place the downloaded checkpoint file in this directory.

