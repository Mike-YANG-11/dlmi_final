# Retina-UNETR for Ultrasound Video Needle Localization

## Code Directory Layout
```
dlmi_final/
    ├── README.md
    ├── dataset.py
    ├── model.py
    ├── loss.py
    ├── evaluation.py
    ├── visualization.py
    ├── train.py
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
- `dataset.py`
  - Contains the custom dataset and augmentation classes.
  - Current implementation is straightforward but not optimized for speed.
  - It reads the images and apply the transformation every time it is called, which may can be optimized by loading some images in a buffer memory.
  - The masks are redundant in the current implementation, as only the mask from the last frame are used in the training process.
- `model.py`
  - Video-UNETR model architecture.
  - Video-Retina-UNETR model architecture.
  - Currently, the model only supports 3x224x224 input size.
- `loss.py`
  - Loss functions for the segmentation and line detection tasks.
  - Only dice & focal loss are used right now.
- `evaluation.py`
  - Evaluation functions based on the segmentation results.
- `visualization.py`
  - Visualization function for showing the training data before training & prediction results during training.
  - The visualization function can be disabled by setting `visualize = False` in the `train.py` script.
- `train.py`
  - The main training script. Run `python train.py` to train the model.
  - The directory paths should be modified in the `construct_datasets()` function.
  - Hyperparameters can be modified in the `main()` function.
  - Pseudo label training is only activated if the mask threshold is set to non `null` value in `config.json`.
- `test.py`
  - Pass `run id` in wandb and checkpoint path when running, so that the model follows the setting during training.
  - Inference medium and hard test folders in `config.json`
  - Record results into `run.summary` in wandb.
- `pseudo_label.py`
  - Evaluate the predicted masks of unlabeled dataset.
  - If the confidence of the mask is high enough, then `mask_XXXX_pl.png` is saved and recorded to `pl.csv` in the `pseudo_label` folder.
- `logs/`
  - Contains the logs for the training process.
- `video_unetr_checkpoints/`
  - Directory to save the trained model checkpoints.
- `mae_pretrain_vit_base_checkpoints/`
  - Directory to the MAE ImageNet-1K pre-trained model checkpoints.
  - The checkpoint file is not included in the repository and should be downloaded from [here](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth).
  - Place the downloaded checkpoint file in this directory.

