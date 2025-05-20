# Training script for the ViT-B/16 model using the dinov2 module, with reference to:
# https://github.com/facebookresearch/dinov2?tab=readme-ov-file#fast-setup-training-dinov2-vit-l16-on-imagenet-1k

# The default number of GPUs: 4, so the total batch size is 4*128=512.
# Change the number of GPUs and batch size accordingly if the number of GPUs is different.
# Remind to check the config file for the arguments (`OFFICIAL_EPOCH_LENGTH`, `batch_size_per_gpu` & `eval_period_iterations`) used in the training script.

# If you want to add extra datasets, you can place them in the train folder following the same folder structure.
# Then, you need to modify the training config in the `./dinov2/configs/custom/vitb16.yml` file.

DATASET_ROOT=/path/to/dataset

torchrun --nproc_per_node=1 train_dinov2.py \
    --config-file ./dinov2/configs/custom/vitb16.yml \
    --output-dir ./artifacts-dinov2/ViT-B-16 \
    train.dataset_path=DINOv2Dataset:root=$DATASET_ROOT
