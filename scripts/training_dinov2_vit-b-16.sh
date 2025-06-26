# Training script for the ViT-B/16 model using the dinov2 module, with reference to:
# https://github.com/facebookresearch/dinov2?tab=readme-ov-file#fast-setup-training-dinov2-vit-l16-on-imagenet-1k

CONFIG_FILE=./dinov2/configs/custom/vitb16.yml
DATASET_ROOT=/path/to/dataset
OUTPUT_DIR=./artifacts-dinov2/ViT-B-16
NUM_GPUS=1
BATCH_SIZE_PER_GPU=128


torchrun --nproc_per_node=$NUM_GPUS train_dinov2.py \
    --config-file $CONFIG_FILE \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE_PER_GPU \
    train.dataset_path=DINOv2Dataset:root=$DATASET_ROOT
