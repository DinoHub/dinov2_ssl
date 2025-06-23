# Linear probing script for the ViT-B/16 model with self-supervised pretrained weight using the DINOv2 module, with reference to:
# https://github.com/facebookresearch/dinov2?tab=readme-ov-file#linear-classification-with-data-augmentation-on-imagenet-1k

# The default number of GPUs: 4, so the total batch size is 512.
# Change the number of GPUs and batch size accordingly if the number of GPUs is different.
# Remind to check the `eval_linear.py` script for the arguments (`ncls`, `ngpus`, `batch_size`, `epoch_length` & `epochs`) used in the script.

# According to the discussion in the DINOv2 repo, the teacher checkpoint is used for linear probing.
# The path to the teacher checkpoint is something like `./artifacts-dinov2/ViT-B-16/eval/training_<iteration>/teacher_checkpoint.pth`.

TRAIN_DATASET_ROOT=/path/to/dataset/train
VAL_DATASET_ROOT=/path/to/dataset/val
PRETRAINED_WEIGHTS=/path/to/teacher_checkpoint.pth
EPOCHS=10
NUM_CLASSES=6
NUM_GPUS=1
TOTAL_BATCH_SIZE=32

batch_size=$(( TOTAL_BATCH_SIZE / NUM_GPUS ))

torchrun --nproc_per_node=$NUM_GPUS eval_linear.py \
--config-file ./dinov2/configs/custom/vitb16.yml \
--pretrained-weights $PRETRAINED_WEIGHTS \
--output-dir ./artifacts-linear-classifier/ViT-B-16-v3 \
--train-dataset DINOv2Dataset:root=$TRAIN_DATASET_ROOT \
--val-dataset DINOv2Dataset:root=$VAL_DATASET_ROOT \
--num-workers 0 \
--batch-size $batch_size \
--epochs $EPOCHS \
--save-checkpoint-frequency 1 \
--num-classes $NUM_CLASSES \
--num-gpus $NUM_GPUS \
--total-batch-size $TOTAL_BATCH_SIZE

# --balanced-sampler \
# --balanced-sampler-mode 200000 \
# --logit-adjusted-loss \
