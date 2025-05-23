# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import torch
from torch.utils.data import Sampler

from .datasets import ImageNet, ImageNet22k, ImageShipID, ImageShipID_Extra, ImageShipID_20P, ImageShipID_40P, ImageShipID_60P, ImageShipID_80P, ImageShipOOD, ImageShipID_100I, ImageShipID_500I, ImageShipID_1000I, ImageShipID_5000I, ImageShipID_10000I, ImageShipID_1M, ImageShipID_200k, ImageShipID_25k, DINOv2Dataset
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler, ShardedInfiniteBalancedSampler


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4
    SHARDED_INFINITE_BALANCED = 5


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(image_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
    elif name == "ImageShipID":
        class_ = ImageShipID
        if "split" in kwargs:
            kwargs["split"] = ImageShipID.Split[kwargs["split"]]
    elif name == "ImageShipID_Extra":
        class_ = ImageShipID_Extra
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_Extra.Split[kwargs["split"]]
    elif name == "ImageShipID_20P":
        class_ = ImageShipID_20P
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_20P.Split[kwargs["split"]]
    elif name == "ImageShipID_40P":
        class_ = ImageShipID_40P
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_40P.Split[kwargs["split"]]
    elif name == "ImageShipID_60P":
        class_ = ImageShipID_60P
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_60P.Split[kwargs["split"]]
    elif name == "ImageShipID_80P":
        class_ = ImageShipID_80P
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_80P.Split[kwargs["split"]]
    elif name == "ImageShipID_100I":
        class_ = ImageShipID_100I
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_100I.Split[kwargs["split"]]
    elif name == "ImageShipID_500I":
        class_ = ImageShipID_500I
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_500I.Split[kwargs["split"]]
    elif name == "ImageShipID_1000I":
        class_ = ImageShipID_1000I
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_1000I.Split[kwargs["split"]]
    elif name == "ImageShipID_5000I":
        class_ = ImageShipID_5000I
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_5000I.Split[kwargs["split"]]
    elif name == "ImageShipID_10000I":
        class_ = ImageShipID_10000I
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_10000I.Split[kwargs["split"]]
    elif name == "ImageShipID_1M":
        class_ = ImageShipID_1M
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_1M.Split[kwargs["split"]]
    elif name == "ImageShipID_200k":
        class_ = ImageShipID_200k
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_200k.Split[kwargs["split"]]
    elif name == "ImageShipID_25k":
        class_ = ImageShipID_25k
        if "split" in kwargs:
            kwargs["split"] = ImageShipID_25k.Split[kwargs["split"]]
    elif name == "ImageShipOOD":
        class_ = ImageShipOOD
        if "split" in kwargs:
            kwargs["split"] = ImageShipOOD.Split[kwargs["split"]]
    elif name == "DINOv2Dataset":
        class_ = DINOv2Dataset
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    dataset = class_(transform=transform, target_transform=target_transform, **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
    **kwargs,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
    elif type == SamplerType.SHARDED_INFINITE_BALANCED:
        logger.info("sampler: sharded infinite balanced")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return ShardedInfiniteBalancedSampler(
            labels=dataset.get_targets(),
            mode=kwargs["balanced_sampler_mode"],
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    **kwargs,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
        **kwargs,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
