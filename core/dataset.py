"""
Dataset Module for SemCom_Infra.

This module provides image dataset loading utilities with dual-mode support:
local filesystem and Hugging Face Hub online download. It is designed for
semantic communication experiments but retains extensibility for future
multi-modal tasks.

Abstract Base Classes:
    - SemComDataset: Abstract dataset interface for all SemCom data sources.

Concrete Implementations:
    - ImageDataset: Unified image dataset supporting local files and HF Hub.

Utility Functions:
    - get_standard_transforms: Build standard train/eval transform pipelines.
    - get_dataloader: High-level one-liner to get a ready-to-use DataLoader.

Usage Examples:
    Local folder::

        ds = ImageDataset(source="./data/cifar10", source_type="local",
                          transform=get_standard_transforms(crop_size=32))
        print(len(ds), ds[0].shape)

    Hugging Face Hub::

        ds = ImageDataset(source="cifar10", source_type="hf", split="train",
                          transform=get_standard_transforms(crop_size=32))
        print(len(ds), ds[0].shape)

    Quick DataLoader::

        transform = get_standard_transforms(crop_size=32, is_train=True)
        dataset = ImageDataset("./data/cifar10", source_type="local",
                               transform=transform)
        loader = get_dataloader(dataset, batch_size=64, shuffle=True)
        for batch in loader:
            print(batch.shape)  # (64, 3, 32, 32)
            break
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
# Hugging Face `datasets` is optional — deferred import with friendly message
try:
    import datasets as hf_datasets

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# Supported image extensions for local scanning
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ============================================================================
# Abstract Base Class
# ============================================================================


class SemComDataset(Dataset, ABC):
    """
    Abstract base class for all Semantic Communication datasets.

    Any dataset used within the SemCom_Infra framework must inherit from
    this class and implement ``__getitem__`` and ``__len__``.

    This interface ensures consistent data loading behaviour across
    different modalities (image, text, point-cloud, etc.) and data
    sources (local, HuggingFace, streaming).

    Subclass Implementation Guide:
        1. Override ``__getitem__`` to return a single sample tensor
           (or a tuple / dict for multi-modal tasks).
        2. Override ``__len__`` to return the total number of samples.
    """

    @abstractmethod
    def __getitem__(self, idx: int):
        """Return a single sample at *idx*."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples."""
        ...


# ============================================================================
# Concrete Implementation: ImageDataset
# ============================================================================


class ImageDataset(SemComDataset):
    """
    Unified image dataset with *local* and *Hugging Face Hub* dual-mode.

    Depending on ``source_type``, samples are loaded from either a local
    directory tree or a Hugging Face dataset repository.

    Args:
        source: Path to a local folder **or** a Hugging Face dataset name
            (e.g., ``"cifar10"``, ``"imagenet-1k"``).
        source_type: One of ``'auto'``, ``'local'``, ``'hf'``.

            - ``'local'``: treat *source* as a local directory path.
            - ``'hf'``:    treat *source* as a HuggingFace dataset identifier.
            - ``'auto'`` (default): if *source* resolves to an existing local
              directory, use local mode; otherwise fall back to HF mode.
        split: Dataset split for HF mode (``'train'``, ``'test'``,
            ``'validation'``). Ignored in local mode. Default: ``'train'``.
        transform: An optional ``torchvision.transforms`` (or any callable)
            applied to each PIL Image before returning.

    Raises:
        FileNotFoundError: In local mode, if *source* does not exist.
        RuntimeError: In HF mode, if the ``datasets`` library is not installed
            or the dataset cannot be loaded / the image column is not found.

    Example — local directory::

        ds = ImageDataset("./data/my_images", source_type="local",
                          transform=transforms.ToTensor())
        img = ds[0]          # torch.Tensor (C, H, W)

    Example — HuggingFace Hub::

        ds = ImageDataset("cifar10", source_type="hf", split="train",
                          transform=transforms.ToTensor())
        img = ds[0]          # torch.Tensor (C, H, W)
    """

    def __init__(
        self,
        source: str,
        source_type: str = "auto",
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.source = source
        self.split = split
        self.transform = transform

        # ----- resolve source_type -----
        if source_type == "auto":
            source_type = "local" if Path(source).is_dir() else "hf"
        if source_type not in ("local", "hf"):
            raise ValueError(
                f"source_type must be 'auto', 'local', or 'hf', "
                f"got '{source_type}'."
            )
        self.source_type = source_type

        # ----- initialise backend -----
        if self.source_type == "local":
            self._init_local(source)
        else:
            self._init_hf(source, split)

    # ------------------------------------------------------------------ local
    def _init_local(self, root: str) -> None:
        """Recursively scan *root* for image files."""
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(
                f"Local data directory not found: '{root}'."
            )

        self.files: List[Path] = sorted(
            p
            for p in root_path.rglob("*")
            if p.suffix.lower() in _IMAGE_EXTENSIONS and p.is_file()
        )

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No image files ({', '.join(_IMAGE_EXTENSIONS)}) "
                f"found under '{root}'."
            )

    # -------------------------------------------------------- hugging face
    def _init_hf(self, name: str, split: str) -> None:
        """Load a dataset from the Hugging Face Hub."""
        if not _HF_AVAILABLE:
            raise RuntimeError(
                "Hugging Face `datasets` library is required for HF mode. "
                "Install it with:  pip install datasets"
            )

        try:
            self.hf_dataset = hf_datasets.load_dataset(
                name, split=split, trust_remote_code=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HuggingFace dataset '{name}' "
                f"(split='{split}'). Original error:\n  {exc}"
            ) from exc

        # Auto-detect the image column name
        self.image_key = self._detect_image_key(self.hf_dataset)

    @staticmethod
    def _detect_image_key(dataset) -> str:
        """Find the first column that contains PIL Images.

        Checks common column names first, then falls back to feature-type
        inspection.

        Args:
            dataset: A HuggingFace ``datasets.Dataset`` instance.

        Returns:
            The column name containing images.

        Raises:
            RuntimeError: If no image column can be found.
        """
        # Fast path: check common names
        for candidate in ("image", "img", "pixel_values"):
            if candidate in dataset.column_names:
                return candidate

        # Slow path: inspect feature types
        for col_name, feature in dataset.features.items():
            if isinstance(feature, hf_datasets.Image):
                return col_name

        raise RuntimeError(
            f"Cannot detect image column. "
            f"Available columns: {dataset.column_names}. "
            f"Please ensure your dataset contains an image column "
            f"named 'image' or 'img', or of type datasets.Image."
        )

    # ---------------------------------------------------------------- API
    def __len__(self) -> int:
        if self.source_type == "local":
            return len(self.files)
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieve one sample and apply transforms.

        Args:
            idx: Sample index.

        Returns:
            Transformed image tensor of shape ``(C, H, W)``.
            If no transform is supplied, returns a PIL Image instead.
        """
        if self.source_type == "local":
            img = Image.open(self.files[idx]).convert("RGB")
        else:
            img = self.hf_dataset[idx][self.image_key]
            # HF may return a PIL Image or a dict; ensure PIL Image
            if not isinstance(img, Image.Image):
                raise TypeError(
                    f"Expected PIL.Image from HF dataset column "
                    f"'{self.image_key}', got {type(img)}."
                )
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img


# ============================================================================
# Utility Functions
# ============================================================================


def get_standard_transforms(
    crop_size: int = 128,
    is_train: bool = True,
) -> transforms.Compose:
    """
    Build a standard image transform pipeline for SemCom experiments.

    Args:
        crop_size: Desired spatial size (square crop). Default: 128.
        is_train: If ``True``, use random augmentation (RandomCrop +
            RandomHorizontalFlip). Otherwise, use deterministic CenterCrop.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline that produces a
        ``torch.Tensor`` of shape ``(3, crop_size, crop_size)`` with
        values in ``[0, 1]``.

    Example:
        >>> train_tf = get_standard_transforms(crop_size=32, is_train=True)
        >>> eval_tf  = get_standard_transforms(crop_size=32, is_train=False)
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomCrop(crop_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ]
        )


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """
    Wrap a dataset in a DataLoader with standard settings.

    This is a lightweight helper that wraps any PyTorch ``Dataset`` 
    (including :class:`ImageDataset`) in a ``DataLoader`` with 
    sensible defaults for SemCom experiments.

    Args:
        dataset: A PyTorch ``Dataset`` instance (e.g., :class:`ImageDataset`).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data at each epoch. Default: ``True``.
        num_workers: Number of worker processes for data loading. Default: 4.
        **kwargs: Extra keyword arguments forwarded to ``DataLoader``
            (e.g., ``drop_last=True``, ``pin_memory=False``).

    Returns:
        A ``torch.utils.data.DataLoader`` ready for training or evaluation.

    Example — Create dataset first, then wrap in DataLoader::

        # Step 1: Create transform and dataset
        transform = get_standard_transforms(crop_size=32, is_train=True)
        train_ds = ImageDataset(
            source="./data/cifar10", 
            source_type="local",
            transform=transform
        )
        
        # Step 2: Wrap in DataLoader
        train_loader = get_dataloader(
            dataset=train_ds, 
            batch_size=64, 
            shuffle=True
        )
        
        for batch in train_loader:
            print(batch.shape)   # (64, 3, 32, 32)
            break

    Example — Using with HuggingFace dataset::

        transform = get_standard_transforms(crop_size=128, is_train=False)
        val_ds = ImageDataset(
            source="imagenet-1k", 
            source_type="hf", 
            split="validation",
            transform=transform
        )
        val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )

    return loader
