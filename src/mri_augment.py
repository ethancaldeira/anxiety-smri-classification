"""
mri_augment.py — TorchIO-backed augmentation for sMRI 2D slices.

Implements MRI-appropriate augmentation with TorchIO (Pérez-García et al.
2021, Computer Methods and Programs in Biomedicine,
doi: 10.1016/j.cmpb.2021.106236). Off by default (Config.USE_AUGMENTATION).

Transforms used:
  RandomFlip (LR axis only)  — never along the superior/inferior or
                                 A/P axes.
  RandomAffine               — small rotation / scale / translation;
                                 magnitudes chosen conservatively to
                                 preserve anatomical interpretability.
  RandomBiasField            — simulates residual RF-coil inhomogeneity
                                 (T1-specific artefact that CAT12 doesn't
                                 fully remove). Genuinely MRI-domain and
                                 has no Keras equivalent.
  RandomGamma                — intensity transfer function variation.
  RandomNoise                — low-magnitude Gaussian noise as a Rician
                                 proxy at typical T1 SNR.

Augmentation is applied to RAW intensity pixels (0–255) before the
backbone's preprocess_input (mean subtraction, channel order, etc.).
Bias-field multiplication only makes physical sense on raw intensities,
not on mean-subtracted ImageNet-space values.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import tensorflow as tf
import torch
import torchio as tio


def make_train_transform(config) -> tio.Compose:
    """Build the TorchIO Compose used during training.

    All knobs come from Config so they're snapshot to results/<run>/config.json
    for reproducibility.
    """
    return tio.Compose([
        tio.RandomFlip(axes=('LR',), flip_probability=config.AUG_FLIP_PROB),
        tio.RandomAffine(
            scales=(1.0 - config.AUG_SCALE, 1.0 + config.AUG_SCALE),
            degrees=config.AUG_DEGREES,
            translation=config.AUG_TRANSLATE_PX,
            image_interpolation='linear',
        ),
        tio.RandomBiasField(
            coefficients=config.AUG_BIAS_COEFF,
            order=config.AUG_BIAS_ORDER,
        ),
        tio.RandomGamma(log_gamma=(-config.AUG_LOG_GAMMA, config.AUG_LOG_GAMMA)),
        tio.RandomNoise(mean=0.0, std=(0.0, config.AUG_NOISE_STD)),
    ])


def _np_to_tio_subject(img_hwc: np.ndarray) -> tio.Subject:
    """numpy (H, W, 3) uint8/float -> tio.Subject with a single image of shape (3, H, W, 1)."""
    # TorchIO expects (C, H, W, D); add a singleton D for 2D slices.
    arr = np.transpose(img_hwc, (2, 0, 1))[..., None].astype(np.float32)
    return tio.Subject(image=tio.ScalarImage(tensor=torch.from_numpy(arr)))


def _tio_subject_to_np(subj: tio.Subject) -> np.ndarray:
    """tio.Subject -> numpy (H, W, 3)."""
    arr = subj.image.data.numpy()             # (3, H, W, 1)
    arr = arr[..., 0]                         # drop singleton D
    arr = np.transpose(arr, (1, 2, 0))        # (H, W, 3)
    return arr


def apply_transform_batch(transform: tio.Compose, batch: np.ndarray) -> np.ndarray:
    """Apply a TorchIO Compose to every image in a (B, H, W, 3) batch."""
    out = np.empty_like(batch)
    for i in range(batch.shape[0]):
        out[i] = _tio_subject_to_np(transform(_np_to_tio_subject(batch[i])))
    return out


class MRIBatchSequence(tf.keras.utils.Sequence):
    """Keras Sequence wrapping numpy arrays.

    - For training, pass `train_transform=tio.Compose(...)`; transforms run
      per-batch in __getitem__ so each epoch sees freshly-augmented samples.
    - For val/test, pass `train_transform=None` (no augmentation).
    - `preprocess_fn` is the backbone-specific preprocess_input (ResNet50 /
      DenseNet121 / InceptionV3); it runs AFTER augmentation on raw pixels.

    Class weights and shuffling are handled by Keras `model.fit(...)` via
    its `class_weight=` arg and the `shuffle=True` default, so we shuffle
    the indices ourselves at epoch end.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        preprocess_fn,
        train_transform: Optional[tio.Compose] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        assert len(X) == len(y), "X and y must have matching length"
        self.X = X
        self.y = y.astype(np.float32)
        self.batch_size = int(batch_size)
        self.preprocess_fn = preprocess_fn
        self.train_transform = train_transform
        self.shuffle = bool(shuffle)
        self._rng = np.random.default_rng(seed)
        self._indices = np.arange(len(X))
        if self.shuffle:
            self._rng.shuffle(self._indices)

    def __len__(self):
        return int(math.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        sel = self._indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.X[sel].astype(np.float32)
        batch_y = self.y[sel]
        if self.train_transform is not None:
            batch_x = apply_transform_batch(self.train_transform, batch_x)
        if self.preprocess_fn is not None:
            batch_x = self.preprocess_fn(batch_x)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            self._rng.shuffle(self._indices)
