"""RandAugment-style transforms (albumentations)."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def rand_augment_transforms(img_size: int, num_ops: int = 2, magnitude: int = 9) -> A.Compose:
    """RandAugment 근사 — num_ops=2, magnitude≈9."""
    m = magnitude / 10.0
    pool = [
        A.Rotate(limit=int(15 + 15 * m), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05 * m, scale_limit=0.1 * m,
            rotate_limit=int(10 + 10 * m), p=1.0,
        ),
        A.RandomBrightnessContrast(0.2 * m, 0.2 * m, p=1.0),
        A.HueSaturationValue(
            int(10 * m), int(15 * m), int(10 * m), p=1.0,
        ),
        A.GaussNoise(std_range=(0.02, 0.06 * m), p=1.0),
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(int(img_size * 0.05), int(img_size * 0.12)),
            hole_width_range=(int(img_size * 0.05), int(img_size * 0.12)),
            p=1.0,
        ),
    ]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.SomeOf(pool, n=min(num_ops, len(pool)), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
