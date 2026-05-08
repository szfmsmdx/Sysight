"""Data pipeline with augmentation and preprocessing.

BUGS:
  F01: Augmentation done on CPU, then transferred to GPU — should use GPU augmentation
  F02: Normalize called per-sample instead of batched
  F03: pin_memory not used in DataLoader
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ImageTransformPipeline:
    """Image augmentation and preprocessing pipeline.

    BUG F01: All transforms run on CPU. For large batches, GPU-based
    augmentation (kornia, torchvision GPU ops) would be faster.
    BUG F02: normalize() called per-sample — should be batched.
    """

    def __init__(self, image_size: int = 224, mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)):
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation.

        BUG F01: CPU-side augmentation — should use GPU ops for batch processing.
        """
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[-1])

        # Random crop + resize
        _, H, W = image.shape
        crop_size = int(self.image_size * 0.8)
        if H > crop_size and W > crop_size:
            top = torch.randint(0, H - crop_size, (1,)).item()
            left = torch.randint(0, W - crop_size, (1,)).item()
            image = image[:, top:top+crop_size, left:left+crop_size]

        # Resize
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return image

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image.

        BUG F02: Called per-sample — should normalize entire batch at once.
        """
        return (image - self.mean.squeeze(0)) / self.std.squeeze(0)


class SyntheticImageDataset(Dataset):
    """Synthetic image dataset with augmentation."""

    def __init__(
        self,
        num_samples: int,
        image_size: int = 224,
        num_classes: int = 100,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes

        g = torch.Generator()
        g.manual_seed(seed)
        self.images = torch.rand(num_samples, 3, image_size, image_size, generator=g)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

        self.transform = ImageTransformPipeline(image_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]

        # BUG F01: CPU augmentation
        image = self.transform.augment(image)
        # BUG F02: per-sample normalize
        image = self.transform.normalize(image)

        return {"image": image, "label": label}
