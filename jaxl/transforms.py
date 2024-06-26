from PIL import Image
from typing import Sequence

import chex
import torch
import torchvision.transforms as torch_transforms


class DefaultPILToImageTransform:
    """
    A transform that converts PIL to a tensor scaled between 0 to 1.
    Assumes that the PIL images are in range [0, 255].
    """

    def __init__(self, scale: float = 255.0):
        self.scale = scale

    def __call__(self, img: Image) -> chex.Array:
        img = torch_transforms.functional.pil_to_tensor(img)
        img = torch_transforms.functional.convert_image_dtype(img) / self.scale
        return img


class Transpose:
    """
    A transform that transposes a tensor
    """

    def __init__(self, axes: Sequence[int]):
        self.axes = axes

    def __call__(self, x: chex.Array) -> chex.Array:
        return torch.permute(x, self.axes)


class StandardImageTransform:
    """
    A transform that converts uint8 to float tensor.
    Assumes that the images are in range [0, 255].
    """

    def __call__(self, img: chex.Array) -> chex.Array:
        img = torch_transforms.functional.convert_image_dtype(img) / 255.0
        return img


class GaussianNoise(object):
    """
    A transform that adds Gaussian noise to each feature
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.std = std
        self.mean = mean

    def __call__(self, x: chex.Array):
        return x + torch.randn(x.size()) * self.std + self.mean
