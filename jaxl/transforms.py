from PIL import Image

import chex
import numpy as np
import torchvision.transforms as torch_transforms


class DefaultPILToImageTransform:
    """
    A transform that converts PIL to a tensor scaled between 0 to 1.
    Assumes that the PIL images are in range [0, 255].
    """

    def __call__(self, img: Image) -> chex.Array:
        img = torch_transforms.functional.pil_to_tensor(img)
        img = torch_transforms.functional.convert_image_dtype(img) / 255.0
        return img


class StandardImageTransform:
    """
    A transform that converts uint8 to float tensor.
    Assumes that the images are in range [0, 255].
    """

    def __call__(self, img: chex.Array) -> chex.Array:
        img = torch_transforms.functional.convert_image_dtype(img) / 255.0
        return img
