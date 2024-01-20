from torch.utils.data import Dataset
from typing import Tuple

from jaxl.constants import VALID_MNIST_TASKS

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

import jaxl.transforms as jaxl_transforms


def construct_mnist(
    save_path: str,
    task_name: str = None,
    train: bool = True,
) -> Dataset:
    """
    Constructs a customized MNIST dataset.

    :param save_path: the path to store the MNIST dataset
    :param task_name: the task to construct
    :type save_path: str
    :type task_name: str:  (Default value = None)
    :return: Customized MNIST dataset
    :rtype: Dataset

    """
    assert (
        task_name is None or task_name in VALID_MNIST_TASKS
    ), f"{task_name} is not supported (one of {VALID_MNIST_TASKS})"
    input_transforms = [jaxl_transforms.DefaultPILToImageTransform()]
    target_transform = None

    if task_name is None:
        # By default, the MNIST task will be normalized to be between 0 to 1.
        pass
    else:
        raise ValueError(f"{task_name} is invalid (one of {VALID_MNIST_TASKS})")

    return torch_datasets.MNIST(
        save_path,
        train=train,
        download=True,
        transform=torch_transforms.Compose(input_transforms),
        target_transform=target_transform,
    )


class MultitaskMNISTFineGrain(Dataset):
    """
    The dataset contains multiple ND (fixed) linear classification problems.
    """

    def __init__(
        self,
        num_sequences: int,
        sequence_length: int,
        input_dim: int,
        seed: int = 0,
        inputs_range: Tuple[float, float] = (-1.0, 1.0),
        save_path: str = None,
    ):
        pass
