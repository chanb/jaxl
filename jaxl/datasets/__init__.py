from torch.utils.data import Dataset
from types import SimpleNamespace
from typing import Callable

from jaxl.constants import *
from jaxl.datasets.basis_regression import (
    MultitaskFixedBasisRegression1D,
)
from jaxl.datasets.linear_regression import (
    MultitaskLinearRegressionND,
)
from jaxl.datasets.linear_classification import (
    MultitaskLinearClassificationND,
)
from jaxl.datasets.mnist import construct_mnist
from jaxl.datasets.omniglot import construct_omniglot
from jaxl.datasets.tf_omniglot import utils
from jaxl.datasets.wrappers import DatasetWrapper

import chex
import numpy as np

import jaxl.datasets.wrappers as dataset_wrappers


"""
Getters for the datasets.
XXX: Feel free to add new components as needed.
"""


def get_basis(dataset_kwargs: SimpleNamespace) -> Callable[..., chex.Array]:
    """
    Gets a basis function.

    :param dataset_config: the dataset configuration
    :type dataset_config: SimpleNamespace
    :return: the basis function
    :rtype: Callable[..., chex.Array]
    """
    assert (
        dataset_kwargs.basis in VALID_BASIS_FUNCTION
    ), f"{dataset_kwargs.basis} is not supported (one of {VALID_BASIS_FUNCTION})"

    if dataset_kwargs.basis == CONST_BASIS_POLYNOMIAL:

        def polynomial_basis(x):
            return x ** np.arange(dataset_kwargs.degree + 1)

        return polynomial_basis
    else:
        raise ValueError(
            f"{dataset_kwargs.basis} is invalid (one of {VALID_BASIS_FUNCTION})"
        )


def get_dataset(
    dataset_config: SimpleNamespace,
    seed: int,
) -> DatasetWrapper:
    """
    Gets a dataset.

    :param dataset_config: the dataset configuration
    :param seed: the seed to generate the dataset
    :type dataset_config: SimpleNamespace
    :type seed: int
    :return: the dataset
    :rtype: DatasetWrapper
    """
    assert (
        dataset_config.dataset_name in VALID_DATASET
    ), f"{dataset_config.dataset_name} is not supported (one of {VALID_DATASET})"

    dataset_kwargs = dataset_config.dataset_kwargs
    if dataset_config.dataset_name == CONST_MULTITASK_TOY_REGRESSION:
        basis = get_basis(dataset_kwargs=dataset_kwargs)
        dataset = MultitaskFixedBasisRegression1D(
            num_sequences=dataset_kwargs.num_sequences,
            sequence_length=dataset_kwargs.sequence_length,
            basis=basis,
            seed=seed,
            noise=dataset_kwargs.noise,
            params_bound=getattr(dataset_kwargs, "params_bound", [-0.5, 0.5]),
            inputs_range=getattr(dataset_kwargs, "inputs_range", [-1.0, 1.0]),
        )
    elif dataset_config.dataset_name == CONST_MULTITASK_ND_LINEAR_REGRESSION:
        dataset = MultitaskLinearRegressionND(
            num_sequences=dataset_kwargs.num_sequences,
            sequence_length=dataset_kwargs.sequence_length,
            input_dim=dataset_kwargs.input_dim,
            seed=seed,
            noise=dataset_kwargs.noise,
            params_bound=getattr(dataset_kwargs, "params_bound", [-0.5, 0.5]),
            inputs_range=getattr(dataset_kwargs, "inputs_range", [-1.0, 1.0]),
            num_active_params=getattr(
                dataset_kwargs, "num_active_params", dataset_kwargs.input_dim + 1
            ),
        )
    elif dataset_config.dataset_name == CONST_MULTITASK_ND_LINEAR_CLASSIFICATION:
        dataset = MultitaskLinearClassificationND(
            num_sequences=dataset_kwargs.num_sequences,
            sequence_length=dataset_kwargs.sequence_length,
            input_dim=dataset_kwargs.input_dim,
            seed=seed,
            noise=dataset_kwargs.noise,
            params_bound=getattr(dataset_kwargs, "params_bound", [-0.5, 0.5]),
            inputs_range=getattr(dataset_kwargs, "inputs_range", [-1.0, 1.0]),
            num_active_params=getattr(
                dataset_kwargs, "num_active_params", dataset_kwargs.input_dim + 1
            ),
            bias=getattr(
                dataset_kwargs,
                "bias",
                False,
            ),
            margin=getattr(
                dataset_kwargs,
                "margin",
                0.0,
            ),
            save_path=getattr(
                dataset_kwargs,
                "save_path",
                None,
            ),
        )
    elif dataset_config.dataset_name == CONST_MNIST:
        dataset = construct_mnist(
            dataset_kwargs.save_path,
            dataset_kwargs.task_name,
            dataset_kwargs.task_config,
            seed=seed,
            train=getattr(dataset_kwargs, "train", True),
            remap=getattr(dataset_kwargs, "remap", False),
        )
    elif dataset_config.dataset_name == CONST_OMNIGLOT:
        dataset = construct_omniglot(
            dataset_kwargs.save_path,
            dataset_kwargs.task_name,
            dataset_kwargs.task_config,
            seed=seed,
            train=getattr(dataset_kwargs, "train", True),
            remap=getattr(dataset_kwargs, "remap", False),
        )
    elif dataset_config.dataset_name == CONST_OMNIGLOT_TF:
        dataset = utils.get_omniglot_seq_generator(
            dataset_kwargs,
            seed,
        )
    elif dataset_config.dataset_name == CONST_TIGHT_FRAME:
        from jaxl.datasets.tight_frame_classification import TightFrameClassification

        dataset = TightFrameClassification(
            tight_frame_path=dataset_kwargs.tight_frame_path,
            num_sequences=dataset_kwargs.num_sequences,
            sequence_length=dataset_kwargs.sequence_length,
            num_holdout=dataset_kwargs.num_holdout,
            split=dataset_kwargs.split,
            p_bursty=dataset_kwargs.p_bursty,
            random_label=getattr(dataset_kwargs, "random_label", False),
            seed=seed,
        )
    else:
        raise ValueError(
            f"{dataset_config.dataset_name} is not supported (one of {VALID_DATASET})"
        )

    if hasattr(dataset_config, CONST_DATASET_WRAPPER):
        wrapper_constructor = getattr(
            dataset_wrappers, dataset_config.dataset_wrapper.type, None
        )
        if wrapper_constructor:
            dataset = wrapper_constructor(
                dataset, **vars(dataset_config.dataset_wrapper.kwargs)
            )
        else:
            raise NotImplementedError
    else:
        dataset = DatasetWrapper(dataset)

    return dataset
