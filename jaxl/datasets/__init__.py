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
        from jaxl.datasets.tight_frame_classification import (
            TightFrameClassification,
            TightFrameClassificationNShotKWay,
            TightFrameAbstractClassification,
            OneSequencePerClassTightFrameClassification,
        )

        task_name = getattr(dataset_kwargs, "task_name", None)

        if task_name == "n_shot_k_way":
            dataset = TightFrameClassificationNShotKWay(
                tight_frame_path=dataset_kwargs.tight_frame_path,
                num_sequences=dataset_kwargs.num_sequences,
                sequence_length=dataset_kwargs.sequence_length,
                num_holdout=dataset_kwargs.num_holdout,
                split=dataset_kwargs.split,
                k_way=dataset_kwargs.k_way,
                perturb_query=getattr(dataset_kwargs, "perturb_query", False),
                perturb_context=getattr(dataset_kwargs, "perturb_context", False),
                perturb_noise=getattr(dataset_kwargs, "perturb_noise", 0.001),
                seed=seed,
            )
        elif task_name == "abstract_class":
            dataset = TightFrameAbstractClassification(
                tight_frame_path=dataset_kwargs.tight_frame_path,
                num_sequences=dataset_kwargs.num_sequences,
                sequence_length=dataset_kwargs.sequence_length,
                num_holdout=dataset_kwargs.num_holdout,
                split=dataset_kwargs.split,
                abstraction=dataset_kwargs.abstraction,
                seed=seed,
            )
        elif task_name == "one_sequence_per_class":
            dataset = OneSequencePerClassTightFrameClassification(
                tight_frame_path=dataset_kwargs.tight_frame_path,
                sequence_length=dataset_kwargs.sequence_length,
                num_holdout=dataset_kwargs.num_holdout,
                split=dataset_kwargs.split,
                p_bursty=dataset_kwargs.p_bursty,
                bursty_len=getattr(dataset_kwargs, "bursty_len", 3),
                unique_classes=getattr(dataset_kwargs, "unique_classes", False),
                perturb_query=getattr(dataset_kwargs, "perturb_query", False),
                perturb_context=getattr(dataset_kwargs, "perturb_context", False),
                seed=seed,
            )
        else:
            dataset = TightFrameClassification(
                tight_frame_path=dataset_kwargs.tight_frame_path,
                num_sequences=dataset_kwargs.num_sequences,
                sequence_length=dataset_kwargs.sequence_length,
                num_holdout=dataset_kwargs.num_holdout,
                split=dataset_kwargs.split,
                p_bursty=dataset_kwargs.p_bursty,
                bursty_len=getattr(dataset_kwargs, "bursty_len", 3),
                unique_classes=getattr(dataset_kwargs, "unique_classes", False),
                p_random_label=getattr(dataset_kwargs, "p_random_label", 0.0),
                perturb_query=getattr(dataset_kwargs, "perturb_query", False),
                perturb_context=getattr(dataset_kwargs, "perturb_context", False),
                perturb_noise=getattr(dataset_kwargs, "perturb_noise", 0.001),
                novel_query=getattr(dataset_kwargs, "novel_query", False),
                zipf_exp=getattr(dataset_kwargs, "zipf_exp", 0.0),
                seed=seed,
            )
    elif dataset_config.dataset_name == CONST_REDDY:
        import jaxl.datasets.icl.reddy as data

        dataset = data.get_dataset(
            num_examples=dataset_kwargs.num_examples,
            p_bursty=dataset_kwargs.p_bursty,
            bursty_len=dataset_kwargs.bursty_len,
            zipf_exp=dataset_kwargs.zipf_exp,
            input_noise_std=dataset_kwargs.input_noise_std,
            target_allowed_in_example=dataset_kwargs.target_allowed_in_example,
            empty_examples=getattr(dataset_kwargs, "empty_examples", False),
            num_base_classes=dataset_kwargs.num_base_classes,
            num_abstract_classes=dataset_kwargs.num_abstract_classes,
            num_dims=dataset_kwargs.num_dims,
            seed=seed,
            base_per_abstract_map=dataset_kwargs.base_per_abstract_map,
            novel_abstract_class=getattr(dataset_kwargs, "novel_abstract_class", False),
        )
    elif dataset_config.dataset_name == CONST_STREAM_BLOCK:
        import jaxl.datasets.icl.stream_block as data

        dataset = data.get_dataset(
            num_examples=dataset_kwargs.num_examples,
            zipf_exp=dataset_kwargs.zipf_exp,
            input_noise_std=dataset_kwargs.input_noise_std,
            num_base_classes=dataset_kwargs.num_base_classes,
            num_clusters=dataset_kwargs.num_clusters,
            num_abstract_classes=dataset_kwargs.num_abstract_classes,
            num_dims=dataset_kwargs.num_dims,
            seed=seed,
            novel_abstract_class=getattr(dataset_kwargs, "novel_abstract_class", False),
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
