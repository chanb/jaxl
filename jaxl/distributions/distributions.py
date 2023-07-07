from abc import abstractstaticmethod
from typing import Optional

import chex
import jax.numpy as jnp
import jax.scipy as js
import jax.random as jrandom
import math


class Distribution:
    """
    Abstract distribution class
    """

    @abstractstaticmethod
    def sample(
        *, key: jrandom.PRNGKey, num_samples: Optional[int] = None
    ) -> chex.Array:
        """
        Samples from the distribution

        :param *:
        :param key: the random number generator key
        :param num_samples: the number of samples to have
        :type key: jrandom.PRNGKey
        :type num_samples: Optional[int]:  (Default value = None)
        :return: the samples
        :rtype: chex.Array

        """
        raise NotImplementedError

    @abstractstaticmethod
    def lprob(*, x: chex.Array) -> chex.Array:
        """
        Computes the log probabilities.

        :param *:
        :param x: the samples
        :type x: chex.Array
        :return: the log probabilities of the given samples
        :rtype: chex.Array

        """
        raise NotImplementedError


class Normal(Distribution):
    """
    Normal distribution that extends the ``Distribution`` class
    """

    @staticmethod
    def sample(
        mean: chex.Array,
        std: chex.Array,
        key: jrandom.PRNGKey,
        num_samples: Optional[int] = None,
    ) -> chex.Array:
        """
        Samples from normal distribution.

        :param mean: mean of the normal distribution
        :param std: standard deviation of the normal distribution
        :param key: the random number generator key
        :param num_samples: the number of samples to have
        :type mean: chex.Array
        :type std: chex.Array
        :type key: jrandom.PRNGKey
        :type num_samples: Optional[int]:  (Default value = None)
        :return: the samples
        :rtype: chex.Array

        """
        if num_samples:
            shape = (num_samples, *mean.shape)
        else:
            shape = mean.shape
        return mean + jrandom.normal(key=key, shape=shape) * std

    def lprob(mean: chex.Array, std: chex.Array, x: chex.Array) -> chex.Array:
        """
        Computes the log probabilities with normal distribution.

        :param mean: mean of the normal distribution
        :param std: standard deviation of the normal distribution
        :param x: the samples
        :type mean: chex.Array
        :type std: chex.Array
        :type x: chex.Array
        :return: the log probabilities of the given samples
        :rtype: chex.Array

        """
        var = std**2
        log_std = jnp.log(std)
        return (
            -((x - mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
        )


class Softmax(Distribution):
    """
    Softmax distribution that extends the ``Distribution`` class
    """

    @staticmethod
    def sample(
        logits: chex.Array,
        key: jrandom.PRNGKey,
        num_samples: Optional[int] = None,
    ) -> chex.Array:
        """
        Samples from softmax distribution.

        :param logits: logits of the softmax distribution
        :param key: the random number generator key
        :param num_samples: the number of samples to have
        :type logits: chex.Array
        :type key: jrandom.PRNGKey
        :type num_samples: Optional[int]:  (Default value = None)
        :return: the samples
        :rtype: chex.Array

        """
        if num_samples:
            shape = (num_samples, *logits.shape[:-1], 1)
        else:
            shape = (*logits.shape[:-1], 1)
        return jrandom.categorical(key, logits, shape=shape)

    def lprob(logits: chex.Array, x: chex.Array) -> chex.Array:
        """
        Computes the log probabilities with softmax distribution.

        :param logits: logits of the softmax distribution
        :param x: the samples
        :type logits: chex.Array
        :type x: chex.Array
        :return: the log probabilities of the given samples
        :rtype: chex.Array

        """
        return jnp.sum(
            jnp.eye(logits.shape[-1])[x.astype(int)[..., 0]] * logits, axis=-1
        ) - js.special.logsumexp(logits, axis=-1)
