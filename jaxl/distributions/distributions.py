from abc import abstractmethod
from jax.scipy.special import entr
from typing import Optional

import chex
import jax.nn as nn
import jax.numpy as jnp
import jax.random as jrandom
import math
import optax


class Distribution:
    """
    Abstract distribution class
    """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def entropy(*args) -> chex.Array:
        """
        Computes the entropy of the distribution.

        :return: the entropy of the distribution
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

    @staticmethod
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

    @staticmethod
    def entropy(mean: chex.Array, std: chex.Array) -> chex.Array:
        """
        Computes the entropy of the normal distribution.

        :param mean: mean of the normal distribution
        :param std: standard deviation of the normal distribution
        :type mean: chex.Array
        :type std: chex.Array
        :return: the entropy
        :rtype: chex.Array

        """
        return 0.5 + 0.5 * math.log(2 * math.pi) + jnp.log(std)


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

    @staticmethod
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
            nn.one_hot(x[..., 0], num_classes=logits.shape[-1]) * logits, axis=-1
        ) - nn.logsumexp(logits, axis=-1)

    @staticmethod
    def entropy(logits: chex.Array) -> chex.Array:
        """
        Computes the entropy with softmax distribution.

        :param logits: logits of the softmax distribution
        :type logits: chex.Array
        :return: the entropy
        :rtype: chex.Array

        """
        probs = nn.softmax(logits, axis=-1)
        return optax.softmax_cross_entropy(logits, probs)


class Bernoulli(Distribution):
    """
    Bernoulli distribution that extends the ``Distribution`` class
    """

    @staticmethod
    def sample(
        logits: chex.Array,
        key: jrandom.PRNGKey,
        num_samples: Optional[int] = None,
    ) -> chex.Array:
        """
        Samples from bernoulli distribution.

        :param logits: the logits of the bernoulli distribution
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
            shape = logits.shape
        return jrandom.bernoulli(key, nn.sigmoid(logits), shape=shape)

    @staticmethod
    def lprob(logits: chex.Array, x: chex.Array) -> chex.Array:
        """
        Computes the log probabilities with bernoulli distribution.

        :param logits: the logits of the bernoulli distribution
        :param x: the samples
        :type logits: chex.Array
        :type x: chex.Array
        :return: the log probabilities of the given samples
        :rtype: chex.Array

        """
        return optax.sigmoid_binary_cross_entropy(logits, x)

    @staticmethod
    def entropy(logits: chex.Array) -> chex.Array:
        """
        Computes the entropy with bernoulli distribution.

        :param logits: the logits of the bernoulli distribution
        :type logits: chex.Array
        :return: the entropy of the given samples
        :rtype: chex.Array

        """
        probs = nn.sigmoid(logits)
        return optax.sigmoid_binary_cross_entropy(logits, probs)
