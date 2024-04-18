from abc import abstractmethod, ABC

import chex
import jax
import math


class Transform(ABC):
    """
    Abstract transform class.
    """

    @abstractmethod
    def transform(x: chex.Array) -> chex.Array:
        """
        Transforms the samples.

        :param x: the samples
        :type x: chex.Array
        :return: the transformed samples
        :rtype: chex.Array

        """
        pass

    @abstractmethod
    def log_abs_det_jacobian(self, x: chex.Array, x_t: chex.Array) -> chex.Array:
        """
        Computes the "adjustment" term for the transformation when
        computing the log probabilities.

        :param x: the samples
        :param x_t: the transformed samples
        :type x: chex.Array
        :type x_t: chex.Array
        :return: the "adjustment" term for transformed log probabilities.
        :rtype: chex.Array

        """
        pass

    @abstractmethod
    def inv(self, x: chex.Array) -> chex.Array:
        """
        Invert the transformed samples.

        :param x: the transformed samples
        :type x: chex.Array
        :return: the original samples
        :rtype: chex.Array

        """
        pass


class TanhTransform(Transform):
    """
    The hyperbolic tangent transform (``tanh``).
    """

    def transform(x: chex.Array) -> chex.Array:
        """
        Transforms the samples using ``tanh``.

        :param x: the samples
        :type x: chex.Array
        :return: the transformed samples
        :rtype: chex.Array

        """
        return jax.nn.tanh(x)

    def log_abs_det_jacobian(x: chex.Array, x_t: chex.Array) -> chex.Array:
        """
        Computes the "adjustment" term for the ``tanh`` when
        computing the log probabilities.

        :param x: the samples
        :param x_t: the transformed samples
        :type x: chex.Array
        :type x_t: chex.Array
        :return: the "adjustment" term for transformed log probabilities.
        :rtype: chex.Array

        """
        return 2.0 * (math.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    @abstractmethod
    def inv(self, x: chex.Array) -> chex.Array:
        """
        Invert the transformed samples.

        :param x: the transformed samples
        :type x: chex.Array
        :return: the original samples
        :rtype: chex.Array

        """
        return jax.nn.arctanh(x)
