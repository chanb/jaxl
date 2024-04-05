from flax import linen as nn
from jax import lax
from jax.nn import initializers
from flax.linen import dtypes
from flax.typing import (
    Array,
    PRNGKey as PRNGKey,
    Dtype,
    Shape as Shape,
    Initializer,
    Axes,
)
from typing import Any, Tuple, Optional, Iterable

import jax
import jax.numpy as jnp


def _abs_sq(x):
    """Computes the elementwise square of the absolute value |x|^2."""
    if jnp.iscomplexobj(x):
        return lax.square(lax.real(x)) + lax.square(lax.imag(x))
    else:
        return lax.square(x)


def _compute_stats(
    x: Array,
    axes: Axes,
    dtype: Optional[Dtype],
    axis_name: Optional[str] = None,
    axis_index_groups: Any = None,
    use_mean: bool = True,
    use_fast_variance: bool = True,
    mask: Optional[Array] = None,
):
    """Computes mean and variance statistics.

    This implementation takes care of a few important details:
    - Computes in float32 precision for stability in half precision training.
    - If `use_fast_variance` is `True`, mean and variance are computed using
      Var = E[|x|^2] - |E[x]|^2, instead of Var = E[|x - E[x]|^2]), in a single
      XLA fusion.
    - Clips negative variances to zero which can happen due to
      roundoff errors. This avoids downstream NaNs.
    - Supports averaging across a parallel axis and subgroups of a parallel axis
      with a single `lax.pmean` call to avoid latency.

    Arguments:
      x: Input array.
      axes: The axes in ``x`` to compute mean and variance statistics for.
      dtype: Optional dtype specifying the minimal precision. Statistics are
        always at least float32 for stability (default: dtype of x).
      axis_name: Optional name for the pmapped axis to compute mean over. Note,
        this is only used for pmap and shard map. For SPMD jit, you do not need to
        manually synchronize. Just make sure that the axes are correctly annotated
        and XLA:SPMD will insert the necessary collectives.
      axis_index_groups: Optional axis indices.
      use_mean: If true, calculate the mean from the input and use it when
        computing the variance. If false, set the mean to zero and compute the
        variance without subtracting the mean.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
      mask: Binary array of shape broadcastable to `inputs` tensor, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      A pair ``(mean, var)``.
    """
    if dtype is None:
        dtype = jnp.result_type(x)
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    dtype = jnp.promote_types(dtype, jnp.float32)
    x = jnp.asarray(x, dtype)
    axes = _canonicalize_axes(x.ndim, axes)

    def maybe_distributed_mean(*xs, mask=None):
        mus = tuple(x.mean(axes, where=mask) for x in xs)
        if axis_name is None:
            return mus if len(xs) > 1 else mus[0]
        else:
            # In the distributed case we stack multiple arrays to speed comms.
            if len(xs) > 1:
                reduced_mus = lax.pmean(
                    jnp.stack(mus, axis=0),
                    axis_name,
                    axis_index_groups=axis_index_groups,
                )
                return tuple(reduced_mus[i] for i in range(len(xs)))
            else:
                return lax.pmean(mus[0], axis_name, axis_index_groups=axis_index_groups)

    if use_mean:
        if use_fast_variance:
            mu, mu2 = maybe_distributed_mean(x, _abs_sq(x), mask=mask)
            # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
            # to floating point round-off errors.
            var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
        else:
            mu = maybe_distributed_mean(x, mask=mask)
            var = maybe_distributed_mean(
                _abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask
            )
    else:
        var = maybe_distributed_mean(_abs_sq(x), mask=mask)
        mu = jnp.zeros_like(var)
    return mu, var


def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, Iterable):
        axes = (axes,)
    return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _normalize(
    mdl: nn.Module,
    x: Array,
    mean: Array,
    var: Array,
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Optional[Dtype],
    param_dtype: Dtype,
    epsilon: float,
    use_bias: bool,
    use_scale: bool,
    bias_init: Initializer,
    scale_init: Initializer,
):
    """Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

    Arguments:
      mdl: Module to apply the normalization in (normalization params will reside
        in this module).
      x: The input.
      mean: Mean to use for normalization.
      var: Variance to use for normalization.
      reduction_axes: The axes in ``x`` to reduce.
      feature_axes: Axes containing features. A separate bias and scale is learned
        for each specified feature.
      dtype: The dtype of the result (default: infer from input and params).
      param_dtype: The dtype of the parameters.
      epsilon: Normalization epsilon.
      use_bias: If true, add a bias term to the output.
      use_scale: If true, scale the output.
      bias_init: Initialization function for the bias term.
      scale_init: Initialization function for the scaling function.

    Returns:
      The normalized input.
    """
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])

    mean = jnp.expand_dims(mean, reduction_axes)
    var = jnp.expand_dims(var, reduction_axes)
    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    args = [x]
    if use_scale:
        scale = mdl.param(
            "scale", scale_init, reduced_feature_shape, param_dtype
        ).reshape(feature_shape)
        mul *= scale
        args.append(scale)
    y *= mul
    if use_bias:
        bias = mdl.param("bias", bias_init, reduced_feature_shape, param_dtype).reshape(
            feature_shape
        )
        y += bias
        args.append(bias)
    dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


class DebiasedBatchNorm(nn.Module):
    """BatchNorm Module."""

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Initializer = initializers.zeros
    scale_init: Initializer = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @nn.compact
    def __call__(
        self,
        x,
        use_running_average: Optional[bool] = None,
        *,
        mask: Optional[jax.Array] = None,
    ):
        """Normalizes the input using batch statistics.

        .. note::
          During initialization (when ``self.is_initializing()`` is ``True``) the running
          average of the batch statistics will not be updated. Therefore, the inputs
          fed during initialization don't need to match that of the actual input
          distribution and the reduction axis (set with ``axis_name``) does not have
          to exist.

        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.
          mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = nn.module.merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
        )
        ra_counter = self.variable(
            "batch_stats", "counter", lambda: jnp.ones([], jnp.float32)
        )

        if use_running_average:
            one = jnp.ones([], jnp.float32)
            mean, var, counter = ra_mean.value, ra_var.value, ra_counter.value
            mean /= one - jnp.power(self.momentum, counter)
            var /= one - jnp.power(self.momentum, counter)
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
                mask=mask,
            )

            if not self.is_initializing():
                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

                ra_counter.value += 1

        return _normalize(
            self,
            x,
            mean,
            var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )
