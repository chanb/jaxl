import chex
import jax.numpy as jnp


from optax import Schedule


def linear_warmup_sqrt_decay(
    max_lr: chex.Scalar,
    warmup_steps: int,
) -> Schedule:
    """
    Returns a scheduler that performs linear warmup followed by an inverse square root decay of learning rate.
    Reference: https://arxiv.org/pdf/2205.05055.pdf
    """
    assert max_lr > 0, "maximum learning rate {} must be positive".format(max_lr)
    assert warmup_steps > 0, "warm up steps {} must be positive".format(warmup_steps)

    def schedule(count):
        """Linear warmup and then an inverse square root decay of learning rate."""
        linear_ratio = max_lr / warmup_steps
        decay_ratio = jnp.power(warmup_steps * 1.0, 0.5) * max_lr
        return jnp.min(
            jnp.array([linear_ratio * (count + 1), decay_ratio * jnp.power((count + 1), -0.5)])
        )

    return schedule
