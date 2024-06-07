from abc import ABC
from types import SimpleNamespace
from typing import Any, Callable, Dict, Union, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from jaxl.constants import *
from jaxl.distributions import Normal
from jaxl.distributions.transforms import TanhTransform
from jaxl.models.common import Model
from jaxl.models.policies import StochasticPolicy


class SquashedGaussianResidualPolicy(StochasticPolicy):
    """
    Squashed Gaussian Residual Policy. Assumes backbone is deterministic.
    TODO: Think if we need to pass backbone's action to the model
    """

    def __init__(
        self,
        backbone: Model,
        model: Model,
        backbone_absorbing_state: bool = False,
        residual_impact: float = 0.5,
        min_std: float = DEFAULT_MIN_STD,
        std_transform: Callable[chex.Array, chex.Array] = jax.nn.squareplus,
        use_backbone_only: bool = False,
    ):
        self.residual_impact = residual_impact
        self.reset = jax.jit(self.make_reset(model))
        self._min_std = min_std
        self._std_transform = std_transform
        self.use_backbone_only = 1 - int(use_backbone_only)

        if backbone_absorbing_state:

            def process_backbone_obs(x):
                return jnp.concatenate((x, jnp.zeros_like(x)[..., [0]]), axis=-1)

        else:

            def process_backbone_obs(x):
                return x

        self.process_backbone_obs = process_backbone_obs

        self.deterministic_action = jax.jit(
            self.make_deterministic_action(backbone, model)
        )
        self.random_action = jax.jit(self.make_random_action(backbone, model))
        self.compute_action = jax.jit(self.make_compute_action(backbone, model))
        self.act_lprob = jax.jit(
            self.make_act_lprob(backbone, model), static_argnames=[CONST_EVAL]
        )
        self.lprob = jax.jit(
            self.make_lprob(backbone, model), static_argnames=[CONST_EVAL]
        )

    def make_reset(self, model: Model) -> Callable[..., chex.Array]:
        def _reset() -> chex.Array:
            return model.reset_h_state()

        return _reset

    def make_compute_action(self, backbone: Model, model: Model) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        def compute_action(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array]:
            backbone_act, h_state, _ = backbone.forward(
                params[CONST_BACKBONE],
                self.process_backbone_obs(obs),
                h_state,
                eval=True,
                **kwargs,
            )
            act_params, h_state, _ = model.forward(
                params[CONST_RESIDUAL], obs, h_state, eval=True, **kwargs
            )
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = self._std_transform(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            act = TanhTransform.transform(act)
            act = self.use_backbone_only * self.residual_impact * act + jax.nn.tanh(
                backbone_act
            )
            return act, h_state

        return compute_action

    def make_deterministic_action(self, backbone: Model, model: Model) -> Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array],
        Tuple[chex.Array, chex.Array],
    ]:
        def deterministic_action(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array]:
            backbone_act, h_state, _ = backbone.forward(
                params[CONST_BACKBONE],
                self.process_backbone_obs(obs),
                h_state,
                eval=True,
                **kwargs,
            )
            act_params, h_state, _ = model.forward(
                params[CONST_RESIDUAL], obs, h_state, eval=True, **kwargs
            )
            act_mean, _ = jnp.split(act_params, 2, axis=-1)
            act_mean = TanhTransform.transform(act_mean)
            act_mean = (
                self.use_backbone_only * self.residual_impact * act_mean
                + jax.nn.tanh(backbone_act)
            )
            return act_mean, h_state

        return deterministic_action

    def make_random_action(self, backbone: Model, model: Model) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        def random_action(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array]:
            backbone_act, h_state, _ = backbone.forward(
                params[CONST_BACKBONE],
                self.process_backbone_obs(obs),
                h_state,
                eval=True,
                **kwargs,
            )
            act_params, h_state, _ = model.forward(
                params[CONST_RESIDUAL], obs, h_state, eval=True, **kwargs
            )
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = self._std_transform(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            act = TanhTransform.transform(act)
            act = self.use_backbone_only * self.residual_impact * act + jax.nn.tanh(
                backbone_act
            )
            return act, h_state

        return random_action

    def make_act_lprob(self, backbone: Model, model: Model) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            jrandom.PRNGKey,
            bool,
        ],
        Tuple[chex.Array, chex.Array, chex.Array],
    ]:
        def act_lprob(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            key: jrandom.PRNGKey,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, chex.Array]:
            backbone_act, h_state, backbone_updates = backbone.forward(
                params[CONST_BACKBONE],
                self.process_backbone_obs(obs),
                h_state,
                eval=True,
                **kwargs,
            )
            act_params, h_state, updates = model.forward(
                params[CONST_RESIDUAL], obs, h_state, eval=eval, **kwargs
            )
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = self._std_transform(act_raw_std) + self._min_std
            act = Normal.sample(act_mean, act_std, key)
            act_t = TanhTransform.transform(act)
            act_t = self.use_backbone_only * self.residual_impact * act_t + jax.nn.tanh(
                backbone_act
            )
            lprob = Normal.lprob(act_mean, act_std, act).sum(-1, keepdims=True)
            lprob = lprob - TanhTransform.log_abs_det_jacobian(act, act_t)
            return (
                act_t,
                lprob,
                h_state,
                updates,
            )

        return act_lprob

    def make_lprob(self, backbone: Model, model: Model) -> Callable[
        [Union[optax.Params, Dict[str, Any]], chex.Array, chex.Array, chex.Array, bool],
        chex.Array,
    ]:
        def lprob(
            params: Union[optax.Params, Dict[str, Any]],
            obs: chex.Array,
            h_state: chex.Array,
            act: chex.Array,
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, Dict[str, Any]]:
            backbone_act, h_state, backbone_updates = backbone.forward(
                params[CONST_BACKBONE],
                self.process_backbone_obs(obs),
                h_state,
                eval=True,
                **kwargs,
            )
            act_params, h_state, updates = model.forward(
                params[CONST_RESIDUAL], obs, h_state, eval=eval, **kwargs
            )
            act_mean, act_raw_std = jnp.split(act_params, 2, axis=-1)
            act_std = self._std_transform(act_raw_std) + self._min_std

            act_inv = TanhTransform.inv(
                (act - jax.nn.tanh(backbone_act)) / self.residual_impact
            )

            lprob = Normal.lprob(act_mean, act_std, act_inv).sum(-1, keepdims=True)
            lprob = lprob - TanhTransform.log_abs_det_jacobian(act_inv, act)
            return lprob, {
                CONST_MEAN: act_mean,
                CONST_STD: act_std,
                CONST_ENTROPY: Normal.entropy(act_inv, act_std),
                CONST_UPDATES: updates,
            }

        return lprob
