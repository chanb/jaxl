from flax import linen as nn
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from jaxl.constants import *
from jaxl.models.common import Model
from jaxl.models.encodings import get_positional_encoding
from jaxl.models.modules import GPTModule, MLPModule, CNNModule, ResNetV1Module


def get_tokenizer(tokenizer_config: SimpleNamespace, embed_dim: int) -> Model:
    """
    Get tokenizer.

    :param tokenizer_config: the tokenizer configuration
    :param embed_dim: the embedding dimension
    :type tokenizer_config: SimpleNameSpace
    :type embed_dim: int
    :return: a tokenizer
    :rtype: Model
    """
    assert (
        tokenizer_config.type in VALID_TOKENIZER_TYPE
    ), f"{tokenizer_config.type} is not supported (one of {VALID_TOKENIZER_TYPE})"

    tokenizer_kwargs = tokenizer_config.kwargs
    if tokenizer_config.type == CONST_MLP:
        return MLPModule(
            tokenizer_kwargs.layers + [embed_dim],
            getattr(tokenizer_kwargs, "activation", CONST_RELU),
            getattr(tokenizer_kwargs, "output_activation", CONST_IDENTITY),
            getattr(tokenizer_kwargs, "use_batch_norm", False),
            getattr(tokenizer_kwargs, "use_bias", False),
        )
    elif tokenizer_config.type == CONST_CNN:
        return CNNModule(
            tokenizer_kwargs.features,
            tokenizer_kwargs.kernel_sizes,
            getattr(tokenizer_kwargs, "activation", CONST_RELU),
            getattr(tokenizer_kwargs, "use_batch_norm", False),
        )
    elif tokenizer_config.type == CONST_RESNET:
        return ResNetV1Module(
            tokenizer_kwargs.blocks_per_group,
            tokenizer_kwargs.features,
            tokenizer_kwargs.stride,
            tokenizer_kwargs.use_projection,
            tokenizer_kwargs.use_bottleneck,
            getattr(tokenizer_kwargs, "use_batch_norm", True),
        )
    else:
        raise ValueError(
            f"{tokenizer_config.type} is not supported (one of {VALID_TOKENIZER_TYPE})"
        )


class InContextSupervisedTransformer(Model):
    """A GPT for in-context learning with customized tokenizers."""

    def __init__(
        self,
        output_dim: int,
        num_contexts: int,
        num_blocks: int,
        num_heads: int,
        embed_dim: int,
        widening_factor: int,
        positional_encoding: SimpleNamespace,
        input_tokenizer_config: SimpleNamespace,
        output_tokenizer_config: SimpleNamespace,
        query_pred_only: bool = False,
        input_output_same_encoding: bool = True,
    ):
        self.gpt = GPTModule(
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
            widening_factor=widening_factor,
        )
        self.input_tokenizer = get_tokenizer(input_tokenizer_config, embed_dim)
        self.output_tokenizer = get_tokenizer(output_tokenizer_config, embed_dim)
        self.predictor = nn.Dense(int(np.product(output_dim)))
        self.positional_encoding = get_positional_encoding(positional_encoding)
        self.num_tokens = num_contexts * 2 + 1
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.apply_positional_encoding = self._make_get_positional_encoding(
            input_output_same_encoding
        )
        self.tokenize = jax.jit(self.make_tokenize(), static_argnames=[CONST_EVAL])
        self.get_latent = jax.jit(self.make_get_latent(), static_argnames=[CONST_EVAL])
        self.forward = jax.jit(
            self.make_forward(query_pred_only), static_argnames=[CONST_EVAL]
        )

    def _make_get_positional_encoding(
        self, input_output_same_encoding: bool
    ) -> Callable:
        if input_output_same_encoding:

            def apply_positional_encoding(
                params, queries, input_embedding, context_output_embedding, **kwargs
            ):
                # Treat input-output pair with same position
                input_embedding = self.positional_encoding.apply(
                    params[CONST_POSITIONAL_ENCODING],
                    input_embedding,
                    **kwargs,
                )

                context_input_embedding, query_embedding = (
                    input_embedding[:, :-1],
                    input_embedding[:, -1:],
                )
                context_output_embedding = self.positional_encoding.apply(
                    params[CONST_POSITIONAL_ENCODING],
                    context_output_embedding,
                    **kwargs,
                )

                stacked_inputs = jnp.concatenate(
                    (context_input_embedding, context_output_embedding), axis=-1
                ).reshape((len(queries), -1, self.embed_dim))

                stacked_inputs = jnp.concatenate(
                    (stacked_inputs, query_embedding), axis=1
                )
                return stacked_inputs

        else:

            def apply_positional_encoding(
                params, queries, input_embedding, context_output_embedding, **kwargs
            ):
                # Treat each token separately position
                context_input_embedding, query_embedding = (
                    input_embedding[:, :-1],
                    input_embedding[:, -1:],
                )
                stacked_inputs = jnp.concatenate(
                    (
                        context_input_embedding,
                        context_output_embedding,
                    ),
                    axis=-1,
                ).reshape((len(queries), -1, self.embed_dim))
                stacked_inputs = jnp.concatenate(
                    (
                        stacked_inputs,
                        query_embedding,
                    ),
                    axis=1,
                )
                stacked_inputs = self.positional_encoding.apply(
                    params[CONST_POSITIONAL_ENCODING],
                    stacked_inputs,
                    **kwargs,
                )
                return stacked_inputs

        return apply_positional_encoding

    def init(
        self,
        model_key: jrandom.PRNGKey,
        dummy_input: chex.Array,
        dummy_output: chex.Array,
    ) -> Dict[str, Any]:
        """
        Initialize model train states.

        :param model_key: the random number generation key for initializing train states
        :param dummy_input: the input data
        :param dummy_output: the output data
        :type model_key: jrandom.PRNGKey
        :type dummy_input: chex.Array
        :type dummy_output: chex.Array
        :return: the initialized train states
        :rtype: Dict[str, Any]

        """
        input_key, output_key, gpt_key, predictor_key, pe_key = jrandom.split(
            model_key, 5
        )

        dummy_token = np.zeros((1, 1, self.embed_dim))
        dummy_repr = np.zeros((1, 1, self.embed_dim))

        it_params = self.input_tokenizer.init(input_key, dummy_input)
        ot_params = self.output_tokenizer.init(output_key, dummy_output)
        gpt_params = self.gpt.init(gpt_key, dummy_token, eval=True)
        pe_params = self.positional_encoding.init(pe_key, dummy_token)
        predictor_params = self.predictor.init(predictor_key, dummy_repr)
        return {
            CONST_PARAMS: {
                CONST_INPUT_TOKENIZER: it_params,
                CONST_OUTPUT_TOKENIZER: ot_params,
                CONST_GPT: gpt_params,
                CONST_POSITIONAL_ENCODING: pe_params,
                CONST_PREDICTOR: predictor_params,
            },
            CONST_RANDOM_KEYS: {},  # TODO: Should probably have something to deal with random_keys
        }

    def make_tokenize(
        self,
    ) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        """
        Makes the tokenize call of the ICL model.

        :return: the tokenize call.
        :rtype: Callable[
            [
                Dict[str, Any],
                chex.Array,
                chex.Array,
                chex.Array,
                bool,
            ],
            Tuple[chex.Array, chex.Array, Any],
        ]
        """

        def tokenize(
            train_states: Dict[str, Any],
            queries: chex.Array,
            contexts: Dict[str, chex.Array],
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            input_embedding, input_updates = self.input_tokenizer.apply(
                train_states[CONST_PARAMS][CONST_INPUT_TOKENIZER],
                jnp.concatenate(
                    (
                        contexts[CONST_CONTEXT_INPUT],
                        queries,
                    ),
                    axis=1,
                ),
                eval,
                mutable=[CONST_BATCH_STATS],
            )
            context_output_embedding, output_updates = self.output_tokenizer.apply(
                train_states[CONST_PARAMS][CONST_OUTPUT_TOKENIZER],
                contexts[CONST_CONTEXT_OUTPUT],
                eval,
                mutable=[CONST_BATCH_STATS],
            )

            stacked_inputs = self.apply_positional_encoding(
                train_states,
                queries,
                input_embedding,
                context_output_embedding,
                **kwargs,
            )

            return (
                stacked_inputs,
                None,
                {
                    CONST_INPUT_TOKENIZER: input_updates,
                    CONST_OUTPUT_TOKENIZER: output_updates,
                },
            )

        return tokenize

    def make_get_latent(
        self,
    ) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        """
        Makes the get latent call of the ICL model.

        :return: the get latent call.
        :rtype: Callable[
            [
                Dict[str, Any],
                chex.Array,
                chex.Array,
                chex.Array,
                bool,
            ],
            Tuple[chex.Array, chex.Array, Any],
        ]
        """

        def get_latent(
            train_states: Dict[str, Any],
            queries: chex.Array,
            contexts: Dict[str, chex.Array],
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            stacked_inputs, _, token_updates = self.tokenize(
                train_states, queries, contexts, eval, **kwargs
            )
            repr = self.gpt.apply(
                train_states[CONST_PARAMS][CONST_GPT],
                stacked_inputs,
                eval,
            )

            return repr, None, token_updates

        return get_latent

    def make_forward(self, query_pred_only: bool) -> Callable[
        [
            Dict[str, Any],
            chex.Array,
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        if query_pred_only:

            def process_prediction(preds):
                return preds[:, -1]

        else:

            def process_prediction(preds):
                return preds[:, ::2]

        def forward(
            train_states: Dict[str, Any],
            queries: chex.Array,
            contexts: Dict[str, chex.Array],
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            repr, carry, latent_updates = self.get_latent(
                train_states, queries, contexts, eval, **kwargs
            )
            outputs = self.predictor.apply(
                train_states[CONST_PARAMS][CONST_PREDICTOR],
                repr,
            )

            return process_prediction(outputs), carry, latent_updates

        return forward

    def update_batch_stats(
        self, params: Dict[str, Any], batch_stats: Any
    ) -> Dict[str, Any]:
        if CONST_BATCH_STATS in params[CONST_INPUT_TOKENIZER]:
            params[CONST_INPUT_TOKENIZER][CONST_BATCH_STATS] = batch_stats[
                CONST_INPUT_TOKENIZER
            ]

        if CONST_BATCH_STATS in params[CONST_OUTPUT_TOKENIZER]:
            params[CONST_OUTPUT_TOKENIZER][CONST_BATCH_STATS] = batch_stats[
                CONST_OUTPUT_TOKENIZER
            ]

        return params
