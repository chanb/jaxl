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
from jaxl.models.common import (
    Model,
    CNN,
    MLP,
    ResNetV1,
)
from jaxl.models.encodings import get_positional_encoding
from jaxl.models.modules import GPTModule


class InContextSupervisedTransformer(Model):
    """A GPT for in-context learning."""

    def __init__(
        self,
        output_dim: int,
        num_contexts: int,
        num_blocks: int,
        num_heads: int,
        embed_dim: int,
        positional_encoding: SimpleNamespace,
        query_pred_only: bool = False,
        input_output_same_encoding: bool = True,
    ) -> None:
        self.gpt = GPTModule(
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
        )
        self.input_tokenizer = nn.Dense(embed_dim)
        self.output_tokenizer = nn.Dense(embed_dim)
        self.predictor = nn.Dense(int(np.product(output_dim)))
        self.positional_encoding = get_positional_encoding(positional_encoding)
        self.num_tokens = num_contexts * 2 + 1
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.apply_positional_encoding = self._make_get_positional_encoding(
            input_output_same_encoding
        )
        self.tokenize = jax.jit(self.make_tokenize())
        self.get_latent = jax.jit(self.make_get_latent())
        self.forward = jax.jit(self.make_forward(query_pred_only))

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
    ) -> Union[optax.Params, Dict[str, Any]]:
        """
        Initialize model parameters.

        :param model_key: the random number generation key for initializing parameters
        :param dummy_input: the input data
        :param dummy_output: the output data
        :type model_key: jrandom.PRNGKey
        :type dummy_input: chex.Array
        :type dummy_output: chex.Array
        :return: the initialized parameters
        :rtype: Union[optax.Params, Dict[str, Any]]

        """
        input_key, output_key, gpt_key, predictor_key, pe_key = jrandom.split(
            model_key, 5
        )
        dummy_token = np.zeros((1, 1, self.embed_dim))
        dummy_repr = np.zeros((1, 1, self.embed_dim))
        return {
            CONST_INPUT_TOKENIZER: self.input_tokenizer.init(input_key, dummy_input),
            CONST_OUTPUT_TOKENIZER: self.output_tokenizer.init(
                output_key, dummy_output
            ),
            CONST_GPT: self.gpt.init(gpt_key, dummy_token, eval=True),
            CONST_POSITIONAL_ENCODING: self.positional_encoding.init(
                pe_key, dummy_token
            ),
            CONST_PREDICTOR: self.predictor.init(predictor_key, dummy_repr),
        }

    def make_tokenize(
        self,
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            chex.Array,
        ],
        Tuple[chex.Array, chex.Array],
    ]:
        """
        Makes the tokenize call of the ICL model.

        :return: the tokenize call.
        :rtype: Callable[
            [
                Union[optax.Params, Dict[str, Any]],
                chex.Array,
                chex.Array,
                chex.Array,
            ],
            Tuple[chex.Array, chex.Array],
        ]
        """

        def tokenize(
            params: Union[optax.Params, Dict[str, Any]],
            queries: chex.Array,
            contexts: Dict[str, chex.Array],
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array]:
            """
            Get latent call of the GPT.

            :param params: the model parameters
            :param queries: the queries
            :param contexts: the context with keys `context_input` and `context_output`
            :type params: Union[optax.Params, Dict[str, Any]]
            :type queries: chex.Array
            :type contexts: Dict[str, chex.Array]
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array]

            """
            num_samples, context_len = contexts[CONST_CONTEXT_INPUT].shape[:2]
            input_dim = contexts[CONST_CONTEXT_INPUT].shape[2:]
            output_dim = contexts[CONST_CONTEXT_OUTPUT].shape[2:]
            batch_size = num_samples * context_len

            input_embedding, input_updates = self.input_tokenizer.apply(
                params[CONST_INPUT_TOKENIZER],
                jnp.concatenate(
                    (
                        contexts[CONST_CONTEXT_INPUT],
                        queries,
                    ),
                    axis=1,
                ).reshape((batch_size + num_samples, *input_dim)),
                None,
                eval,
                mutable=[CONST_BATCH_STATS],
            )

            input_embedding = input_embedding.reshape(
                (num_samples, context_len + 1, -1)
            )

            context_output_embedding, output_updates = self.output_tokenizer.apply(
                params[CONST_OUTPUT_TOKENIZER],
                contexts[CONST_CONTEXT_OUTPUT],
                eval,
                mutable=[CONST_BATCH_STATS],
            )

            stacked_inputs = self.apply_positional_encoding(
                params, queries, input_embedding, context_output_embedding, **kwargs
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
            Union[optax.Params, Dict[str, Any]],
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
                Union[optax.Params, Dict[str, Any]],
                chex.Array,
                chex.Array,
                chex.Array,
                bool,
            ],
            Tuple[chex.Array, chex.Array, Any],
        ]
        """

        def get_latent(
            params: Union[optax.Params, Dict[str, Any]],
            queries: chex.Array,
            contexts: Dict[str, chex.Array],
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            """
            Get latent call of the GPT.

            :param params: the model parameters
            :param queries: the queries
            :param contexts: the context with keys `context_input` and `context_output`
            :type params: Union[optax.Params, Dict[str, Any]]
            :type queries: chex.Array
            :type contexts: Dict[str, chex.Array]
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array, Any]

            """
            stacked_inputs, _, token_updates = self.tokenize(
                params, queries, contexts, **kwargs
            )
            (repr, gpt_updates) = self.gpt.apply(
                params[CONST_GPT],
                stacked_inputs,
                eval,
                mutable=[CONST_BATCH_STATS],
                **kwargs,
            )

            return repr, None, {**token_updates, CONST_GPT: gpt_updates}

        return get_latent

    def make_forward(self, query_pred_only: bool) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
            chex.Array,
            chex.Array,
            chex.Array,
            bool,
        ],
        Tuple[chex.Array, chex.Array, Any],
    ]:
        """
        Makes the forward call of the ICL model.

        :param query_pred_only: whether or not to output the query prediciton only
        :type query_pred_only: bool
        :return: the forward call.
        :rtype: Callable[
            [
                Union[optax.Params, Dict[str, Any]],
                chex.Array,
                chex.Array,
                chex.Array,
                bool,
            ],
            Tuple[chex.Array, chex.Array, Any],
        ]
        """

        if query_pred_only:

            def process_prediction(preds):
                return preds[:, -1]

        else:

            def process_prediction(preds):
                return preds[:, ::2]

        def forward(
            params: Union[optax.Params, Dict[str, Any]],
            queries: chex.Array,
            contexts: Dict[str, chex.Array],
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            """
            Forward call of the GPT.

            :param params: the model parameters
            :param queries: the queries
            :param contexts: the context with keys `context_input` and `context_output`
            :type params: Union[optax.Params, Dict[str, Any]]
            :type queries: chex.Array
            :type contexts: Dict[str, chex.Array]
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array, Any]

            """
            repr, carry, latent_updates = self.get_latent(
                params, queries, contexts, **kwargs
            )
            outputs = self.predictor.apply(
                params[CONST_PREDICTOR],
                repr,
            )

            return process_prediction(outputs), carry, latent_updates

        return forward

    def update_batch_stats(
        self, params: Dict[str, Any], batch_stats: Any
    ) -> Dict[str, Any]:
        params[CONST_INPUT_TOKENIZER] = self.input_tokenizer.update_batch_stats(
            params[CONST_INPUT_TOKENIZER],
            batch_stats[CONST_INPUT_TOKENIZER],
        )
        params[CONST_OUTPUT_TOKENIZER] = self.output_tokenizer.update_batch_stats(
            params[CONST_OUTPUT_TOKENIZER],
            batch_stats[CONST_OUTPUT_TOKENIZER],
        )
        params[CONST_GPT] = self.output_tokenizer.update_batch_stats(
            params[CONST_GPT],
            batch_stats[CONST_GPT],
        )
        return params


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
        return MLP(
            layers=tokenizer_kwargs.layers + [embed_dim],
            activation=getattr(tokenizer_kwargs, "activation", CONST_RELU),
            output_activation=getattr(
                tokenizer_kwargs, "output_activation", CONST_IDENTITY
            ),
            use_batch_norm=getattr(tokenizer_kwargs, "use_batch_norm", False),
        )
    elif tokenizer_config.type == CONST_CNN:
        return CNN(
            features=tokenizer_kwargs.features,
            kernel_sizes=tokenizer_kwargs.kernel_sizes,
            layers=tokenizer_kwargs.layers + [embed_dim],
            activation=getattr(tokenizer_kwargs, "activation", CONST_RELU),
            output_activation=getattr(
                tokenizer_kwargs, "output_activation", CONST_IDENTITY
            ),
            use_batch_norm=getattr(tokenizer_kwargs, "use_batch_norm", False),
        )
    elif tokenizer_config.type == CONST_RESNET:
        return ResNetV1(
            blocks_per_group=tokenizer_kwargs.blocks_per_group,
            features=tokenizer_kwargs.features,
            stride=tokenizer_kwargs.stride,
            use_projection=tokenizer_kwargs.use_projection,
            use_bottleneck=tokenizer_kwargs.use_bottleneck,
            use_batch_norm=getattr(tokenizer_kwargs, "use_batch_norm", True),
        )
    else:
        raise ValueError(
            f"{tokenizer_config.type} is not supported (one of {VALID_TOKENIZER_TYPE})"
        )


class CustomTokenizerICSupervisedTransformer(InContextSupervisedTransformer):
    """A GPT for in-context learning with customized tokenizers."""

    def __init__(
        self,
        output_dim: int,
        num_contexts: int,
        num_blocks: int,
        num_heads: int,
        embed_dim: int,
        positional_encoding: SimpleNamespace,
        input_tokenizer_config: SimpleNamespace,
        output_tokenizer_config: SimpleNamespace,
        query_pred_only: bool = False,
        input_output_same_encoding: bool = True,
    ) -> None:
        self.gpt = GPTModule(
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_dim=embed_dim,
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

    def make_tokenize(
        self,
    ) -> Callable[
        [
            Union[optax.Params, Dict[str, Any]],
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
                Union[optax.Params, Dict[str, Any]],
                chex.Array,
                chex.Array,
                chex.Array,
                bool,
            ],
            Tuple[chex.Array, chex.Array, Any],
        ]
        """

        def tokenize(
            params: Union[optax.Params, Dict[str, Any]],
            queries: chex.Array,
            contexts: Dict[str, chex.Array],
            eval: bool = False,
            **kwargs,
        ) -> Tuple[chex.Array, chex.Array, Any]:
            """
            Get latent call of the GPT.

            :param params: the model parameters
            :param queries: the queries
            :param contexts: the context with keys `context_input` and `context_output`
            :type params: Union[optax.Params, Dict[str, Any]]
            :type queries: chex.Array
            :type contexts: Dict[str, chex.Array]
            :return: the output and a pass-through carry
            :rtype: Tuple[chex.Array, chex.Array, Any]

            """
            num_samples, context_len = contexts[CONST_CONTEXT_INPUT].shape[:2]
            input_dim = contexts[CONST_CONTEXT_INPUT].shape[2:]
            output_dim = contexts[CONST_CONTEXT_OUTPUT].shape[2:]
            batch_size = num_samples * context_len

            input_embedding, _, input_updates = self.input_tokenizer.forward(
                params[CONST_INPUT_TOKENIZER],
                jnp.concatenate(
                    (
                        contexts[CONST_CONTEXT_INPUT],
                        queries,
                    ),
                    axis=1,
                ).reshape((batch_size + num_samples, *input_dim)),
                None,
                eval,
                **kwargs,
            )
            context_output_embedding, _, output_updates = self.output_tokenizer.forward(
                params[CONST_OUTPUT_TOKENIZER],
                contexts[CONST_CONTEXT_OUTPUT].reshape((batch_size, *output_dim)),
                None,
                eval,
                **kwargs,
            )

            input_embedding = input_embedding.reshape(
                (num_samples, context_len + 1, -1)
            )
            context_output_embedding = context_output_embedding.reshape(
                (num_samples, context_len, -1)
            )

            stacked_inputs = self.apply_positional_encoding(
                params, queries, input_embedding, context_output_embedding, **kwargs
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
