from flax import linen as nn
from flax.linen.initializers import zeros
from typing import Callable, Sequence, Any, Dict

import chex
import jax
import jax.numpy as jnp
import math

from jaxl.constants import CONST_SAME_PADDING


class MLPModule(nn.Module):
    """Multilayer Perceptron."""

    # The number of hidden units in each hidden layer.
    layers: Sequence[int]
    activation: Callable
    output_activation: Callable
    use_batch_norm: bool
    use_bias: bool

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        idx = -1
        for idx, layer in enumerate(self.layers[:-1]):
            x = self.activation(nn.Dense(layer)(x))
            if self.use_batch_norm:
                x = nn.BatchNorm(
                    momentum=0.9,
                    epsilon=1e-5,
                    use_bias=self.use_bias,
                    use_scale=True,
                )(x, eval)
            # self.sow("mlp_latents", "mlp_{}".format(idx), x)
        x = self.output_activation(nn.Dense(self.layers[-1], use_bias=self.use_bias)(x))
        # self.sow("mlp_latents", "mlp_{}".format(idx + 1), x)
        return x


class CNNModule(nn.Module):
    """Convolutional layer."""

    # The number of kernels/filters per layer
    features: Sequence[int]

    # The kernel/filter size per layer
    kernel_sizes: Sequence[Sequence[int]]

    # The activation to use after convolutional layer
    activation: Callable
    use_batch_norm: bool

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        for idx, (feature, kernel_size) in enumerate(
            zip(self.features, self.kernel_sizes)
        ):
            x = self.activation(nn.Conv(feature, kernel_size)(x))
            if self.use_batch_norm:
                x = nn.BatchNorm(
                    momentum=0.9,
                    epsilon=1e-5,
                    use_bias=True,
                    use_scale=True,
                )(x, eval)
            # self.sow("cnn_latents", "cnn_{}".format(idx), x)
        return x


class SelfAttentionModule(nn.Module):
    """Self-Attention layer."""
    num_heads: int
    qkv_features: int = None
    
    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, mask=None, **kwargs) -> chex.Array:
        in_dim = x.shape[-1]

        if self.qkv_features is not None:
            qkv_hiddens = self.qkv_features
        else:
            qkv_hiddens = in_dim

        q = nn.Dense(qkv_hiddens)(x)
        k = nn.Dense(qkv_hiddens)(x)
        v = nn.Dense(qkv_hiddens)(x)

        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape
        head_dim = qkv_hiddens // self.num_heads
        q = jnp.reshape(q, [batch, q_time, self.num_heads, head_dim])
        k = jnp.reshape(k, [batch, kv_time, self.num_heads, head_dim])
        v = jnp.reshape(v, [batch, kv_time, self.num_heads, head_dim])

        # attend
        hiddens = self.num_heads * head_dim
        scale = 1. / math.sqrt(head_dim)
        attention = jnp.einsum('bthd,bThd->bhtT', q, k)
        attention *= scale
        if mask is not None:
            attention = attention * mask - 1e10 * (1 - mask)
        normalized = jax.nn.softmax(attention)
        summed = jnp.einsum('bhtT,bThd->bthd', normalized, v)
        out = jnp.reshape(summed, [batch, q_time, hiddens])

        return nn.Dense(qkv_hiddens)(out)



class ResNetV1Block(nn.Module):
    """
    ResNet V1 Block.
    Reference: https://github.com/google-deepmind/emergent_in_context_learning/blob/eba75a4208b8927cc1e981384a2cc7e014677095/modules/resnet.py
    """

    # The number of kernels/filters
    features: int

    # The strides of the convolution
    stride: Sequence[int]

    # Whether or not to bottleneck
    use_projection: bool

    # Whether or not to bottleneck
    use_bottleneck: bool

    use_batch_norm: bool

    def setup(self):
        assert (
            not self.use_bottleneck or self.features >= 4 and self.features % 4 == 0
        ), "must have at least 4n kernels {} when using bottleneck".format(
            self.features
        )
        if self.use_projection:
            self.projection = nn.Conv(
                self.features,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                padding=CONST_SAME_PADDING,
            )
            if self.use_batch_norm:
                self.projection_batchnorm = nn.BatchNorm(
                    momentum=0.9,
                    epsilon=1e-5,
                    use_bias=True,
                    use_scale=True,
                )

        conv_features = self.features
        conv_0_kernel = (3, 3)
        conv_0_stride = self.stride
        conv_1_stride = 1
        if self.use_bottleneck:
            conv_features = self.features // 4
            conv_0_kernel = (1, 1)
            conv_0_stride = 1
            conv_1_stride = self.stride

        self.conv_0 = nn.Conv(
            conv_features,
            kernel_size=conv_0_kernel,
            strides=conv_0_stride,
            use_bias=False,
            padding=CONST_SAME_PADDING,
        )

        if self.use_batch_norm:
            self.batch_norm_0 = nn.BatchNorm(
                momentum=0.9,
                epsilon=1e-5,
                use_bias=True,
                use_scale=True,
            )

        self.conv_1 = nn.Conv(
            conv_features,
            kernel_size=(3, 3),
            strides=conv_1_stride,
            use_bias=False,
            padding=CONST_SAME_PADDING,
        )

        if self.use_batch_norm:
            self.batch_norm_1 = nn.BatchNorm(
                momentum=0.9,
                epsilon=1e-5,
                use_bias=True,
                use_scale=True,
            )

        if self.use_batch_norm:
            layers = [
                (self.conv_0, self.batch_norm_0),
                (self.conv_1, self.batch_norm_1),
            ]
        else:
            layers = [self.conv_0, self.conv_1]

        if self.use_bottleneck:
            self.conv_2 = nn.Conv(
                self.features,
                kernel_size=(1, 1),
                strides=1,
                use_bias=False,
                padding=CONST_SAME_PADDING,
            )
            if self.use_batch_norm:
                self.batch_norm_2 = nn.BatchNorm(
                    momentum=0.9,
                    epsilon=1e-5,
                    use_bias=True,
                    use_scale=True,
                    scale_init=zeros,
                )
                layers.append((self.conv_2, self.batch_norm_2))
            else:
                layers.append(self.conv_2)
        self.layers = layers

    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        out = shortcut = x

        if self.use_projection:
            shortcut = self.projection(shortcut)
            if self.use_batch_norm:
                shortcut = self.projection_batchnorm(shortcut, eval)
            # self.sow("resnet_v1", "resnet_v1_projection", shortcut)

        idx = -1

        if self.use_batch_norm:
            for idx, (conv_i, batch_norm_i) in enumerate(self.layers[:-1]):
                out = conv_i(out)
                out = batch_norm_i(out, eval)
                out = jax.nn.relu(out)
                # self.sow("resnet_v1", "resnet_v1_{}".format(idx), out)
            out = self.layers[-1][0](out)
            out = self.layers[-1][1](out, eval)
        else:
            for idx, conv_i in enumerate(self.layers[:-1]):
                out = conv_i(out)
                out = jax.nn.relu(out)
                # self.sow("resnet_v1", "resnet_v1_{}".format(idx), out)
            out = self.layers[-1](out)

        out = jax.nn.relu(out + shortcut)
        # self.sow("resnet_v1_latents", "resnet_v1_{}".format(idx + 1), out)
        return out


class ResNetV1BlockGroup(nn.Module):
    # THe number of residual blocks
    num_blocks: int

    # The number of kernels/filters
    features: int

    # The strides of the convolution
    stride: Sequence[int]

    # Whether or not to bottleneck
    use_projection: bool

    # Whether or not to bottleneck
    use_bottleneck: bool

    use_batch_norm: bool

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        for block_i in range(self.num_blocks):
            x = ResNetV1Block(
                self.features,
                1 if block_i else self.stride,
                block_i == 0 and self.use_projection,
                self.use_bottleneck,
                self.use_batch_norm,
            )(x, eval)
            # self.sow(
            #     "resnet_v1_block_group_latents", "resnet_v1_{}".format(block_i + 1), x
            # )
        return x


class ResNetV1Module(nn.Module):
    # The number of residual blocks per block group
    blocks_per_group: Sequence[int]

    # The number of kernels/filters per layer
    features: Sequence[int]

    # The strides of the convolution per layer
    stride: Sequence[Sequence[int]]

    # Whether or not to bottleneck
    use_projection: Sequence[bool]

    # Whether or not to bottleneck
    use_bottleneck: bool

    use_batch_norm: bool

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=2,
            use_bias=False,
            padding=CONST_SAME_PADDING,
        )(x)

        if self.use_batch_norm:
            x = nn.BatchNorm(
                momentum=0.9,
                epsilon=1e-5,
                use_bias=True,
                use_scale=True,
            )(x, eval)
        x = jax.nn.relu(x)
        x = nn.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding=CONST_SAME_PADDING,
        )
        # self.sow("resnet_v1_module", "resnet_v1_before_blocks", x)

        for (
            curr_blocks,
            curr_features,
            curr_stride,
            curr_projection,
        ) in zip(
            self.blocks_per_group, self.features, self.stride, self.use_projection
        ):
            x = ResNetV1BlockGroup(
                curr_blocks,
                curr_features,
                curr_stride,
                curr_projection,
                self.use_bottleneck,
                self.use_batch_norm,
            )(x, eval)
        return jnp.mean(x, axis=(-3, -2))


class GPTBlock(nn.Module):
    """GPT Block."""

    # : The number of attention heads
    num_heads: int

    # : The embedding dimensionality
    embed_dim: int

    # : Widening dimension
    widening_factor: int

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        mask = nn.make_causal_mask(x[..., 0])
        # x = x + nn.MultiHeadDotProductAttention(
        #     self.num_heads, qkv_features=self.embed_dim
        # )(nn.LayerNorm()(x), mask=mask)
        x = x + SelfAttentionModule(
            self.num_heads, self.embed_dim
        )(nn.LayerNorm()(x), mask=mask)
        normed_x = nn.gelu(
            nn.Dense(self.embed_dim * self.widening_factor)(nn.LayerNorm()(x))
        )
        x = x + nn.Dense(self.embed_dim)(normed_x)
        return x


class GPTModule(nn.Module):
    """GPT."""

    # : The number of GPT Blocks
    num_blocks: int

    # : The number of attention heads
    num_heads: int

    # : The embedding dimensionality
    embed_dim: int

    # : Widening dimension
    widening_factor: int

    @nn.compact
    def __call__(self, x: chex.Array, eval: bool, **kwargs) -> chex.Array:
        # jax.debug.print("result={x}", x=x[0])
        for idx, _ in enumerate(range(self.num_blocks)):
            x = GPTBlock(self.num_heads, self.embed_dim, self.widening_factor)(x, eval)
            # self.sow("gpt_latents", "gpt_{}".format(idx), x)
        x = nn.LayerNorm()(x)
        # self.sow("gpt_latents", "gpt_{}".format(idx + 1), x)
        return x


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self, **kwargs) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda _: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)

    def update_batch_stats(self, params, batch_stats):
        return params
