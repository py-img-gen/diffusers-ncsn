import math
from typing import Optional, Tuple, Union

import torch
from diffusers import UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class UNet2DModelForNCSN(UNet2DModel, ModelMixin, ConfigMixin):  # type: ignore[misc]
    @register_to_config
    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        num_train_timesteps: int,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
    ) -> None:
        super().__init__(
            sample_size,
            in_channels,
            out_channels,
            center_input_sample,
            time_embedding_type,
            time_embedding_dim,
            freq_shift,
            flip_sin_to_cos,
            down_block_types,
            up_block_types,
            block_out_channels,
            layers_per_block,
            mid_block_scale_factor,
            downsample_padding,
            downsample_type,
            upsample_type,
            dropout,
            act_fn,
            attention_head_dim,
            norm_num_groups,
            attn_norm_num_groups,
            norm_eps,
            resnet_time_scale_shift,
            add_attention,
            class_embed_type,
            num_class_embeds,
            num_train_timesteps,
        )
        sigmas = torch.exp(
            torch.linspace(
                start=math.log(sigma_max),
                end=math.log(sigma_min),
                steps=num_train_timesteps,
            )
        )
        self.register_buffer("sigmas", sigmas)  # type: ignore
