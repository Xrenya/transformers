# coding=utf-8
# Copyright 2023 Meta AI Research, FAIR The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Sam model. """


import math
import copy
import random
from typing import Optional, Tuple, List, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_sam import SamConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Xrenya/segment-anything-vit-b"
_CONFIG_FOR_DOC = "SamConfig"


SAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Xrenya/segment-anything-vit-b",
    # See all Sam models at https://huggingface.co/models?filter=sam
]


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_channels: int = 3,
        out_channels: int = 768,
    ) -> None:
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        # B C H W -> B H W C
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        return hidden_states


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean = hidden_states.mean(1, keep_dims=True)
        mu = (hidden_states - mean).pow(2).mean(1, keep_dims=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(mu + self.eps)
        hidden_states = self.weight[:, None, None] * hidden_states - self.bias[:, None, None]
        return hidden_states


def get_relative_position(q_size: int, k_size: int, relative_position: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        relative_position (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_relative_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolation relative position if needed
    if relative_position.shape[0] != max_relative_dist:
        # interpolate relative position
        relative_position_resized = F.interpolate(
            relative_position.reshape(1, relative_position.shape[0], -1).permute(0, 2, 1),
            size=max_relative_dist,
            mode="linear",
        )
        relative_position_resized = relative_position_resized.reshape(-1, max_relative_dist).permute(1, 0)
    else:
        relative_position_resized = relative_position

    # Scale coodinates with short length if shape for q and k are different
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)

    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return relative_position_resized[relative_coords.long()]


def add_decomposed_relative_position(
    attention_score: torch.Tensor,
    query: torch.Tensor,
    relative_position_h: torch.Tensor,
    relative_position_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attention_score (Tensor): attention map.
        query (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        relative_position_h (Tensor): relative position embeddings (Lh, C) for height axis.
        relative_position_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple(int, int)): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple(int, int)): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attention_score (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    relative_position_h = get_relative_position(q_h, k_h, relative_position_h)
    relative_position_w = get_relative_position(q_w, k_w, relative_position_w)

    batch_size, _, embed_dim = query.shape
    relative_query = query.rehspae(batch_size, q_h, q_w, embed_dim)
    relative_h = torch.einsum("bhwc, hkc -> bhwk", relative_query, relative_position_h)
    relative_w = torch.einsum("bhwc, hkc -> bhwk", relative_query, relative_position_w)

    attention_score = (
        attention_score.view(batch_size, q_h, q_w, k_h, k_w)
        + relative_h[:, :, :, :, None]
        + relative_w[:, :, :, None, :]
    ).view(batch_size, q_h * q_w, k_h * k_w)

    return attention_score


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""
    def __init__(
        self,
        config: SAMConfig,
        hidden_size: int,
        num_attention_heads: int = 8,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.proj = nn.Linear(self.all_head_size, self.all_head_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError(
                    "Input size must be provided if using relative positional encoding. "
                    "Expected resolution with height and width: 'Tuple[int, int]', "
                    "but got '{input_size}'"
                )
            self.relative_position_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.all_head_size))
            self.relative_position_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.all_head_size))

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, _ = x.shape
        # B H W C -> B HxW N E (C = NxE)
        new_x_shape = (batch_size, height * width, self.num_attention_heads, self.attention_head_size)
        # B HxW N E -> BxN HxW E
        x = x.view(*new_x_shape).permute(0, 2, 1, 3)
        return x.reshape(batch_size * self.num_attention_heads, height * width, -1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.key(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_score = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_score /= math.sqrt(self.attention_head_size)

        if self.use_rel_pos:
            attention_score = add_decomposed_relative_position(
                attention_score=attention_score,
                query=query_layer,
                relative_position_h=self.relative_position_h,
                relative_position_w=self.relative_position_w,
                q_size=(height, width),
                k_size=(height, width),
            )

        attention_score = attention_score.softmax(-1)
        hidden_states = torch.matmul(attention_score, value_layer) \
            .veiw(batch_size, self.num_attention_heads, height, width, -1) \
            .permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        hidden_states = self.proj(hidden_states)

        return hidden_states

class MLPBlock(nn.Module):
    def __init__(self, config, mlp_dim: int, act: Optional[str] = "gelu"):
        super().__init__()
        self.dense1 = nn.Linear(config.encoder_hidden_size, mlp_dim)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(mlp_dim, config.encoder_hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2
        return hidden_states


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
        Partition into non-overlapping windows with padding if needed.
        Args:
            x (tensor): input tokens with [B, H, W, C].
            window_size (int): window size.

        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C].
            (Hp, Wp): padded height and width before partition
    """
    batch_size, height, width, channels = x.shape

    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, 0, 0, pad_w, 0, pad_h)
    height_p, width_p = height + pad_h, width + pad_w

    x = x.view(batch_size, height_p // window_size, window_size, width_p // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
    return windows, (height_p, width_p)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    height_p, width_p = pad_hw
    height, width = hw
    batch_size = windows.shape[0] // (height_p * width_p // window_size // window_size)
    x = windows.view(batch_size, height_p // windows, width_p // windows, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height_p, width_p, -1)

    # remove padding
    if height_p > height or width_p > width:
        x = x[:, :height, :width_p, :].contiguous()

    return x




class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(self, config: SAMConfig, norm_layer: nn.Module = nn.LayerNorm, window_size: int = 0, input_size: Optional[Tuple[int, int]] = None,) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.layer_norm_1 = norm_layer(config.encoder_hidden_size)
        self.attention = Attention(
            config=config,
            hidden_size=config.encoder_hidden_size,
            num_attention_heads=config.num_attention_heads,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.layer_norm_2 = norm_layer(config.encoder_hidden_size)
        self.mlp = MLPBlock(
            embedding_dim=config.encoder_hidden_size,
            mlp_dim=int(config.encoder_hidden_size * config.mlp_ratio),
            act=config.act_layer
        )
        self.window_size = window_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        short_cut = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            x, padding = window_partition(hidden_states, self.window_size)

        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.window_size > 0:
            attention_output = window_unpartition(attention_output, self.window_size, padding, (height, width))

        hidden_states = short_cut + attention_output
        mlp_output = hidden_states + self.mlp(self.layer_norm_2(hidden_states))
        layer_output  = hidden_states + mlp_output

        outputs = (layer_output,) + outputs

        return outputs



class ImageEncoderViT(nn.Module):
    def __init__(self, config: SAMConfig, window_size: int) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = PatchEmbed(
            kernel_size=config.patch_size,
            stride=(config.patch_size, config.patch_size),
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None

        if config.use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, config.image_size // config.patch_size, config.image_size // config.patch_size, config.encoder_embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(config.encoder_layers):
            block = Block(
                config=config,
                window_size=config.window_size if i not in config.encoder_global_attention_indexes else 0,

            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                in_channels=config.encoder_embed_dim,
                out_channels=config.encoder_hidden_size,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(
                num_channels=config.encoder_hidden_size,
                eps=config.layer_norm_eps
            ),
            nn.Conv2d(
                in_channels=config.encoder_hidden_size,
                out_channels=config.encoder_hidden_size,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(
                num_channels=config.encoder_hidden_size,
                eps=config.layer_norm_eps
            ),
        )







class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats))
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        height, width = size
        grid = torch.ones(
            (height, width),
            device=self.positional_encoding_gaussian_matrix.device,
            dtype=self.positional_encoding_gaussian_matrix.dtype
        )
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed /= height
        x_embed /= width
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        # batch_size x height x width
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0, 1]"""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        # batch_size x N x C
        return self._pe_encoding(coords.to(self.positional_encoding_gaussian_matrix.dtype))



class PromtEncoder(nn.Module):
    def __init__(self, config: SAMConfig) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__inint__()
        self.embed_dim = config.prompt_embed_dim
        self.input_image_size = config.image_size if isinstance(Iterable, config.image_size) else (config.image_size, config.image_size)
        image_embedding_size = (config.image_size[0] // config.patch_size, config.image_size[1] // config.patch_size) if isinstance(Iterable, config.image_size) else (config.image_size // config.patch_size, config.image_size // config.patch_size)
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)

        # pos/neg point + 2 box corners
        self.num_point_embeddings: int = 4
        point_embeggins = [nn.Embedding(1, self.embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeggins)
        self.not_a_point_embed = nn.Embedding(1, self.embed_dim)

        self.mask_input_size = (4 * self.image_embedding_size[0], 4 * self.image_embedding_size[1])


        self.mask_downscalig = nn.Sequential(
            nn.Conv2d(1, config.mask_embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(config.mask_embed_dim // 4),
            ACT2FN[config.prompt_ecndoer_activation] if isinstance(config.prompt_ecndoer_activation, str) else config.prompt_ecndoer_activation,
            nn.Conv2d(config.mask_embed_dim // 4, config.mask_embed_dim, kernel_size=2, stride=2),
            LayerNorm2d(config.mask_embed_dim),
            ACT2FN[config.prompt_ecndoer_activation] if isinstance(config.prompt_ecndoer_activation, str) else config.prompt_ecndoer_activation,
            nn.Conv2d(config.mask_embed_dim, self.embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, self.embed_dim)

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weihts.device

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embed point prommpts."""
        points += 0.5 # shift to the central of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embed box prompts."""
        boxes += 0.5 # shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding


    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        batch_size = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((batch_size, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([
            nn.Linear(in_features=n, out_features=k) for n, k in zip([input_dim] + h, h + [output_dim])
        ])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = F.relu(layer(hidden_states)) if i < self.num_layers - 1 else layer(hidden_states)

        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class MaskDecoder(nn.Module):
    def __init__(self, config: SAMConfig, transformer: nn.Module) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = config.transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = config.num_multimask_outputs

        self.iou_token = nn.Embedding(1, self.transformer_dim)
        self.num_mask_tokens = self.num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.transformer_dim // 4),
            ACT2FN[config.mask_decoder_activation] if isinstance(config.mask_decoder_activation, str) else config.mask_decoder_activation,
            nn.ConvTranspose2d(self.transformer_dim // 4, self.transformer_dim // 8, kernel_size=2, stride=2),
            ACT2FN[config.mask_decoder_activation] if isinstance(config.mask_decoder_activation, str) else config.mask_decoder_activation,
        )

        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = MLP(
            self.transformer_dim, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_promt_embeddings: torch.Tensor,
        dense_prompt_embeggins: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # prepare output
        return masks, iou_pred


class SamPreTrainedModel(PreTrainedModel):
    config_class = SamConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (SamDecoder, SamEncoder)):
            module.gradient_checkpointing = value


SAM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config ([`~SamConfig`]):
            Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model
            weights.
"""

SAM_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
    >>> from transformers import SamTokenizer, SamForConditionalGeneration

    >>> model = SamForConditionalGeneration.from_pretrained('Xrenya/segment-anything-vit-b')
    >>> tokenizer = SamTokenizer.from_pretrained('Xrenya/segment-anything-vit-b')

    >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5)
    >>> print(tokenizer.decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```
"""

SAM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`~SamTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Provide for translation and summarization training. By default, the model will create this tensor by
            shifting the `input_ids` to the right, following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read [`modeling_sam._prepare_decoder_attention_mask`] and
            modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
            `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`,
            *optional*) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors
            of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape `(batch_size, 1)`
            instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds`
            have to be input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds`
            takes the value of `inputs_embeds`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up
            decoding (see `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


SAM_STANDALONE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`ProphetNetTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare Sam Model outputting raw hidden-states without any specific head on top.",
    SAM_START_DOCSTRING,
)
class SAMModel(SAMPreTrainedModel):
    def __init__(self, config: SAMConfig):
        super().__init__(config)
        self.config = config

        # Transformer image encoder and promt encoder
        self.image_encoder = ImageEncoderViT(config)
        self.prompt_encoder = PromptEncoder(config)
        self.image_features = None
        self.mask_decoder = MaskDecoder()

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SAM_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        points: Optional[torch.FloatTensor] = None,
        point_labels: Optional[torch.FloatTensor] = None,
        boxes: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # set image to get image features
        if pixel_values is None and self.image_features is None:
            raise ValueError(
                "An image must be feed once before mask prediction on it."
            )

        if pixel_values is not None and self.image_features is not None:
            logger.warning_once(
                "Do not need to input the same image for different prompts, "
                "the image features were already extracted."
            )

        if pixel_values:
            image_encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            self.image_features = image_encoder_outputs

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        sequence_output = (self.image_features[0], sparse_embeddings, dense_embeddings)

        if not return_dict:
            return (sequence_output,) + self.image_features[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=self.image_features.hidden_states,
            attentions=self.image_features.attentions,
        )

