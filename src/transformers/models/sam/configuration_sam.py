# coding=utf-8
# Copyright 2023 Meta AI Research, FAIR and The HuggingFace Inc. team. All rights reserved.
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
""" Sam model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SAM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Xrenya/segment-anything-vit-b": "https://huggingface.co/Xrenya/segment-anything-vit-b/resolve/main/config.json",
    # See all Sam models at https://huggingface.co/models?filter=sam
}


class SamConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~SamModel`].
    It is used to instantiate an Sam model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the Sam [Xrenya/segment-anything-vit-b](https://huggingface.co/Xrenya/segment-anything-vit-b) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the Sam model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~SamModel`] or
            [`~TFSamModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimension of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see
            https://arxiv.org/abs/1909.11556) for more details.
        decoder_layerdrop: (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see
            https://arxiv.org/abs/1909.11556) for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        Example:

    ```python
    >>> from transformers import SamModel, SamConfig

    >>> # Initializing a Sam Xrenya/segment-anything-vit-b style configuration
    >>> configuration = SamConfig()

    >>> # Initializing a model from the Xrenya/segment-anything-vit-b style configuration
    >>> model = SamModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "sam"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        image_size: int = 1024,
        num_channels: int = 3,
        patch_size: int = 16,
        encoder_embed_dim: int = 4096,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 12,
        encoder_global_attention_indexes: List[int] = [2, 5, 8, 11],

        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,

        encoder_hidden_size: int = 256,
        qkv_bias: bool = True,
        encoder_act_layer: str = "gelu",
        layer_norm_eps: float = 1e-06,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,

        prompt_ecndoer_activation: str = "gelu",
        prompt_embed_dim: int = 256,
        mask_embed_dim: int = 16,

        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        mask_decoder_activation: str = "gelu",
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        init_std: float = 0.02,
        **kwargs
    ):
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_global_attention_indexes = encoder_global_attention_indexes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.encoder_hidden_size = encoder_hidden_size
        self.qkv_bias = qkv_bias
        self.encoder_act_layer = encoder_act_layer
        self.layer_norm_eps = layer_norm_eps
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size

        self.prompt_ecndoer_activation = prompt_ecndoer_activation
        self.prompt_embed_dim = prompt_embed_dim
        self.mask_embed_dim = mask_embed_dim

        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.mask_decoder_activation = mask_decoder_activation
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim

        self.init_std = init_std
        super().__init__(
            **kwargs
        )

    # attribute_map = {
    #     "num_attention_heads": "encoder_attention_heads",
    #     "hidden_size": "d_model"
    # }

    