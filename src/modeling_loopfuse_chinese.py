# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch CLIP model."""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    logging,
    torch_int,
)
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


logger = logging.get_logger(__name__)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def image_text_contrastive_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


@auto_docstring(
    custom_intro="""
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    """
)
@dataclass
class CLIPVisionModelOutput(ModelOutput):
    r"""
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The image embeddings obtained by applying the projection layer to the pooler_output.
    """

    image_embeds: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    Base class for text model's outputs that also contains a pooling of the last hidden states.
    """
)
@dataclass
class CLIPTextModelOutput(ModelOutput):
    r"""
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The text embeddings obtained by applying the projection layer to the pooler_output.
    """

    text_embeds: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
@dataclass
class CLIPOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for image-text similarity.
    logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
        The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
        similarity scores.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
        The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
        similarity scores.
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`CLIPTextModel`].
    vision_model_output (`BaseModelOutputWithPooling`):
        The output of the [`CLIPVisionModel`].
    """

    loss: torch.FloatTensor | None = None
    logits_per_image: torch.FloatTensor | None = None
    logits_per_text: torch.FloatTensor | None = None
    text_embeds: torch.FloatTensor | None = None
    image_embeds: torch.FloatTensor | None = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(v.to_tuple() if isinstance(v, ModelOutput) else v for v in self.values())


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = nn.Buffer(torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
            )
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = nn.Buffer(torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CLIPVisionConfig | CLIPTextConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        input_shape = hidden_states.shape[:-1]

        hidden_shape = (*input_shape, -1, self.head_dim)
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(hidden_shape).transpose(1, 2)
        keys = keys.view(hidden_shape).transpose(1, 2)
        values = values.view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: CLIPVisionConfig | CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def _init_child_weight(child: nn.Module, std: float) -> None:
    """Initialize `child.weight` if the child module still carries one. Guards against quantization methods that set `weight` to `None`."""
    if getattr(child, "weight", None) is not None:
        init.normal_(child.weight, std=std)


@auto_docstring
class CLIPPreTrainedModel(PreTrainedModel):
    config: CLIPConfig
    base_model_prefix = "clip"
    input_modalities = ("image", "text")
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer", "CLIPVisionEmbeddings"]

    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": CLIPEncoderLayer,
        "attentions": CLIPAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        factor = self.config.initializer_factor
        if isinstance(module, CLIPTextEmbeddings):
            _init_child_weight(module.token_embedding, factor * 0.02)
            _init_child_weight(module.position_embedding, factor * 0.02)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, CLIPVisionEmbeddings):
            init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            _init_child_weight(module.patch_embedding, module.config.initializer_range * factor)
            _init_child_weight(module.position_embedding, module.config.initializer_range * factor)
            init.copy_(module.position_ids, torch.arange(module.num_positions).expand((1, -1)))
        elif isinstance(module, CLIPAttention):
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            _init_child_weight(module.q_proj, in_proj_std)
            _init_child_weight(module.k_proj, in_proj_std)
            _init_child_weight(module.v_proj, in_proj_std)
            _init_child_weight(module.out_proj, out_proj_std)
        elif isinstance(module, CLIPMLP):
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            _init_child_weight(module.fc1, fc_std)
            _init_child_weight(module.fc2, in_proj_std)
        elif isinstance(module, CLIPModel):
            _init_child_weight(module.text_projection, module.text_embed_dim**-0.5 * factor)
            _init_child_weight(module.visual_projection, module.vision_embed_dim**-0.5 * factor)
        elif isinstance(module, CLIPVisionModelWithProjection):
            _init_child_weight(module.visual_projection, self.config.hidden_size**-0.5 * factor)
        elif isinstance(module, CLIPTextModelWithProjection):
            _init_child_weight(module.text_projection, self.config.hidden_size**-0.5 * factor)
        elif isinstance(module, CLIPForImageClassification):
            _init_child_weight(module.classifier, self.config.vision_config.hidden_size**-0.5 * factor)


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                **kwargs,
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The text model from CLIP without any head or projection on top.
    """
)
class CLIPTextModel(CLIPPreTrainedModel):
    config: CLIPTextConfig
    input_modalities = ("text",)
    _input_embed_layer = "token_embedding"

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
        )

        kwargs.pop("is_causal", None)
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            is_causal=True,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of extra new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


@auto_docstring(
    custom_intro="""
    The vision model from CLIP without any head or projection on top.
    """
)
class CLIPVisionModel(CLIPPreTrainedModel):
    config: CLIPVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _input_embed_layer = "patch_embedding"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


@auto_docstring
class CLIPModel(CLIPPreTrainedModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextModel._from_config(text_config)
        self.vision_model = CLIPVisionModel._from_config(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> with torch.inference_mode():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )
        pooled_output = text_outputs.pooler_output
        text_outputs.pooler_output = self.text_projection(pooled_output)

        return text_outputs

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, CLIPModel
        >>> from transformers.image_utils import load_image

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.inference_mode():
        ...     image_features = model.get_image_features(**inputs)
        ```"""
        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
            **kwargs,
        )
        pooled_output = vision_outputs.pooler_output
        vision_outputs.pooler_output = self.visual_projection(pooled_output)

        return vision_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_loss: bool | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CLIPOutput:
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, CLIPModel
        >>> from transformers.image_utils import load_image

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        vision_outputs: BaseModelOutputWithPooling = self.get_image_features(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        text_outputs: BaseModelOutputWithPooling = self.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # normalized features
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))
        logits_per_text = logits_per_text * self.logit_scale.exp().to(text_embeds.device)

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = image_text_contrastive_loss(logits_per_text)

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


@auto_docstring
class CLIPTextModelWithProjection(CLIPPreTrainedModel):
    config: CLIPTextConfig
    input_modalities = ("text",)

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)

        self.text_model = CLIPTextModel._from_config(config)
        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CLIPTextModelOutput:
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, CLIPTextModelWithProjection

        >>> model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        pooled_output = text_outputs.pooler_output
        text_embeds = self.text_projection(pooled_output)

        return CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


@auto_docstring
class CLIPVisionModelWithProjection(CLIPPreTrainedModel):
    config: CLIPVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel._from_config(config)
        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CLIPVisionModelOutput:
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, CLIPVisionModelWithProjection
        >>> from transformers.image_utils import load_image

        >>> model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)
        >>> image_embeds = outputs.image_embeds
        ```"""

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )
        pooled_output = vision_outputs.pooler_output
        image_embeds = self.visual_projection(pooled_output)

        return CLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    CLIP vision encoder with an image classification head on top (a linear layer on top of the pooled final hidden states of
    the patch tokens) e.g. for ImageNet.
    """
)
class CLIPForImageClassification(CLIPPreTrainedModel):
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    def __init__(self, config: CLIPConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vision_model = CLIPVisionModel._from_config(config.vision_config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.vision_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state

        sequence_output = torch.mean(sequence_output[:, 1:, :], dim=1)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )





#####################        Our Model          #########################



class LoopedCLIPTextModel(CLIPTextModel):
    def __init__(
        self,
        config: CLIPTextConfig,
    ):
        super().__init__(config)

        # self.embedding
        # self.encoder
    
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_loops: int=3,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[BaseModelOutputWithPooling]:
        # input_ids: bsz, seq_len
        # attention_mask: bsz, seq_len
        # position_ids: bsz, seq_len
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
        )

        kwargs.pop("is_causal", None)

        out_ls = []
        for _ in range(num_loops):
            
            encoder_outputs: BaseModelOutput = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                is_causal=True,
                **kwargs,
            )

            last_hidden_state = encoder_outputs.last_hidden_state
            last_hidden_state = self.final_layer_norm(last_hidden_state)

            if self.eos_token_id == 2:
                # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
                # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
                # ------------------------------------------------------------
                # text_embeds.shape = [batch_size, sequence_length, transformer.width]
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
                pooled_output = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                ]
            else:
                # The config gets updated `eos_token_id` from PR #24773 (so the use of extra new tokens is possible)
                pooled_output = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                    # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                    # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                    (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                    .int()
                    .argmax(dim=-1),
                ]
            
            out_ls.append(
                BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooled_output,
                )
            )

            hidden_states = last_hidden_state # bsz, seq_len, text_hidden

        return out_ls


class LoopedCLIPVisionModel(CLIPVisionModel):
    def __init__(
        self,
        config: CLIPVisionConfig,
    ):
        super().__init__(config)
    
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        num_loops: int=3,
        **kwargs: Unpack[TransformersKwargs],
    ) -> list[BaseModelOutputWithPooling]:
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        out_ls = []
        for _ in range(num_loops):
            encoder_outputs: BaseModelOutput = self.encoder(
                inputs_embeds=hidden_states,
                **kwargs,
            )

            last_hidden_state = encoder_outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.post_layernorm(pooled_output)

            out_ls.append(
                BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooled_output
                )
            )

            hidden_states = last_hidden_state # bsz, seq_len, vision_hidden
            
        return out_ls



class GQA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads_q: int,
        num_heads_kv: int
    ):
        super().__init__()
        assert num_heads_q % num_heads_kv == 0, (num_heads_q, num_heads_kv)
        assert d_model % num_heads_q == 0, d_model

        self.d_model = d_model
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv

        self.d_k = self.d_model // self.num_heads_q

        self.wq = nn.Linear(d_model, self.d_k * self.num_heads_q, bias=False) # d_model = d_k * num_heads_q
        self.wk = nn.Linear(d_model, self.d_k * self.num_heads_kv, bias=False)
        self.wv = nn.Linear(d_model, self.d_k * self.num_heads_kv, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,mask=None) -> torch.Tensor:
        # x: batch_size, seq_len ,d_model
        # check
        assert len(q.shape) == len(k.shape) == len(v.shape) == 3, (q.shape, k.shape, v.shape)
        assert q.size(-1) == k.size(-1) == v.size(-1) == self.d_model, (q.size(-1), k.size(-1), v.size(-1))
        assert k.size(1) == v.size(1), (k.size(1), v.size(1))
        bsz, seq_len_q, seq_len_kv = q.size(0), q.size(1), k.size(1)
        # bsz, seq_len_q, d_model -> bsz, seq_len_q, num_heads_q, d_k -> bsz, num_heads_q, seq_len_q, d_k
        q = self.wq(q).view(bsz, seq_len_q, self.num_heads_q, self.d_k).transpose(1,2)
        # bsz, seq_len_kv, d_model -> bsz, seq_len_kv, num_heads_kv, d_k -> bsz, num_heads_kv, seq_len_kv, d_k
        k = self.wk(k).view(bsz, seq_len_kv, self.num_heads_kv, self.d_k).transpose(1,2)
        v = self.wv(v).view(bsz, seq_len_kv, self.num_heads_kv, self.d_k).transpose(1,2)
        # Flash only accepts fp16/bf16. Otherwise leave the kernel choice to SDPA.
        use_flash_ctx = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)

        # scaled_dot_product_attention with flash_attn_v2
        with (
            torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION])
            if use_flash_ctx else contextlib.nullcontext()
        ):
            attn_out = F.scaled_dot_product_attention(
                q,k,v,attn_mask=mask, dropout_p=0.0, is_causal=False, enable_gqa=True
            ) # bsz, num_heads_q, seq_len_q, d_k
        # bsz, seq_len_q, num_heads_q, d_k -> bsz, seq_len_q, d_model
        attn_out = attn_out.transpose(1,2).contiguous().view(bsz, seq_len_q, -1)
        out = self.wo(attn_out)
        return out


class MHA(GQA):
    def __init__(
        self,
        d_model: int,
        num_heads:int,
    ):
        super().__init__(
            d_model=d_model,
            num_heads_q=num_heads,
            num_heads_kv=num_heads
        )
        assert d_model % num_heads == 0, (num_heads, d_model)


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout_p: float=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout_p = dropout_p

        self.fc1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))
        

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads_q: int,
        num_heads_kv: int,
        d_ffn: int,
        dropout_p: float=0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.d_ffn = d_ffn
        self.dropout_p = dropout_p

        self.gqa = GQA(
            d_model=d_model,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv
        )

        self.ffn = FFN(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout_p=dropout_p
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        nn.init.ones_(self.ln1.weight)
        nn.init.zeros_(self.ln1.bias)
        nn.init.ones_(self.ln2.weight)
        nn.init.zeros_(self.ln2.bias)


        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(
        self,x,mask=None
    ):
        residual = x
        x = self.ln1(x)
        attn_out = self.dropout1(self.gqa(x,x,x,mask=mask))
        x = residual + attn_out

        residual = x
        x = self.ln2(x)
        ffn_out = self.dropout2(self.ffn(x))
        x = residual + ffn_out

        return x


def _stack_loop_features(outputs) -> torch.Tensor:
    """Stack per-loop pooled vectors into `[batch, num_loops, embed_dim]`."""
    if isinstance(outputs, BaseModelOutputWithPooling):
        features = outputs.pooler_output
    elif torch.is_tensor(outputs):
        features = outputs
    else:
        pooled = []
        for item in outputs:
            if isinstance(item, BaseModelOutputWithPooling):
                pooled.append(item.pooler_output)
            elif torch.is_tensor(item):
                pooled.append(item)
            else:
                raise TypeError(
                    "Loop features must be pooled tensors or BaseModelOutputWithPooling, "
                    f"got {type(item).__name__}."
                )
        assert pooled
        return torch.stack(pooled, dim=1)

    if features.dim() == 2:
        return features.unsqueeze(1)
    if features.dim() == 3:
        return features
    raise ValueError(
        "Pooled loop features must have shape [batch, dim] or [batch, loops, dim], "
        f"got {tuple(features.shape)}."
    )



def _encoder_weights(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Drop a CLIP tower prefix and skip non-persistent position buffers."""
    head = prefix + "."
    weights = {}
    for key, value in state_dict.items():
        if not key.startswith(head):
            continue
        subkey = key[len(head):]
        if subkey.endswith("position_ids"):
            continue
        weights[subkey] = value
    if not weights:
        raise RuntimeError(f"Checkpoint has no weights under {prefix!r}.")
    return weights



class Fuse(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads_q: int,
        num_heads_kv: int,
        dropout_p: float=0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv

        self.gqa_t_v = GQA(
            d_model=embed_dim,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
        )

        self.gqa_v_t = GQA(
            d_model=embed_dim,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv
        )

        self.ffn_t_v = FFN(
            d_model=embed_dim,
            d_ffn=4*embed_dim,
            dropout_p=dropout_p
        )

        self.ffn_v_t = FFN(
            d_model=embed_dim,
            d_ffn=4*embed_dim,
            dropout_p=dropout_p
        )

        self.ln_t_v = nn.LayerNorm(embed_dim)
        self.ln_v_t = nn.LayerNorm(embed_dim)
        nn.init.ones_(self.ln_t_v.weight)
        nn.init.zeros_(self.ln_t_v.bias)
        nn.init.ones_(self.ln_v_t.weight)
        nn.init.zeros_(self.ln_v_t.bias)


    def forward(
        self,
        out_ls_t: list[BaseModelOutputWithPooling] | list[torch.Tensor] | torch.Tensor,
        out_ls_v: list[BaseModelOutputWithPooling] | list[torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Each entry is one loop's pooled hidden state: text EOS, or vision CLS.
        t = _stack_loop_features(out_ls_t)  # bsz, num_loops, embed_dim
        v = _stack_loop_features(out_ls_v)  # bsz, num_loops, embed_dim
        if t.size(-1) != self.embed_dim or v.size(-1) != self.embed_dim:
            raise ValueError(
                f"Fuse expected embed_dim={self.embed_dim}, got text {tuple(t.shape)} and vision {tuple(v.shape)}."
            )

        # cross-attention t_to_v
        attn_t_v = self.gqa_t_v(
            q=t, k=v, v=v, mask=None
        ) # bsz, num_loops, embed_dim
        norm_t_v = self.ln_t_v(attn_t_v)
        ffn_t_v = self.ffn_t_v(
            norm_t_v
        ) # bsz, num_loops, embed_dim
        feature_t_v = ffn_t_v[:,-1] # bsz, embed_dim

        # cross-attention v_to_t
        attn_v_t = self.gqa_v_t(
            q=v,k=t,v=t, mask=None
        ) # bsz, num_loops, embed_dim
        norm_v_t = self.ln_v_t(attn_v_t)
        ffn_v_t = self.ffn_v_t(
            norm_v_t
        ) # bsz, num_loops, embed_dim
        feature_v_t = ffn_v_t[:,-1] # bsz, embed_dim

        return feature_t_v, feature_v_t



class LoopFuse(nn.Module):
    def __init__(
        self,
        clip_path: str,
        num_heads_q: int,
        num_heads_kv: int,
        dropout_p: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self._load_looped_models(clip_path)
        self.fuse = Fuse(
            embed_dim=self.projection_dim,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            dropout_p=dropout_p,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.projection_dim * 2, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.projection_dim, num_classes),
        )

    def _load_looped_models(self, clip_path: str) -> None:
        from transformers import CLIPModel, AutoProcessor, AutoTokenizer

        official = CLIPModel.from_pretrained(
            clip_path,
            trust_remote_code=True,
        )
        state_dict = official.state_dict()
        config = official.config
        del official

        self.text_encoder = LoopedCLIPTextModel(config.text_config)
        self.vision_encoder = LoopedCLIPVisionModel(config.vision_config)
        self.text_encoder.load_state_dict(_encoder_weights(state_dict, "text_model"), strict=True)
        self.vision_encoder.load_state_dict(_encoder_weights(state_dict, "vision_model"), strict=True)

        self.text_hidden = int(config.text_config.hidden_size)
        self.vision_hidden = int(config.vision_config.hidden_size)
        self.projection_dim = int(config.projection_dim)
        # Fuse reads one shared width. Keep the text loop hidden states as they are,
        # and map vision CLS states into that width when the towers differ.
        
        self.text_projection = nn.Linear(self.text_hidden, self.projection_dim, bias=False)
        self.visual_projection = nn.Linear(self.vision_hidden, self.projection_dim, bias=False)

        # self.embed_dim = text_hidden
        # if text_hidden == vision_hidden:
        #     self.vision_align = nn.Identity()
        #     return

        assert ("text_projection.weight" in state_dict) and ("visual_projection.weight" in state_dict)

        with torch.no_grad():
            self.text_projection.weight.copy_(state_dict["text_projection.weight"])
            self.visual_projection.weight.copy_(state_dict["visual_projection.weight"])


        # load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(clip_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)


    def process_texts(self, texts: list[str]):
        if isinstance(texts, str):
            texts = [texts]
        processed = self.tokenizer(texts, padding=True, truncation=True,
               max_length=50, return_tensors="pt")
        return processed # input_ids=t["input_ids"], attention_mask=t["attention_mask"]

    def process_images(self, images: list[str]):
        if isinstance(images, str):
            images = [images]
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in images]
        processed = self.processor(images=images, return_tensors="pt")
        return processed # pixel_values=v["pixel_values"]


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        num_loops: int = 3,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input_ids: bsz, seq_len
        # attention_mask: bsz, seq_len
        # pixel_values: bsz, c, h, w
        if num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")

        text_outs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_loops=num_loops,
        )
        vision_outs = self.vision_encoder(
            pixel_values=pixel_values,
            num_loops=num_loops,
        )
        text_loops = [self.text_projection(out.pooler_output) for out in text_outs]
        vision_loops = [self.visual_projection(out.pooler_output) for out in vision_outs]
        feature_t_v, feature_v_t = self.fuse(text_loops, vision_loops)
        return self.classifier(torch.cat([feature_t_v, feature_v_t], dim=-1))


    def compute_loss(self, batch_inputs, batch_labels) -> torch.Tensor:
        pass

    

        


__all__ = [
    'LoopedCLIPTextModel_Chinese',
    'LoopedCLIPVisionModel_Chinese',
    'Fuse',
    'LoopFuse_Chinese'
]
