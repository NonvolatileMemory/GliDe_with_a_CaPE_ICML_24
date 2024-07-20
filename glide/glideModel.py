# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" LLaMA model configuration"""
import itertools
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


from types import MethodType
from typing import Optional, Tuple, Dict, Any
from transformers import LlamaTokenizer, AutoTokenizer

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaAttention,
    LlamaModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

from einops import rearrange

from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
)
from flash_attn.ops.rms_norm import rms_norm
logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from glide.utils.speculation_utils import *
from glide.utils.kv_cache import initialize_past_key_values, KVCache
from glide.utils.generation_utils import beam_sample, beam_search, greedy_search, expand_cape_search

from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    TemperatureLogitsWarper
)
from transformers import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer, TopKLogitsWarper
import contexttimer

class smallLlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
            is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.


    ```python
    >>> from transformers import LlamaModel, smallLlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = smallLlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        block_size=5,
        large_hidden_size=4096,
        large_head_dim=128,
        large_num_attention_heads=32,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_remap=False,
        **kwargs,
    ):
        self.use_remap = use_remap
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.large_hidden_size = large_hidden_size
        self.large_head_dim = large_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.large_hidden_size = large_hidden_size
        self.large_num_attention_heads = large_num_attention_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "smallLlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_single_rotary_pos_emb(q, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: smallLlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # print(past_key_value)
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)

        # print(past_key_value)
        # print(use_cache)
        
        past_key_value = past_key_value if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: smallLlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.large_hidden_size = config.large_hidden_size # config.large_hidden_size if config.large_hidden_size is not None else self.hidden_size
        self.large_num_heads = config.large_num_attention_heads
        self.large_head_dim = self.large_hidden_size // self.large_num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.large_num_heads * self.large_head_dim, bias=False)
        self.o_proj = nn.Linear(self.large_num_heads * self.large_head_dim, self.hidden_size, bias=False)
        self._init_rope()
        self.block_size = config.block_size if config.block_size is not None else 5

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.large_head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.large_head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.large_head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _block_wise_attention_mask(self, q_len, kv_seqlen, block_size):
        seqlen = max(kv_seqlen, q_len)
        block_indices = torch.arange(seqlen) // block_size
        attention_condition = block_indices.unsqueeze(1) < block_indices.unsqueeze(0) + 1
        mask = attention_condition.float()
        mask[mask.eq(1)] = float("-inf")
        return mask[None, None, :q_len, :kv_seqlen]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        large_key_values: Optional[Tuple[torch.Tensor]] = None, # bsz numheads, kvlen, head_dim
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        _, _, kv_len, _ = large_key_values[0].shape
        query_states = self.q_proj(hidden_states)

        kv_seq_len = large_key_values[0].shape[-2]
        
        key_states, value_states = large_key_values[0].data[:,:,0:kv_seq_len,:].clone(), large_key_values[1].data[:,:,0:kv_seq_len,:].clone()
        query_states = query_states.view(bsz, -1, self.large_num_heads, self.large_head_dim).transpose(1, 2)
        
        # for rope
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len + 32) # extra space for query

        query_states = apply_single_rotary_pos_emb(query_states, cos, sin, position_ids)
        
        # repeat k/v heads if n_kv_heads < n_heads
        if key_states.size(1) != query_states.size(1):
            nrep = query_states.size(1) // key_states.size(1)
            key_states = repeat_kv(key_states, nrep)
            value_states = repeat_kv(value_states, nrep)


        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.large_head_dim)

        if attn_weights.size() != (bsz, self.large_num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.large_num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:# and q_len > 1:
            if self.training:
                block_attention_mask = self._block_wise_attention_mask(q_len, large_key_values[0].shape[-2], self.block_size)
                # If no attention mask, then no block wise attention mask
                attention_mask_copy = attention_mask[:, :, :, :large_key_values[0].shape[-2]].clone()
                block_attention_mask = block_attention_mask.expand(bsz, -1, -1, -1).to(attention_mask.device).to(attention_mask.dtype)

                block_attention_mask[attention_mask_copy.eq(float('-inf'))] = float('-inf') # set padding according to original attention mask
                block_attention_mask[:, :, :self.block_size] = attention_mask_copy[:, :, :self.block_size]
                attention_mask = block_attention_mask
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )

                attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.large_num_heads, q_len, self.large_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.large_num_heads, q_len, self.large_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.large_hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, None

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: smallLlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.cross_attn = LlamaCrossAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        large_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + hidden_states

        # for corss attention
        residual = hidden_states

        if large_key_values is not None:
            # print("cross")
            hidden_states, _, _ = self.cross_attn(
                hidden_states = hidden_states,
                attention_mask=attention_mask,
                position_ids =position_ids,
                large_key_values=large_key_values,
                output_attentions=False,
                use_cache=True)

            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`smallLlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = smallLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
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
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: smallLlamaConfig
    """

    def __init__(self, config: smallLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
            # (bsz, 1, tgt_len, tgt_len + past_key_values_length)
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # bsz, 1, input.size(-1), seq_len
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        
        # # [MODIFIED] add medusa mask
        # if hasattr(self, "medusa_mask") and self.medusa_mask is not None:
        #     medusa_mask = self.medusa_mask
        #     medusa_len = medusa_mask.size(-1)
        #     combined_attention_mask[:, :, -medusa_len:, -medusa_len:][
        #         medusa_mask == 0
        #     ] = combined_attention_mask.min()
        #     if hasattr(self, "medusa_mode"):
        #         # debug mode
        #         if self.medusa_mode == "debug":
        #             torch.save(combined_attention_mask, "medusa_mask.pt")

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        large_key_values:  Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = [] if use_cache else None

        if large_key_values is not None:
            large_layers = len(large_key_values)
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # print("decoder")
            # print(past_key_values[idx])
            past_key_value = past_key_values[idx] if past_key_values is not None  else None

            last = False
            if large_key_values is not None:
                if last:
                    large_index = -1
                else:
                    large_index = large_layers - len(self.layers) + idx
                    large_index = min(large_index, large_layers - 1)
                # print("large_index")
                # print(large_index)
                curr_large_kv = large_key_values[large_index]
                # print(curr_large_kv)
                # print(curr_large_kv[0].shape)
                # exit()
            else:
                curr_large_kv = None
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                assert past_key_value is None
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    curr_large_kv,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    large_key_values=curr_large_kv,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache.append(layer_outputs[2 if output_attentions else 1])

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class smallLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.large_model = None
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,# [MODIFIED] past_key_value is KVCache class
        large_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print(type(past_key_values))
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            large_key_values=large_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def medusa_forward(
        self,
        large_model,
        input_ids=None,
        attention_mask=None,
        labels=None,
        large_past_key_values=None,
        small_past_key_values=None,
        output_orig=True, #这个保持true就行，实际上多写了，false反而不对
        position_ids=None,
        large_kv=True,
        num_beams=1, # >1 开启beam search，并且后续只有steps_to_sample起作用了，其他参数要保持默认（不指定）
        num_return_sequences=1,
        do_beam_sample=False,
        large_logits=None,
        topk=None,
        do_speculative_sample=False, # 是否要大模型sample，不能与beam搭配使用。可以用，但是论文没有放出这个的结果，没怎么做实验
        temperature=1.0,
        propose_num=4,
        prob_to_top=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        steps_to_sample=None, # 是否beam sample
        expand_cape=False, # 具体可见beam_utils里面的expand_cape_search的注释
        batch_size=1 # 实验大batch下cape的速度，但是是假定在比较理想情况下的实验
    ) -> torch.Tensor:
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """

        #1 2  3  4  5  6  7   8   9   10 10

        def sample_method(logits):
            return torch.softmax(logits / temperature, dim=-1)
        
        def generate_proposals(logits):
            if steps_to_sample is not None:
                probs = sample_method(logits)
                steps = probs.shape[-2]
                sample_num = max(steps_to_sample)
                sample_num_pad = min(steps_to_sample)
                
                sample_tokens = torch.multinomial(probs[0, :, :], num_samples=sample_num)
                # sample_tokens = torch.topk(probs[0, :, :], k=sample_num, dim=-1).indices
                
                if len(steps_to_sample) < steps:
                    steps_to_sample.extend([sample_num_pad] * (steps - len(steps_to_sample)))
                
                all_choices = [[[token] for token in sample_tokens[0, :steps_to_sample[0]].tolist()]]

                for i in range(1, steps):
                    cur_available_tokens = sample_tokens[i, :steps_to_sample[i]]

                    choices = list(itertools.product(all_choices[-1], cur_available_tokens))
                    all_choices.append([x+[y] for x,y in choices])
                    
                all_choices = [p for proposal in all_choices for p in proposal]

                return all_choices

            elif expand_cape:
                probs = sample_method(logits)
                steps = probs.shape[-2]
                
                max_probs = torch.max(probs, dim=-1).values
                min_max_probs = max_probs.min().item() #torch.min(max_probs) # 最大里面的最小来截断topk, 避免开销

                k = prob_to_top[int(min_max_probs * 10)]

                available_tokens = torch.topk(logits, k=k, dim=-1).indices
                batch_available_tokens = available_tokens.tolist()
                batched_all_choices = []
                
                # 循环处理每一个sequence，因为一个batch里面有不同序列
                for b in range(probs.size(0)):
                    trajectory = available_tokens[b, :, 0].tolist()
                    
                    all_choices = []

                    for i in range(steps):
                        cur_available_tokens = batch_available_tokens[b][i][0: prob_to_top[int(max_probs[b, i] * 10)]]

                        choices = [trajectory[0:i]+[token] for token in cur_available_tokens]
                        all_choices.append(choices)
                    all_choices = [p for proposal in all_choices for p in proposal]
                    batched_all_choices.extend(all_choices)

                return batched_all_choices
            
            # 这里是cape方法
            probs = sample_method(logits)
            steps = probs.shape[-2]
            
            max_probs = torch.max(probs[0, :, :], dim=-1).values
            min_max_probs = max_probs.min().item() #torch.min(max_probs) # 最大里面的最小来截断topk, 避免开销

            k = prob_to_top[int(min_max_probs * 10)]

            # 正常的cape batch是1，所以logits[0]，如果batch大于1也可以直接索引，因为假设batch里面sequence是一样的
            available_tokens = torch.topk(logits[0, :, :], k=k, dim=-1).indices

            # 专门存每步greedy的结果，方便后面构造出proposal
            trajectory = available_tokens[:, 0].tolist()
            available_tokens = available_tokens.tolist()
            
            all_choices = []

            # 对每一步，构造出到这一步的proposal
            # 如果第3步就是构造出000，001，002类似的proposal
            for i in range(steps):
                cur_available_tokens = available_tokens[i][0: prob_to_top[int(max_probs[i] * 10)]]

                choices = [trajectory[0:i]+[token] for token in cur_available_tokens]
                all_choices.append(choices)
            
            # 最后弄成一维度数组
            all_choices = [p for proposal in all_choices for p in proposal]

            return all_choices

        with torch.inference_mode():
            
            # large没有logits说明还需要prefix forward一下
            if large_logits is None:
                outputs = large_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=large_past_key_values,
                    position_ids=position_ids,
                    return_dict=True
                )
                large_key_values = outputs.past_key_values
                
                if output_orig:
                    orig = outputs.logits
                    
                    # 大模型是否要sample
                    if not do_speculative_sample:
                        selected_tokens = orig[:, -1:, :].argmax(dim=-1)
                    else:
                        large_logits = sample_method(orig[:, -1, :])
                        selected_tokens = torch.multinomial(large_logits, num_samples=1)

                    input_ids = torch.cat([input_ids, selected_tokens], dim=-1)

            # 如果有large logit，说明这时候大模型已经验证了一次，这里返回的logit实际上就是下一个token的概率分布，这实际是免费的午餐
            else:
                large_key_values = large_past_key_values
                orig = large_logits
                
                if not do_speculative_sample:
                    selected_tokens = large_logits.argmax(dim=-1)
                else:
                    large_logits = sample_method(large_logits)
                    selected_tokens = torch.multinomial(large_logits[:, -1, :], num_samples=1)

                input_ids = torch.cat([input_ids, selected_tokens.repeat(batch_size, 1)], dim=-1)
                
            
            logits_processor = LogitsProcessorList()
            logits_warper = LogitsProcessorList()

            if topk is not None:
                logits_warper.append(TopKLogitsWarper(top_k=topk))
            if do_beam_sample or do_speculative_sample:
                logits_warper.append(TemperatureLogitsWarper(temperature=temperature))
            
            # 强制最多生成4个
            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=4 + input_ids.size(-1))])

            # num_beams > 1 进入beam推理
            if num_beams > 1:
                algo_time = 0
                with contexttimer.Timer() as algo_t:
                    beam_scorer = BeamSearchScorer(batch_size=1, num_beams=num_beams, device=self.device, num_beam_hyps_to_keep=num_return_sequences)
                    
                    input_ids, model_kwargs = self._expand_inputs_for_generation(
                        input_ids=input_ids,
                        expand_size=num_beams,
                        is_encoder_decoder=self.config.is_encoder_decoder
                    )
                    
                    # 是否beam sample
                    if do_beam_sample:
                        beam_output = beam_sample(self, input_ids,
                            beam_scorer=beam_scorer,
                            past_key_values=small_past_key_values,
                            logits_processor=logits_processor,
                            logits_warper=logits_warper,
                            stopping_criteria=stopping_criteria,
                            large_key_values=large_key_values if large_kv else None, 
                            large_kv=large_kv,
                            use_cache=True,
                        )
                    else:
                        beam_output = beam_search(self, input_ids,
                            beam_scorer=beam_scorer,
                            past_key_values=small_past_key_values,
                            logits_processor=logits_processor,
                            logits_warper=logits_warper,
                            stopping_criteria=stopping_criteria,
                            large_key_values=large_key_values if large_kv else None,
                            large_kv=large_kv,
                            use_cache=True,
                        )
                    
                    # 此时的beam的结果就是proposal（还没有前缀用于生成medusa mask）
                    small_proposed_tokens = beam_output["sequences"][:,input_ids.size(-1):].tolist()
                    # print(small_proposed_tokens)
                    scores = None
                    
                algo_time += algo_t.elapsed
                
                if do_speculative_sample:
                    raise NotImplementedError("Doesn't support speculative sample now!")
            else:
                algo_time = 0
                if expand_cape:
                    beam_output = expand_cape_search(self, input_ids,
                        past_key_values=small_past_key_values,
                        logits_processor=logits_processor,
                        logits_warper=logits_warper,
                        stopping_criteria=stopping_criteria,
                        large_key_values=large_key_values if large_kv else None,
                        large_kv=large_kv,
                        use_cache=True,
                    )
                    steps = len(beam_output["scores"])
                    scores = []
                    for i in range(steps):
                        scores.append(beam_output["scores"][i].repeat(2**(4-i), 1))
                    
                    scores = torch.stack(scores, dim=1)

                    # now return all choices
                    small_proposed_tokens = generate_proposals(scores)
                else:
                    orig_length = input_ids.size(1)
                    
                    with contexttimer.Timer() as algo_t:
                        beam_output = greedy_search(self, input_ids,
                            past_key_values=small_past_key_values,
                            logits_processor=logits_processor,
                            logits_warper=logits_warper,
                            stopping_criteria=stopping_criteria,
                            large_key_values=large_key_values if large_kv else None,
                            large_kv=large_kv,
                            use_cache=True,
                        )
                        
                        if batch_size > 1:
                            scores = None
                            
                            small_proposed_tokens = beam_output["sequences"][0:1, orig_length:].tolist()
                        
                        else:
                            scores = torch.stack(beam_output["scores"], dim=1)
                
                            # now return all choices
                            small_proposed_tokens = generate_proposals(scores)
                    
                    algo_time += algo_t.elapsed
        
        if output_orig:
            return {
                "large_last_token": selected_tokens.item() if batch_size==1 else selected_tokens[0].item(),
                "small_proposed_tokens": small_proposed_tokens,
                "large_logits": orig,
                "small_scores": scores,
                "algo_time": algo_time
            }
        return outputs
    
    # 流式输出与medusa generate不同点就在流式用了yield，其他都差不多
    def medusa_generate_stream(
        self,
        input_ids,
        large_model,
        large_kv=True,
        attention_mask=None,
        temperature=1.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        # medusa_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        num_beams=1,
        num_return_sequences=1,
        topk=None,
        do_beam_sample=False,
        do_speculative_sample=False,
        tokenizer_path='/home/ducunxiao/model/vicuna-7b-v1.5',
        steps_to_sample=None,
        prob_to_top=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        expand_cape=False,  # 具体可见beam_utils里面的expand_cape_search的注释
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        # print("start generate")
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # assert num_beams == 1, "Only support num_beams 1 for now!!"
        # assert num_return_sequences == 1, "Only support num_return_sequences 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            large_past_key_values = large_model.past_key_values
            large_past_key_values_data = large_model.past_key_values_data
            large_current_length_data = large_model.current_length_data
            small_past_key_values = self.past_key_values
            small_past_key_values_data = self.past_key_values_data
            small_current_length_data = self.current_length_data
            # Reset the past key and value states
            large_current_length_data.zero_()
            small_current_length_data.zero_()
        else:
            (
                large_past_key_values,
                large_past_key_values_data,
                large_current_length_data,
                small_past_key_values,
                small_past_key_values_data,
                small_current_length_data,
            ) = initialize_past_key_values(large_model=large_model, small_model=self, num_return_sequences=num_return_sequences, two_kv=True)
            large_model.past_key_values = large_past_key_values
            large_model.past_key_values_data = large_past_key_values_data
            large_model.current_length_data = large_current_length_data
            self.past_key_values = small_past_key_values
            self.past_key_values_data = small_past_key_values_data
            self.current_length_data = small_current_length_data

        input_len = input_ids.shape[1]
        
        reset_medusa_mode(self, large_model)
        
        # 原有的medusa初始化主要是为了加medusa mask（加cache），对我们没用
        # 我们要动态计算mask
        # 返回的token直接就是Cartesian candidates
        # 要适配medusa代码，我们要人为生成medusa choices，后续从top_k里面选token的过程要略过
        # 算mask还是要medusa choices
        
        forward_output = self.medusa_forward(
            input_ids=input_ids, 
            large_past_key_values=large_past_key_values, 
            small_past_key_values=small_past_key_values, 
            large_model=large_model, 
            output_orig=True,
            large_kv=large_kv,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            topk=topk,
            do_beam_sample=do_beam_sample,
            do_speculative_sample=do_speculative_sample,
            temperature=temperature,
            prob_to_top=prob_to_top,
            steps_to_sample=steps_to_sample,
            expand_cape=expand_cape
        )
        
        large_last_token = forward_output["large_last_token"]
        small_proposed_tokens = forward_output["small_proposed_tokens"]
        large_logits = forward_output["large_logits"]
        small_scores = forward_output["small_scores"]

        new_token = 0
        new_token_sum = 0
        accept_length_sum = 0
        alpha_sum = 0
        last_round_token = 0
        
        for idx in range(max_steps):
            new_token = 0
            assert large_model.model.medusa_mask == None
            assert large_model.model.medusa_mode == None
            
            medusa_buffers = generate_medusa_buffers_joint_llama(
                small_proposed_tokens, large_last_token=large_last_token, device=self.device, num_beams=num_beams
            )
            large_model.model.medusa_mask = medusa_buffers["medusa_attn_mask"]
            
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates_joint_llama(
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
            )
            
            
            # Use tree attention to verify the candidates and get predictions
            large_logits, tree_logits = tree_decoding_joint_llama(
                model=large_model,
                tree_candidates=tree_candidates,
                past_key_values=large_past_key_values,
                medusa_position_ids=medusa_buffers["medusa_position_ids"],
                input_ids=input_ids,
                retrieve_indices=medusa_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            evaluation_results = evaluate_posterior_joint_llama(
                large_logits, candidates, temperature, posterior_threshold, posterior_alpha,
                small_scores=small_scores,
                tree_logits=tree_logits,
                tree_candidates=tree_candidates,
                do_speculative_sample=do_speculative_sample,
                retrieve_indices=medusa_buffers["retrieve_indices"],
                retrieve_steps=medusa_buffers["retrieve_steps"]
                # candidates_to_small_kv_indice=candidates_to_small_kv_indice
            )
            best_candidate, accept_length, alpha = evaluation_results["best_candidate"], evaluation_results["accept_length"], evaluation_results["alpha"]
            
            if alpha is not None:
                alpha_sum+=alpha
            
            accept_length_sum += accept_length
            # Update the input_ids and logits
            input_ids, large_logits, new_token = update_inference_inputs_joint_llama(
                input_ids=input_ids,
                candidates=candidates,
                best_candidate=best_candidate,
                accept_length=accept_length,
                retrieve_indices=medusa_buffers["retrieve_indices"],
                logits=large_logits,
                new_token=new_token,
                large_past_key_values_data=large_past_key_values_data,
                small_past_key_values_data=small_past_key_values_data,
                large_current_length_data=large_current_length_data,
                small_current_length_data=small_current_length_data,
            )

            new_token_sum+=new_token

            text = self.tokenizer.decode(
                input_ids[0, input_len:],
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            
            yield {
                "text": text,
                "length": len(self.tokenizer.decode(
                    input_ids[0, -accept_length:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )) if accept_length>0 else 0,
                "accept_length": accept_length,
                "new_token": new_token
            }


            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                print(f"sum accept: {accept_length_sum}")
                print(f"accept rate: {accept_length_sum/new_token_sum}")
                break

            reset_medusa_mode(self, large_model)

            forward_output = self.medusa_forward(
                input_ids=input_ids, 
                large_past_key_values=large_past_key_values, 
                small_past_key_values=small_past_key_values, 
                large_model=large_model, 
                output_orig=True,
                large_kv=large_kv,
                large_logits=large_logits,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                topk=topk,
                do_beam_sample=do_beam_sample,
                do_speculative_sample=do_speculative_sample,
                temperature=temperature,
                prob_to_top=prob_to_top,
                steps_to_sample=steps_to_sample,
                expand_cape=expand_cape
            )
            
            large_last_token = forward_output["large_last_token"]
            small_proposed_tokens = forward_output["small_proposed_tokens"]
            large_logits = forward_output["large_logits"]
            small_scores = forward_output["small_scores"]
        print(f"sum accept: {accept_length_sum}")
        print(f"accept rate: {accept_length_sum/new_token_sum}")    


    def medusa_generate(
        self,
        input_ids,
        large_model,
        large_kv=True,
        attention_mask=None,
        temperature=1.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        # medusa_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        num_beams=1,
        num_return_sequences=1,
        topk=None,
        do_beam_sample=False,
        do_speculative_sample=False,
        steps_to_sample=None,
        prob_to_top=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        expand_cape=False,
        batch_size=1, # batch size>1 时是为了测理想情况下不同batch大小cape的速度
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        # print("start generate")
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # assert num_beams == 1, "Only support num_beams 1 for now!!"
        # assert num_return_sequences == 1, "Only support num_return_sequences 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        
        if batch_size > 1:
            input_ids = input_ids.repeat(batch_size, 1)

        self.tokenizer = AutoTokenizer.from_pretrained("/home/ducunxiao/model/vicuna-7b-v1.5")

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            large_past_key_values = large_model.past_key_values
            large_past_key_values_data = large_model.past_key_values_data
            large_current_length_data = large_model.current_length_data
            small_past_key_values = self.past_key_values
            small_past_key_values_data = self.past_key_values_data
            small_current_length_data = self.current_length_data
            # Reset the past key and value states
            large_current_length_data.zero_()
            small_current_length_data.zero_()
        else:
            (
                large_past_key_values,
                large_past_key_values_data,
                large_current_length_data,
                small_past_key_values,
                small_past_key_values_data,
                small_current_length_data,
            ) = initialize_past_key_values(large_model=large_model, small_model=self, num_return_sequences=num_return_sequences, two_kv=True, batch_size=batch_size)
            large_model.past_key_values = large_past_key_values
            large_model.past_key_values_data = large_past_key_values_data
            large_model.current_length_data = large_current_length_data
            self.past_key_values = small_past_key_values
            self.past_key_values_data = small_past_key_values_data
            self.current_length_data = small_current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self, large_model)
        
        # 原有的medusa初始化主要是为了加medusa mask（加cache），对我们没用
        # 我们要动态计算mask
        # 返回的token直接就是Cartesian candidates
        # 要适配medusa代码，我们要人为生成medusa choices，后续从top_k里面选token的过程要略过
        # 算mask还是要medusa choices
       
        prefix_time = 0
        with contexttimer.Timer() as prefix_t:
            forward_output = self.medusa_forward(
                input_ids=input_ids, 
                large_past_key_values=large_past_key_values, 
                small_past_key_values=small_past_key_values, 
                large_model=large_model, 
                output_orig=True,
                large_kv=large_kv,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                topk=topk,
                do_beam_sample=do_beam_sample,
                do_speculative_sample=do_speculative_sample,
                temperature=temperature,
                prob_to_top=prob_to_top,
                steps_to_sample=steps_to_sample,
                expand_cape=expand_cape,
                batch_size=batch_size
            )
        prefix_time += prefix_t.elapsed
        
        large_last_token = forward_output["large_last_token"]
        small_proposed_tokens = forward_output["small_proposed_tokens"]
        large_logits = forward_output["large_logits"]
        small_scores = forward_output["small_scores"]
        algo_time = forward_output["algo_time"]
   
        new_token_sum = 0
        accept_length_sum = 0
        alpha_sum = 0
        last_round_token = 0
        proposal_num = 0

        generation_time = 0
        algo_time_sum = algo_time
        with contexttimer.Timer() as gen_t:
            for idx in range(max_steps):
                new_token = 0
                assert large_model.model.medusa_mask == None
                assert large_model.model.medusa_mode == None

                medusa_buffers = generate_medusa_buffers_joint_llama(
                    small_proposed_tokens, large_last_token=large_last_token, device=self.device, num_beams=num_beams
                )
                
                # medusa mask用于告诉大模型验证时哪些token组合起来是一个序列
                large_model.model.medusa_mask = medusa_buffers["medusa_attn_mask"].repeat(batch_size, 1, 1, 1)
                
                # Generate candidates with topk predictions from Medusa heads
                candidates, tree_candidates = generate_candidates_joint_llama(
                    medusa_buffers["tree_indices"], # 就是proposal扁平化之后的token id
                    medusa_buffers["retrieve_indices"], # 用于取回proposal
                )

                # print("start verifying using large model")
                # Use tree attention to verify the candidates and get predictions
                # 返回对应logit
                large_logits, tree_logits = tree_decoding_joint_llama(
                    model=large_model,
                    tree_candidates=tree_candidates,
                    past_key_values=large_past_key_values,
                    medusa_position_ids=medusa_buffers["medusa_position_ids"],
                    input_ids=input_ids,
                    retrieve_indices=medusa_buffers["retrieve_indices"],
                    batch_size=batch_size
                )
                
                # Evaluate the posterior of the candidates to select the accepted candidate prefix
                evaluation_results = evaluate_posterior_joint_llama(
                    large_logits, candidates, temperature, posterior_threshold, posterior_alpha,
                    small_scores=small_scores,
                    tree_logits=tree_logits,
                    tree_candidates=tree_candidates,
                    do_speculative_sample=do_speculative_sample,
                    retrieve_indices=medusa_buffers["retrieve_indices"],
                    retrieve_steps=medusa_buffers["retrieve_steps"]
                )
                best_candidate, accept_length, alpha = evaluation_results["best_candidate"], evaluation_results["accept_length"], evaluation_results["alpha"]
                
                if alpha is not None:
                    alpha_sum+=alpha
                
                accept_length_sum+=accept_length
                proposal_num += tree_candidates.size(1)

                # Update the input_ids and logits
                input_ids, large_logits, new_token = update_inference_inputs_joint_llama(
                    input_ids=input_ids,
                    candidates=candidates,
                    best_candidate=best_candidate,
                    # best_small_kv_indice=best_small_kv_indice,
                    accept_length=accept_length,
                    retrieve_indices=medusa_buffers["retrieve_indices"],
                    logits=large_logits,
                    new_token=new_token,
                    large_past_key_values_data=large_past_key_values_data,
                    small_past_key_values_data=small_past_key_values_data,
                    large_current_length_data=large_current_length_data,
                    small_current_length_data=small_current_length_data,
                )
                new_token_sum+=new_token

                if self.tokenizer.eos_token_id in input_ids[0, input_len:] or new_token_sum>=max_steps:
                    # print(f"sum accept: {accept_length_sum}")
                    # print(f"accept rate: {accept_length_sum/new_token_sum}")
                    break
                
                # 验证完后reset
                reset_medusa_mode(self, large_model)

                forward_output = self.medusa_forward(
                    input_ids=input_ids, 
                    large_past_key_values=large_past_key_values, 
                    small_past_key_values=small_past_key_values, 
                    large_model=large_model, 
                    output_orig=True,
                    large_kv=large_kv,
                    large_logits=large_logits,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    topk=topk,
                    do_beam_sample=do_beam_sample,
                    do_speculative_sample=do_speculative_sample,
                    temperature=temperature,
                    prob_to_top=prob_to_top,
                    steps_to_sample=steps_to_sample,
                    expand_cape=expand_cape,
                    batch_size=batch_size
                )
                
                large_last_token = forward_output["large_last_token"]
                small_proposed_tokens = forward_output["small_proposed_tokens"]
                large_logits = forward_output["large_logits"]
                small_scores = forward_output["small_scores"]
                algo_time_sum += forward_output["algo_time"]
                
        
        generation_time += gen_t.elapsed

        return {
            "input_ids": input_ids,
            "steps": idx+1,
            "new_token_sum": new_token_sum,
            "accept_length_sum": accept_length_sum,
            "alpha_sum": alpha_sum,
            "prefix_time": prefix_time,
            "generation_time": generation_time,
            "algo_time": algo_time_sum,
            "proposal_num": proposal_num,
            "prob_to_top": prob_to_top,
        }

    @staticmethod
    def _reorder_cache_medusa(past_key_values, beam_idx):
        # beam search会调用这里的代码
        for i in range(len(past_key_values)):
            past_key_values[i][0].index_select(0, beam_idx.to(past_key_values[i][0].data.device))
            past_key_values[i][1].index_select(0, beam_idx.to(past_key_values[i][1].data.device))
        return past_key_values
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, large_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # print(type(past_key_values))
        if past_key_values and past_key_values[0][0].shape[-2]!=0:
            input_ids = input_ids[:, past_key_values[0][0].shape[-2]:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values and past_key_values[0][0].shape[-2]!=0:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation ste
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "large_key_values": large_key_values,
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs
    
