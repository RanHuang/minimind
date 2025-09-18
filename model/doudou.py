from typing import Any, Tuple, Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import math


class DoudouConfig(PretrainedConfig):
    model_type = "doudou"

    def __init__(self,
                 vocab_size: int = 6400,
                 hidden_size: int = 512,
                 num_attention_heads: int = 8,
                 num_hidden_layers: int = 16,
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout


class DoudouForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DoudouConfig

    def __init__(self, config: DoudouConfig):
        super().__init__(config)
        self.config = config
        self.model = DoudouModel(config)
        self.Output = CausalLMOutputWithPast()

    def forward(self,
                input_ids: torch.Tensor,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: bool = False,
                **kwargs):
        logits, hidden_states = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, use_cache=use_cache)

        self.Output.logits = logits
        self.Output.hidden_states = hidden_states

        return self.Output

class DoudouModel(torch.nn.Module):
    def __init__(self, config: DoudouConfig):
        super().__init__()
        self.config = config
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.hidden_size)
        self.output_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        freq_cos, freq_sin = compute_rope_freq(config.hidden_size // config.num_attention_heads)
        self.register_buffer("freq_cos", freq_cos)
        self.register_buffer("freq_sin", freq_sin)

    def forward(self,
                input_ids: torch.Tensor,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: bool = False) -> tuple[Any, Any]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        _, seq_len = input_ids.shape
        hidden_states = self.token_embedding(input_ids)

        hidden_states = self.dropout(hidden_states)

        position_embedding = (self.freq_cos[0:seq_len], self.freq_sin[0:seq_len])
        for transformer_block in self.layers:
            hidden_states = transformer_block(hidden_states, position_embedding, use_cache=use_cache)

        hidden_states = self.final_norm(hidden_states)
        logits = self.output_head(hidden_states)
        return logits, hidden_states

# Build a Large Language Model(From Scratch) P106
class TransformerBlock(torch.nn.Module):
    def __init__(self, config: DoudouConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size)
        self.multi_head_attention = MultiHeadAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)

    def forward(self,
                input_ids: torch.Tensor,
                position_embedding: Tuple[torch.Tensor, torch.Tensor],
                use_cache: bool = False) -> torch.Tensor:
        shortcut = input_ids
        hidden_states = self.input_norm(input_ids)
        hidden_states = self.multi_head_attention(hidden_states, position_embedding, use_cache=use_cache)
        hidden_states = shortcut + hidden_states

        shortcut = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = shortcut + hidden_states
        return hidden_states


class FeedForward(torch.nn.Module):
    def __init__(self, config: DoudouConfig):
        super().__init__()
        hidden_size = config.hidden_size
        assert hidden_size % 64 == 0
        intermediate_size = hidden_size * 4
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.activation(self.gate_proj(input_ids)) * self.up_proj(input_ids)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: DoudouConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.W_query = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_key = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_value = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attention_dropout = torch.nn.Dropout(config.dropout)
        self.output_dropout = torch.nn.Dropout(config.dropout)


    def forward(self,
            input_ids: torch.Tensor,
            position_embedding: Tuple[torch.Tensor, torch.Tensor],
            use_cache: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = input_ids.shape

        query, key, value = self.W_query(input_ids), self.W_key(input_ids), self.W_value(input_ids)

        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        # RoPE
        cos, sin = position_embedding
        query, key = apply_rotary_pos_emb(query, key, cos[:seq_len], sin[:seq_len])
        # 转置矩阵: [batch_size, seq_length, num_heads, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        # 计算权重参数
        attention_scores = query @ key.transpose(2, 3) / math.sqrt(self.head_dim)
        # 添加数值稳定性检查
        attention_scores = torch.clamp(attention_scores, min=-1000, max=1000)
        # 因果注意力掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=1).bool()
        dtype = attention_scores.dtype
        min_value = torch.finfo(dtype).min
        attention_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), min_value)

        attention_weight = torch.softmax(attention_scores.float(), dim=-1).type_as(query)
        # 添加数值稳定性检查
        attention_weight = torch.clamp(attention_weight, min=0, max=1)
        attention_weight = self.attention_dropout(attention_weight)
        context_vector = attention_weight @ value

        context_vector = context_vector.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context_vector)
        output = self.output_dropout(output)
        return output


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

def compute_rope_freq(dim: int, max_length: int = (32 * 1024), base: int = 1e5):
    freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    max_position = torch.arange(max_length, dtype=freq.dtype, device=freq.device)[:, torch.newaxis]
    freq_matrix = max_position * freq
    freq_cos = torch.cat([torch.cos(freq_matrix), torch.cos(freq_matrix)], dim=-1)
    freq_sin = torch.cat([torch.sin(freq_matrix), torch.sin(freq_matrix)], dim=-1)
    return freq_cos, freq_sin