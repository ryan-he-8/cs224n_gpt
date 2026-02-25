import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from modules.lora import LoRALinear


class CausalSelfAttentionLoRA(nn.Module):
  def __init__(self, config, lora_r=0, lora_alpha=1.0, lora_dropout=0.0, lora_target="qv",
               use_flash_attention=False):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    target = set(list(lora_target))
    use_q = 'q' in target
    use_k = 'k' in target
    use_v = 'v' in target

    self.query = LoRALinear(config.hidden_size, self.all_head_size, r=lora_r,
                            lora_alpha=lora_alpha, lora_dropout=lora_dropout) if use_q else nn.Linear(config.hidden_size, self.all_head_size)
    self.key = LoRALinear(config.hidden_size, self.all_head_size, r=lora_r,
                          lora_alpha=lora_alpha, lora_dropout=lora_dropout) if use_k else nn.Linear(config.hidden_size, self.all_head_size)
    self.value = LoRALinear(config.hidden_size, self.all_head_size, r=lora_r,
                            lora_alpha=lora_alpha, lora_dropout=lora_dropout) if use_v else nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    self.use_flash_attention = use_flash_attention

  def transform(self, x, linear_layer):
    proj = linear_layer(x)
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    _, _, T, D = key.shape

    if self.use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
      dropout_p = self.dropout.p if self.training else 0.0
      try:
        # SDPA dispatches to FlashAttention kernels on supported GPUs/dtypes.
        attention = F.scaled_dot_product_attention(
          query, key, value,
          attn_mask=attention_mask,
          dropout_p=dropout_p,
          is_causal=True,
        )
        return rearrange(attention, 'b h t d -> b t (h d)')
      except RuntimeError:
        # Fall back to the manual implementation if kernel constraints are not met.
        pass

    scores = (query @ key.transpose(-2, -1)) * (D ** -0.5)

    causal_mask = torch.triu(
      torch.ones((T, T), device=scores.device, dtype=torch.bool),
      diagonal=1,
    )[None, None, :, :]
    scores = scores.masked_fill(causal_mask, float('-inf'))

    scores = scores + attention_mask

    norm_scores = scores.softmax(dim=-1)
    norm_scores = self.dropout(norm_scores)
    attention = norm_scores @ value

    return rearrange(attention, 'b h t d -> b t (h d)')

  def forward(self, hidden_states, attention_mask):
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)

    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
