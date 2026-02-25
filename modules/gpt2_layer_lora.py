from torch import nn
import torch.nn.functional as F

from modules.attention_lora import CausalSelfAttentionLoRA


class GPT2LayerLoRA(nn.Module):
  def __init__(self, config, lora_r=0, lora_alpha=1.0, lora_dropout=0.0, lora_target="qv",
               use_flash_attention=False):
    super().__init__()
    self.self_attention = CausalSelfAttentionLoRA(
      config,
      lora_r=lora_r,
      lora_alpha=lora_alpha,
      lora_dropout=lora_dropout,
      lora_target=lora_target,
      use_flash_attention=use_flash_attention,
    )
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    transformed = dense_layer(output)
    transformed = dropout(transformed)
    return input + transformed

  def forward(self, hidden_states, attention_mask):
    hidden_norm = self.attention_layer_norm(hidden_states)
    attention = self.self_attention(hidden_norm, attention_mask)
    attention_processed = self.add(hidden_states, attention, self.attention_dense, self.attention_dropout)

    attention_processed_norm = self.out_layer_norm(attention_processed)
    ffn_output = self.interm_af(self.interm_dense(attention_processed_norm))
    output = self.add(attention_processed, ffn_output, self.out_dense, self.out_dropout)

    return output
