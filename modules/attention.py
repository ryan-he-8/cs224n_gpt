import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    '''
    Implemented multi-head attention
    
    key: Key matrix (batch, head, time, dimension)
    query: Query matrix (batch, head, time, dimension)
    value: Value matrix (batch, head, time, dimension)
    attention_mask: attention_mask (batch, 1, 1, time)

    return shape
    '''
    ### YOUR CODE HERE

    B, H, T, D = key.shape

    # Raw attention scores: [B, H, T, T]
    scores = (query @ key.transpose(-2, -1)) * (D ** -0.5)

    # Causal mask for autoregressive decoding.
    causal_mask = torch.triu(
      torch.ones((T, T), device=scores.device, dtype=torch.bool),
      diagonal=1,
    )[None, None, :, :]
    scores = scores.masked_fill(causal_mask, float('-inf'))

    # attention_mask is [B, 1, 1, T] with 0 for keep and -10000 for masked pads.
    # Add directly to scores so it broadcasts over heads and query positions.
    scores = scores + attention_mask

    norm_scores = scores.softmax(dim=-1)
    norm_scores = self.dropout(norm_scores)
    attention = norm_scores @ value  # [B, H, T, D]

    # Merge heads back to hidden size: [B, T, H * D].
    return rearrange(attention, 'b h t d -> b t (h d)')

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
