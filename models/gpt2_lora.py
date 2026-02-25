import torch
from torch import nn
from transformers import GPT2Model as OpenAIGPT2Model

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer_lora import GPT2LayerLoRA
from modules.lora import LoRALinear
from utils import get_extended_attention_mask


class GPT2ModelLoRA(GPTPreTrainedModel):
  """
  GPT-2 with optional LoRA adapters on attention projections.
  """

  def __init__(self, config, lora_r=0, lora_alpha=1.0, lora_dropout=0.0, lora_target="qv",
               use_flash_attention=False):
    super().__init__(config)
    self.config = config
    self.lora_r = lora_r
    self.lora_alpha = lora_alpha
    self.lora_dropout = lora_dropout
    self.lora_target = lora_target
    self.use_flash_attention = use_flash_attention

    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    self.gpt_layers = nn.ModuleList([
      GPT2LayerLoRA(
        config,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target=lora_target,
        use_flash_attention=use_flash_attention,
      ) for _ in range(config.num_hidden_layers)
    ])

    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    inputs_embeds = self.word_embedding(input_ids)

    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)
    return self.embed_dropout(inputs_embeds + pos_embeds)

  def encode(self, hidden_states, attention_mask):
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    for layer_module in self.gpt_layers:
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    embedding_output = self.embed(input_ids=input_ids)

    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
    sequence_output = self.final_layer_norm(sequence_output)

    last_non_pad_idx = attention_mask.sum(dim=1) - 1
    last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

    return {'last_hidden_state': sequence_output, 'last_token': last_token}

  def hidden_state_to_token(self, hidden_state):
    return hidden_state @ self.word_embedding.T

  @classmethod
  def from_pretrained(cls, model='gpt2', d=768, l=12, num_heads=12,
                      lora_r=0, lora_alpha=1.0, lora_dropout=0.0, lora_target="qv",
                      use_flash_attention=False):
    gpt_model = OpenAIGPT2Model.from_pretrained(model).eval()
    our_model = GPT2ModelLoRA(
      GPT2Config(hidden_size=d, num_hidden_layers=l, num_attention_heads=num_heads, intermediate_size=d * 3),
      lora_r=lora_r,
      lora_alpha=lora_alpha,
      lora_dropout=lora_dropout,
      lora_target=lora_target,
      use_flash_attention=use_flash_attention,
    ).eval()

    our_model.word_embedding.load_state_dict(gpt_model.wte.state_dict())
    our_model.pos_embedding.load_state_dict(gpt_model.wpe.state_dict())

    for i in range(l):
      layer = our_model.gpt_layers[i]

      q_weight = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, :d].T
      q_bias = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][:d]
      k_weight = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d:d * 2].T
      k_bias = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d:d * 2]
      v_weight = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d * 2:].T
      v_bias = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d * 2:]

      def load_linear(linear_module, weight, bias):
        if isinstance(linear_module, LoRALinear):
          linear_module.base.weight.data = weight
          linear_module.base.bias.data = bias
        else:
          linear_module.weight.data = weight
          linear_module.bias.data = bias

      load_linear(layer.self_attention.query, q_weight, q_bias)
      load_linear(layer.self_attention.key, k_weight, k_bias)
      load_linear(layer.self_attention.value, v_weight, v_bias)

      layer.attention_dense.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.weight'].T
      layer.attention_dense.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.bias']

      layer.attention_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_1.weight']
      layer.attention_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_1.bias']

      layer.interm_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.weight'].T
      layer.interm_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.bias']
      layer.out_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.weight'].T
      layer.out_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.bias']

      layer.out_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_2.weight']
      layer.out_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_2.bias']

    our_model.final_layer_norm.weight.data = gpt_model.state_dict()['ln_f.weight']
    our_model.final_layer_norm.bias.data = gpt_model.state_dict()['ln_f.bias']

    return our_model
