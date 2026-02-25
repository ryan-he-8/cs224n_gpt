import math
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
  def __init__(self, in_features, out_features, r=0, lora_alpha=1.0, lora_dropout=0.0, bias=True):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.r = int(r)
    self.lora_alpha = float(lora_alpha)
    self.scaling = self.lora_alpha / self.r if self.r > 0 else 0.0

    self.base = nn.Linear(in_features, out_features, bias=bias)

    if self.r > 0:
      self.lora_A = nn.Linear(in_features, self.r, bias=False)
      self.lora_B = nn.Linear(self.r, out_features, bias=False)
      self.lora_dropout = nn.Dropout(p=float(lora_dropout))
      self.reset_lora_parameters()
    else:
      self.lora_A = None
      self.lora_B = None
      self.lora_dropout = None

  def reset_lora_parameters(self):
    if self.r > 0:
      nn.init.normal_(self.lora_A.weight, std=0.02)
      nn.init.zeros_(self.lora_B.weight)

  def forward(self, x):
    result = self.base(x)
    if self.r > 0:
      lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
      result = result + lora_out * self.scaling
    return result

  @property
  def weight(self):
    return self.base.weight

  @property
  def bias(self):
    return self.base.bias
