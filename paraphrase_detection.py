'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import LambdaLR

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_test_paraphrase
from models.gpt2 import GPT2Model
from models.gpt2_lora import GPT2ModelLoRA

from optimizer import AdamW

TQDM_DISABLE = False
NO_TOKEN_ID = 3919
YES_TOKEN_ID = 8505


def labels_to_class_ids(labels: torch.Tensor) -> torch.Tensor:
  """
  Normalize paraphrase labels to class ids in {0,1}.
  Supports:
    - direct class ids [0,1]
    - token ids [3919 ("no"), 8505 ("yes")] potentially shaped [B, 1]
  """
  if labels.dim() > 1:
    labels = labels[:, 0]
  labels = labels.long().flatten()

  if labels.numel() == 0:
    return labels

  is_class_id = (labels == 0) | (labels == 1)
  if torch.all(is_class_id):
    return labels

  is_token_id = (labels == NO_TOKEN_ID) | (labels == YES_TOKEN_ID)
  if not torch.all(is_token_id):
    bad = labels[~is_token_id]
    raise ValueError(
      f"Unexpected paraphrase labels. Expected only 0/1 or {NO_TOKEN_ID}/{YES_TOKEN_ID}, "
      f"but found values like {bad[:8].tolist()}"
    )

  return (labels == YES_TOKEN_ID).long()


@torch.no_grad()
def model_eval_paraphrase_local(dataloader, model, device):
  model.eval()
  y_true, y_pred, sent_ids = [], [], []
  total_loss = 0.0
  total_examples = 0
  for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
    b_ids = batch['token_ids'].to(device)
    b_mask = batch['attention_mask'].to(device)
    labels = labels_to_class_ids(batch['labels']).to(device)

    logits_t = model(b_ids, b_mask)
    loss = F.cross_entropy(logits_t, labels, reduction='mean')
    batch_size = labels.shape[0]
    total_loss += loss.item() * batch_size
    total_examples += batch_size

    logits = logits_t.cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_true.extend(labels.cpu().numpy().flatten())
    y_pred.extend(preds)
    sent_ids.extend(batch['sent_ids'])

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)
  avg_loss = total_loss / max(total_examples, 1)
  return acc, f1, y_pred, y_true, sent_ids, avg_loss

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    if args.use_lora:
      self.gpt = GPT2ModelLoRA.from_pretrained(
        model=args.model_size,
        d=args.d,
        l=args.l,
        num_heads=args.num_heads,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        use_flash_attention=args.use_flash_attention,
      )
    else:
      self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    if args.use_lora:
      # Freeze base model; train only LoRA adapters and the task head.
      for param in self.gpt.parameters():
        param.requires_grad = False
      for name, param in self.gpt.named_parameters():
        if "lora_" in name:
          param.requires_grad = True
    else:
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    logits = self.paraphrase_detection_head(outputs['last_token'])
    return logits



def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def build_optimizer_and_scheduler(args, model, total_training_steps):
  params = filter(lambda p: p.requires_grad, model.parameters())
  if args.optimizer_type == "torch":
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
  else:
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

  scheduler = None
  if args.use_scheduler:
    warmup_steps = max(args.warmup_steps, 0)
    total_steps = max(total_training_steps, 1)

    def lr_lambda(current_step):
      if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))

      if args.scheduler_type == "linear":
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - current_step) / float(decay_steps))

      # cosine decay
      progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
      progress = min(max(progress, 0.0), 1.0)
      return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

  return optimizer, scheduler


def maybe_subsample(data, subset_size, seed, name):
  if subset_size <= 0 or subset_size >= len(data):
    return data
  rng = random.Random(seed)
  idx = list(range(len(data)))
  rng.shuffle(idx)
  keep = idx[:subset_size]
  sampled = [data[i] for i in keep]
  print(f"Using {len(sampled)}/{len(data)} examples for {name} (subset mode).")
  return sampled


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)
  para_train_data = maybe_subsample(para_train_data, args.train_subset_size, args.seed, "train")
  para_dev_data = maybe_subsample(para_dev_data, args.dev_subset_size, args.seed + 1, "dev")

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  total_training_steps = args.epochs * len(para_train_dataloader)
  optimizer, scheduler = build_optimizer_and_scheduler(args, model, total_training_steps)
  best_dev_loss = float('inf')
  epochs_without_improve = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      labels = labels_to_class_ids(batch['labels'])
      if labels.min().item() < 0 or labels.max().item() >= 2:
        raise ValueError(f"Invalid class labels for paraphrase task: min={labels.min().item()}, max={labels.max().item()}")
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      if scheduler is not None:
        scheduler.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_, dev_loss = model_eval_paraphrase_local(para_dev_dataloader, model, device)

    if dev_loss < best_dev_loss:
      best_dev_loss = dev_loss
      epochs_without_improve = 0
      save_model(model, optimizer, args, args.filepath)
    else:
      epochs_without_improve += 1

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, val loss :: {dev_loss :.3f}, dev acc :: {dev_acc :.3f}")
    if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
      print(f"Early stopping at epoch {epoch}: no val-loss improvement for {args.early_stop_patience} epoch(s).")
      break


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids, _ = model_eval_paraphrase_local(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  # Model predicts class ids {0,1}; submission files require GPT token ids for "no"/"yes".
  def class_to_token_id(pred):
    return YES_TOKEN_ID if int(pred) == 1 else NO_TOKEN_ID

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {class_to_token_id(s)} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {class_to_token_id(s)} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--train_subset_size", type=int, default=0,
                      help="If >0, train only on this many examples (for cheap hyperparameter tuning).")
  parser.add_argument("--dev_subset_size", type=int, default=0,
                      help="If >0, evaluate only on this many dev examples (for cheap hyperparameter tuning).")

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay used by AdamW.")
  parser.add_argument("--optimizer_type", type=str, choices=["custom", "torch"], default="custom",
                      help="AdamW implementation to use.")
  parser.add_argument("--use_scheduler", action='store_true',
                      help="Enable LR scheduler with warmup.")
  parser.add_argument("--scheduler_type", type=str, choices=["linear", "cosine"], default="linear",
                      help="Scheduler decay shape after warmup.")
  parser.add_argument("--warmup_steps", type=int, default=0,
                      help="Number of optimizer steps for linear warmup.")
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  parser.add_argument("--use_lora", action='store_true', help="Enable LoRA adapters in GPT-2 attention layers.")
  parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
  parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA scaling factor.")
  parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
  parser.add_argument("--lora_target", type=str, default="qv", choices=["q", "k", "v", "qk", "qv", "kv", "qkv"],
                      help="Which attention projections to apply LoRA to.")
  parser.add_argument("--use_flash_attention", action='store_true',
                      help="Use PyTorch scaled_dot_product_attention (FlashAttention-backed when supported) in LoRA GPT-2.")
  parser.add_argument("--early_stop_patience", type=int, default=3,
                      help="Stop training after this many epochs without val-loss improvement. Set <=0 to disable.")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
