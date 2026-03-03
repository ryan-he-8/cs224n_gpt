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
import copy
import csv
import itertools
import json
import os
import random
import torch

import matplotlib.pyplot as plt
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
  if args.optim_algorithm == "adam":
    optimizer = torch.optim.Adam(params, lr=args.lr)
  else:
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


def count_parameters(model):
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total, trainable


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
  total_params, trainable_params = count_parameters(model)
  print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")

  total_training_steps = args.epochs * len(para_train_dataloader)
  optimizer, scheduler = build_optimizer_and_scheduler(args, model, total_training_steps)
  best_dev_loss = float('inf')
  epochs_without_improve = 0
  best_epoch = -1
  history = []

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
      best_epoch = epoch
      save_model(model, optimizer, args, args.filepath)
    else:
      epochs_without_improve += 1

    current_lr = optimizer.param_groups[0]["lr"]
    history.append({
      "epoch": epoch,
      "train_loss": float(train_loss),
      "dev_loss": float(dev_loss),
      "dev_acc": float(dev_acc),
      "dev_f1": float(dev_f1),
      "lr": float(current_lr),
    })

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, val loss :: {dev_loss :.3f}, dev acc :: {dev_acc :.3f}")
    if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
      print(f"Early stopping at epoch {epoch}: no val-loss improvement for {args.early_stop_patience} epoch(s).")
      break

  return {
    "history": history,
    "best_dev_loss": float(best_dev_loss),
    "best_epoch": best_epoch,
    "total_params": int(total_params),
    "trainable_params": int(trainable_params),
  }


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
  para_dev_data = maybe_subsample(para_dev_data, args.dev_subset_size, args.seed + 1, "dev(eval)")
  para_test_data = maybe_subsample(para_test_data, args.test_subset_size, args.seed + 2, "test(eval)")

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

  return {
    "dev_acc": float(dev_para_acc),
    "num_dev_predictions": len(dev_para_y_pred),
    "num_test_predictions": len(test_para_y_pred),
    "dev_out_path": args.para_dev_out,
    "test_out_path": args.para_test_out,
  }


def apply_experiment_defaults(args):
  if args.train_subset_size <= 0:
    args.train_subset_size = 2000
  if args.dev_subset_size <= 0:
    args.dev_subset_size = 800
  if args.test_subset_size <= 0:
    args.test_subset_size = 800
  if args.epochs > 5:
    args.epochs = 5
  return args


def get_base_experiment_configs():
  return [
    {"use_lora": False, "optim_algorithm": "adam", "use_scheduler": False, "scheduler_type": "linear"},
    {"use_lora": False, "optim_algorithm": "adamw", "use_scheduler": True, "scheduler_type": "cosine"},
    {"use_lora": True, "optim_algorithm": "adam", "use_scheduler": False, "scheduler_type": "linear"},
    {"use_lora": True, "optim_algorithm": "adamw", "use_scheduler": True, "scheduler_type": "cosine"},
  ]


def format_run_name(config):
  model_label = "lora" if config["use_lora"] else "full"
  opt_label = f'{config["optim_algorithm"]}-cosine' if config["use_scheduler"] else config["optim_algorithm"]
  return f"{model_label}-{opt_label}"


def write_experiment_csv(rows, csv_path):
  if not rows:
    return
  fields = list(rows[0].keys())
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def write_history_csv(rows, csv_path):
  if not rows:
    return
  fields = list(rows[0].keys())
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def plot_loss_curves(history_rows, out_path):
  plt.figure(figsize=(12, 5))
  ax1 = plt.subplot(1, 2, 1)
  ax2 = plt.subplot(1, 2, 2)

  run_names = sorted({row["run_name"] for row in history_rows})
  for run_name in run_names:
    rows = [r for r in history_rows if r["run_name"] == run_name]
    rows = sorted(rows, key=lambda r: r["epoch"])
    epochs = [r["epoch"] for r in rows]
    train_losses = [r["train_loss"] for r in rows]
    dev_losses = [r["dev_loss"] for r in rows]
    ax1.plot(epochs, train_losses, marker="o", label=run_name)
    ax2.plot(epochs, dev_losses, marker="o", label=run_name)

  ax1.set_title("Train Loss")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Loss")
  ax1.grid(True, alpha=0.3)

  ax2.set_title("Dev Loss")
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel("Loss")
  ax2.grid(True, alpha=0.3)
  ax2.legend(loc="best", fontsize=8)

  plt.tight_layout()
  plt.savefig(out_path, dpi=220)
  plt.close()


def plot_performance_bars(summary_rows, out_path):
  names = [r["run_name"] for r in summary_rows]
  x = np.arange(len(names))
  width = 0.36
  dev_acc = [r["dev_acc"] for r in summary_rows]
  best_dev_f1 = [r["best_dev_f1"] for r in summary_rows]

  plt.figure(figsize=(11, 5))
  plt.bar(x - width / 2, dev_acc, width, label="Dev Accuracy")
  plt.bar(x + width / 2, best_dev_f1, width, label="Best Dev F1")
  plt.xticks(x, names, rotation=20)
  plt.ylim(0.0, 1.0)
  plt.ylabel("Score")
  plt.title("Paraphrase Performance Comparison")
  plt.grid(axis="y", alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_path, dpi=220)
  plt.close()


def plot_parameter_efficiency(summary_rows, out_path):
  plt.figure(figsize=(7.5, 5))
  for row in summary_rows:
    marker = "o" if row["use_lora"] else "s"
    plt.scatter(row["trainable_params"], row["best_dev_f1"], s=80, marker=marker, label=row["run_name"])
    plt.annotate(row["run_name"], (row["trainable_params"], row["best_dev_f1"]), fontsize=8)

  plt.xscale("log")
  plt.xlabel("Trainable Parameters (log scale)")
  plt.ylabel("Best Dev F1")
  plt.title("Parameter Efficiency vs Performance")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(out_path, dpi=220)
  plt.close()


def parse_float_list(value):
  return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value):
  return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_run_name(config):
  base = format_run_name(config)
  suffix_parts = []
  if "lr" in config:
    suffix_parts.append(f"lr{config['lr']:.0e}")
  if config.get("use_lora", False):
    suffix_parts.append(f"r{int(config.get('lora_r', 0))}")
    suffix_parts.append(f"a{int(config.get('lora_alpha', 0))}")
    suffix_parts.append(f"d{config.get('lora_dropout', 0.0):.2f}")
  if config.get("optim_algorithm") == "adamw":
    suffix_parts.append(f"wd{config.get('weight_decay', 0.0):.3f}")
  if config.get("use_scheduler", False):
    suffix_parts.append(f"wu{int(config.get('warmup_steps', 0))}")
  if not suffix_parts:
    return base
  return f"{base}-{'-'.join(suffix_parts)}"


def run_config_set(args, experiment_configs, output_dir, run_prefix="run", generate_plots=True):
  os.makedirs(output_dir, exist_ok=True)
  pred_dir = os.path.join(output_dir, "predictions")
  os.makedirs(pred_dir, exist_ok=True)

  summary_rows = []
  history_rows = []
  for idx, config in enumerate(experiment_configs):
    run_args = copy.deepcopy(args)
    run_args.use_lora = config["use_lora"]
    run_args.optim_algorithm = config["optim_algorithm"]
    run_args.use_scheduler = config["use_scheduler"]
    run_args.scheduler_type = config["scheduler_type"]
    if "lr" in config:
      run_args.lr = float(config["lr"])
    if "weight_decay" in config:
      run_args.weight_decay = float(config["weight_decay"])
    if "warmup_steps" in config:
      run_args.warmup_steps = int(config["warmup_steps"])
    if run_args.use_lora:
      if "lora_r" in config:
        run_args.lora_r = int(config["lora_r"])
      if "lora_alpha" in config:
        run_args.lora_alpha = float(config["lora_alpha"])
      if "lora_dropout" in config:
        run_args.lora_dropout = float(config["lora_dropout"])
    run_name = config.get("run_name", build_run_name(config))

    run_args.filepath = os.path.join(output_dir, f"{run_prefix}_{idx + 1}_{run_name}.pt")
    run_args.para_dev_out = os.path.join(pred_dir, f"{run_prefix}_{idx + 1}_{run_name}_dev.csv")
    run_args.para_test_out = os.path.join(pred_dir, f"{run_prefix}_{idx + 1}_{run_name}_test.csv")

    seed_everything(run_args.seed)
    print("\n" + "=" * 80)
    print(f"Running experiment {idx + 1}/{len(experiment_configs)}: {run_name}")
    print("=" * 80)
    train_info = train(run_args)
    eval_info = test(run_args)

    best_dev_f1 = max([r["dev_f1"] for r in train_info["history"]]) if train_info["history"] else 0.0
    summary_rows.append({
      "run_id": idx + 1,
      "run_name": run_name,
      "use_lora": bool(config["use_lora"]),
      "optim_algorithm": config["optim_algorithm"],
      "use_scheduler": bool(config["use_scheduler"]),
      "scheduler_type": config["scheduler_type"],
      "train_subset_size": run_args.train_subset_size,
      "dev_subset_size": run_args.dev_subset_size,
      "test_subset_size": run_args.test_subset_size,
      "epochs": run_args.epochs,
      "lr": run_args.lr,
      "weight_decay": run_args.weight_decay,
      "warmup_steps": run_args.warmup_steps,
      "lora_r": run_args.lora_r if run_args.use_lora else "",
      "lora_alpha": run_args.lora_alpha if run_args.use_lora else "",
      "lora_dropout": run_args.lora_dropout if run_args.use_lora else "",
      "best_dev_loss": train_info["best_dev_loss"],
      "best_epoch": train_info["best_epoch"],
      "best_dev_f1": float(best_dev_f1),
      "dev_acc": eval_info["dev_acc"],
      "total_params": train_info["total_params"],
      "trainable_params": train_info["trainable_params"],
      "checkpoint_path": run_args.filepath,
      "dev_out_path": eval_info["dev_out_path"],
      "test_out_path": eval_info["test_out_path"],
    })

    for epoch_row in train_info["history"]:
      history_rows.append({
        "run_id": idx + 1,
        "run_name": run_name,
        **epoch_row,
      })

  summary_csv = os.path.join(output_dir, "experiment_summary.csv")
  history_csv = os.path.join(output_dir, "experiment_history.csv")
  write_experiment_csv(summary_rows, summary_csv)
  write_history_csv(history_rows, history_csv)
  with open(os.path.join(output_dir, "experiment_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)

  if generate_plots and history_rows:
    plot_loss_curves(history_rows, os.path.join(output_dir, "loss_curves.png"))
  if generate_plots and summary_rows:
    plot_performance_bars(summary_rows, os.path.join(output_dir, "performance_comparison.png"))
    plot_parameter_efficiency(summary_rows, os.path.join(output_dir, "parameter_efficiency.png"))

  print("\nExperiment artifacts:")
  print(f"- {summary_csv}")
  print(f"- {history_csv}")
  if generate_plots:
    print(f"- {os.path.join(output_dir, 'loss_curves.png')}")
    print(f"- {os.path.join(output_dir, 'performance_comparison.png')}")
    print(f"- {os.path.join(output_dir, 'parameter_efficiency.png')}")

  return summary_rows, history_rows


def build_tuning_candidates(args, base_config, method_index):
  lr_values = parse_float_list(args.tune_lora_lrs) if base_config["use_lora"] else parse_float_list(args.tune_full_lrs)
  if base_config["use_lora"]:
    lora_r_values = parse_int_list(args.tune_lora_rs)
    lora_alpha_values = parse_float_list(args.tune_lora_alphas)
    lora_dropout_values = parse_float_list(args.tune_lora_dropouts)
  else:
    lora_r_values = [args.lora_r]
    lora_alpha_values = [args.lora_alpha]
    lora_dropout_values = [args.lora_dropout]

  if base_config["optim_algorithm"] == "adamw":
    weight_decay_values = parse_float_list(args.tune_weight_decays)
  else:
    weight_decay_values = [args.weight_decay]

  if base_config["use_scheduler"]:
    warmup_values = parse_int_list(args.tune_warmup_steps)
  else:
    warmup_values = [args.warmup_steps]

  candidates = []
  for lr, wd, wu, lora_r, lora_alpha, lora_dropout in itertools.product(
      lr_values, weight_decay_values, warmup_values, lora_r_values, lora_alpha_values, lora_dropout_values):
    candidate = {
      **base_config,
      "lr": lr,
      "weight_decay": wd,
      "warmup_steps": wu,
      "lora_r": lora_r,
      "lora_alpha": lora_alpha,
      "lora_dropout": lora_dropout,
    }
    candidates.append(candidate)

  max_trials = max(args.max_tuning_trials_per_approach, 1)
  if len(candidates) > max_trials:
    rng = random.Random(args.seed + 1000 + method_index)
    candidates = rng.sample(candidates, max_trials)

  for i, candidate in enumerate(candidates):
    candidate["run_name"] = f"{format_run_name(base_config)}-trial{i + 1}"
  return candidates


def pick_best_row(rows, metric):
  if metric == "dev_acc":
    key_fn = lambda r: (r["dev_acc"], -r["best_dev_loss"])
  else:
    key_fn = lambda r: (r["best_dev_f1"], -r["best_dev_loss"])
  return max(rows, key=key_fn)


def run_tuned_then_compare(args):
  args = apply_experiment_defaults(args)
  os.makedirs(args.experiment_dir, exist_ok=True)
  base_configs = get_base_experiment_configs()

  tuning_args = copy.deepcopy(args)
  if args.tuning_epochs > 0:
    tuning_args.epochs = args.tuning_epochs

  tuning_root = os.path.join(args.experiment_dir, "phase1_tuning")
  os.makedirs(tuning_root, exist_ok=True)
  tuning_trial_rows = []
  best_config_map = {}

  for method_idx, base_config in enumerate(base_configs):
    method_name = format_run_name(base_config)
    method_dir = os.path.join(tuning_root, f"method_{method_idx + 1}_{method_name}")
    candidates = build_tuning_candidates(tuning_args, base_config, method_idx)
    print(f"\nTuning {method_name}: {len(candidates)} trial(s)")
    trial_summary_rows, _ = run_config_set(
      tuning_args, candidates, method_dir, run_prefix="trial", generate_plots=False
    )
    for row in trial_summary_rows:
      row["phase"] = "tuning"
      row["method_name"] = method_name
      tuning_trial_rows.append(row)

    best_row = pick_best_row(trial_summary_rows, args.tuning_metric)
    best_config_map[method_name] = {
      "lr": best_row["lr"],
      "weight_decay": best_row["weight_decay"],
      "warmup_steps": best_row["warmup_steps"],
      "lora_r": args.lora_r if not base_config["use_lora"] else best_row.get("lora_r", args.lora_r),
      "lora_alpha": args.lora_alpha if not base_config["use_lora"] else best_row.get("lora_alpha", args.lora_alpha),
      "lora_dropout": args.lora_dropout if not base_config["use_lora"] else best_row.get("lora_dropout", args.lora_dropout),
      "selected_metric": best_row[args.tuning_metric],
      "best_dev_f1": best_row["best_dev_f1"],
      "dev_acc": best_row["dev_acc"],
      "best_dev_loss": best_row["best_dev_loss"],
      "run_name": best_row["run_name"],
    }

  write_experiment_csv(tuning_trial_rows, os.path.join(tuning_root, "all_tuning_trials.csv"))
  with open(os.path.join(tuning_root, "best_configs.json"), "w") as f:
    json.dump(best_config_map, f, indent=2)

  compare_args = copy.deepcopy(args)
  if args.comparison_epochs > 0:
    compare_args.epochs = args.comparison_epochs

  controlled_dir = os.path.join(args.experiment_dir, "phase2_controlled_shared")
  controlled_summary, _ = run_config_set(
    compare_args, base_configs, controlled_dir, run_prefix="controlled", generate_plots=True
  )
  for row in controlled_summary:
    row["phase"] = "controlled_shared"

  tuned_best_configs = []
  for base in base_configs:
    method_name = format_run_name(base)
    best_hp = best_config_map[method_name]
    tuned_best_configs.append({
      **base,
      "lr": best_hp["lr"],
      "weight_decay": best_hp["weight_decay"],
      "warmup_steps": best_hp["warmup_steps"],
      "lora_r": best_hp["lora_r"],
      "lora_alpha": best_hp["lora_alpha"],
      "lora_dropout": best_hp["lora_dropout"],
      "run_name": f"{method_name}-tuned",
    })

  tuned_dir = os.path.join(args.experiment_dir, "phase3_tuned_best")
  tuned_summary, _ = run_config_set(
    compare_args, tuned_best_configs, tuned_dir, run_prefix="tuned", generate_plots=True
  )
  for row in tuned_summary:
    row["phase"] = "tuned_best"

  final_rows = controlled_summary + tuned_summary
  write_experiment_csv(final_rows, os.path.join(args.experiment_dir, "final_comparison_summary.csv"))
  print("\nTwo-phase experiment complete.")
  print(f"- Tuning artifacts: {tuning_root}")
  print(f"- Controlled comparison: {controlled_dir}")
  print(f"- Tuned-best comparison: {tuned_dir}")
  print(f"- Final summary: {os.path.join(args.experiment_dir, 'final_comparison_summary.csv')}")


def run_experiments(args):
  args = apply_experiment_defaults(args)
  base_configs = get_base_experiment_configs()
  run_config_set(args, base_configs, args.experiment_dir, run_prefix="run", generate_plots=True)


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
  parser.add_argument("--optim_algorithm", type=str, choices=["adam", "adamw"], default="adam",
                      help="Optimizer family: Adam or AdamW.")
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
  parser.add_argument("--test_subset_size", type=int, default=0,
                      help="If >0, evaluate only on this many test examples.")
  parser.add_argument("--run_experiments", action='store_true',
                      help="Run the 4-way experiment matrix and generate csv + charts.")
  parser.add_argument("--run_tuned_then_compare", action='store_true',
                      help="Phase 1: tune each method. Phase 2: controlled and tuned-best 4-way comparisons.")
  parser.add_argument("--experiment_dir", type=str, default="results/paraphrase_experiments",
                      help="Directory for experiment outputs.")
  parser.add_argument("--tuning_metric", type=str, choices=["best_dev_f1", "dev_acc"], default="best_dev_f1",
                      help="Metric used to select best hyperparameters per method.")
  parser.add_argument("--max_tuning_trials_per_approach", type=int, default=8,
                      help="Cap number of sampled hyperparameter trials per approach.")
  parser.add_argument("--tuning_epochs", type=int, default=0,
                      help="If >0, override epochs during phase-1 tuning.")
  parser.add_argument("--comparison_epochs", type=int, default=0,
                      help="If >0, override epochs for phase-2/3 comparisons.")
  parser.add_argument("--tune_full_lrs", type=str, default="1e-5,2e-5,3e-5",
                      help="Comma-separated LR list for full fine-tuning.")
  parser.add_argument("--tune_lora_lrs", type=str, default="5e-5,1e-4,2e-4",
                      help="Comma-separated LR list for LoRA fine-tuning.")
  parser.add_argument("--tune_weight_decays", type=str, default="0.0,0.01",
                      help="Comma-separated weight decay list for AdamW methods.")
  parser.add_argument("--tune_warmup_steps", type=str, default="0,20,50",
                      help="Comma-separated warmup steps list for scheduled methods.")
  parser.add_argument("--tune_lora_rs", type=str, default="8,16",
                      help="Comma-separated LoRA rank list.")
  parser.add_argument("--tune_lora_alphas", type=str, default="16,32",
                      help="Comma-separated LoRA alpha list.")
  parser.add_argument("--tune_lora_dropouts", type=str, default="0.0,0.1",
                      help="Comma-separated LoRA dropout list.")

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
  if args.run_tuned_then_compare:
    run_tuned_then_compare(args)
  elif args.run_experiments:
    run_experiments(args)
  else:
    train(args)
    test(args)
