'''
Sonnet generation with experiment runners.

Default run:
  python sonnet_generation.py --use_gpu

4-way tuned-then-final run (full/lora x adam/adamw+cosine):
  python sonnet_generation.py --run_four_way_tuned_then_final --use_gpu
'''

import argparse
import copy
import csv
import itertools
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import GPT2Tokenizer

from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from models.gpt2_lora import GPT2ModelLoRA
from optimizer import AdamW

TQDM_DISABLE = False


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def parse_float_list(value):
  return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value):
  return [int(x.strip()) for x in value.split(",") if x.strip()]


def maybe_subset_dataset(dataset, subset_size, seed, name):
  if subset_size <= 0 or subset_size >= len(dataset):
    return dataset
  idx = list(range(len(dataset)))
  rng = random.Random(seed)
  rng.shuffle(idx)
  keep = idx[:subset_size]
  print(f"Using {len(keep)}/{len(dataset)} examples for {name} (subset mode).")
  return Subset(dataset, keep)


class SonnetGPT(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

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

    # Explicit LM head for language modeling.
    self.output_head = nn.Linear(args.d, self.tokenizer.vocab_size, bias=False)
    with torch.no_grad():
      self.output_head.weight.copy_(self.gpt.word_embedding.weight.data)

    if args.use_lora:
      for p in self.gpt.parameters():
        p.requires_grad = False
      for name, p in self.gpt.named_parameters():
        if "lora_" in name:
          p.requires_grad = True
      for p in self.output_head.parameters():
        p.requires_grad = True
    else:
      for p in self.gpt.parameters():
        p.requires_grad = True
      for p in self.output_head.parameters():
        p.requires_grad = True

  def forward(self, input_ids, attention_mask):
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    return self.output_head(outputs['last_hidden_state'])

  def get_device(self):
    return next(self.parameters()).device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      logits_seq = self.forward(token_ids, attention_mask)
      logits_last = logits_seq[:, -1, :] / max(temperature, 1e-5)

      probs = torch.softmax(logits_last, dim=-1)
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
      top_p_mask[..., 0] = True
      filtered_probs = sorted_probs * top_p_mask
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1)

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def count_parameters(model):
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total, trainable


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
      progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
      progress = min(max(progress, 0.0), 1.0)
      return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
  return optimizer, scheduler


@torch.no_grad()
def eval_lm_loss(dataloader, model, device):
  model.eval()
  total_loss = 0.0
  total_tokens = 0
  for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
    b_ids = batch['token_ids'].to(device)
    b_mask = batch['attention_mask'].to(device)
    logits = model(b_ids, b_mask)
    logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
    labels = b_ids[:, 1:].contiguous().flatten()
    loss = F.cross_entropy(logits, labels, reduction='sum')
    total_loss += loss.item()
    total_tokens += labels.numel()
  avg_loss = total_loss / max(total_tokens, 1)
  return avg_loss


def make_dataloaders(args):
  train_dataset_base = SonnetsDataset(args.sonnet_path)
  dev_dataset_base = SonnetsDataset(args.held_out_dev_sonnet_path)
  held_out_dataset = SonnetsDataset(args.held_out_sonnet_path)

  train_dataset = maybe_subset_dataset(train_dataset_base, args.train_subset_size, args.seed, "train")
  dev_dataset = maybe_subset_dataset(dev_dataset_base, args.dev_subset_size, args.seed + 1, "dev")

  collate_fn = train_dataset_base.collate_fn
  train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
  dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)
  return train_loader, dev_loader, held_out_dataset


def train_one_config(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  args = add_arguments(args)
  train_loader, dev_loader, held_out_dataset = make_dataloaders(args)

  model = SonnetGPT(args).to(device)
  total_params, trainable_params = count_parameters(model)
  print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")

  total_steps = args.epochs * max(len(train_loader), 1)
  optimizer, scheduler = build_optimizer_and_scheduler(args, model, total_steps)

  best_dev_loss = float('inf')
  best_epoch = -1
  epochs_no_improve = 0
  history = []

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    num_batches = 0
    for batch in tqdm(train_loader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].contiguous().flatten()
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      if scheduler is not None:
        scheduler.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / max(num_batches, 1)
    dev_loss = eval_lm_loss(dev_loader, model, device)
    dev_ppl = float(np.exp(min(dev_loss, 20)))

    if dev_loss < best_dev_loss:
      best_dev_loss = dev_loss
      best_epoch = epoch
      epochs_no_improve = 0
      save_model(model, optimizer, args, args.filepath)
    else:
      epochs_no_improve += 1

    lr_now = optimizer.param_groups[0]["lr"]
    history.append({
      "epoch": epoch,
      "train_loss": float(train_loss),
      "dev_loss": float(dev_loss),
      "dev_ppl": float(dev_ppl),
      "lr": float(lr_now),
    })
    print(f"Epoch {epoch}: train loss {train_loss:.4f}, dev loss {dev_loss:.4f}, dev ppl {dev_ppl:.2f}")

    if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
      print(f"Early stopping at epoch {epoch}: no dev-loss improvement for {args.early_stop_patience} epochs.")
      break

  return {
    "history": history,
    "best_dev_loss": float(best_dev_loss),
    "best_dev_ppl": float(np.exp(min(best_dev_loss, 20))),
    "best_epoch": best_epoch,
    "total_params": int(total_params),
    "trainable_params": int(trainable_params),
    "held_out_count": len(held_out_dataset),
  }


@torch.no_grad()
def generate_submission_sonnets(args, checkpoint_path):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(checkpoint_path, weights_only=False)
  model = SonnetGPT(saved['args']).to(device)
  model.load_state_dict(saved['model'])
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    generated_sonnets.append((sonnet_id, f'{decoded_output}\n\n'))

  with open(args.sonnet_out, "w+") as f:
    f.write("--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def write_csv(rows, path):
  if not rows:
    return
  fields = list(rows[0].keys())
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def plot_loss_curves(history_rows, out_path):
  plt.figure(figsize=(12, 5))
  ax1 = plt.subplot(1, 2, 1)
  ax2 = plt.subplot(1, 2, 2)
  run_names = sorted({r["run_name"] for r in history_rows})
  for run_name in run_names:
    rows = sorted([r for r in history_rows if r["run_name"] == run_name], key=lambda x: x["epoch"])
    ep = [r["epoch"] for r in rows]
    tr = [r["train_loss"] for r in rows]
    dv = [r["dev_loss"] for r in rows]
    ax1.plot(ep, tr, marker="o", label=run_name)
    ax2.plot(ep, dv, marker="o", label=run_name)
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


def plot_performance(summary_rows, out_path):
  names = [r["run_name"] for r in summary_rows]
  x = np.arange(len(names))
  width = 0.36
  ppl = [r["best_dev_ppl"] for r in summary_rows]
  loss = [r["best_dev_loss"] for r in summary_rows]
  plt.figure(figsize=(11, 5))
  plt.bar(x - width / 2, loss, width, label="Best Dev Loss")
  plt.bar(x + width / 2, ppl, width, label="Best Dev Perplexity")
  plt.xticks(x, names, rotation=20)
  plt.ylabel("Metric (lower is better)")
  plt.title("Sonnet LM Performance Comparison")
  plt.grid(axis="y", alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_path, dpi=220)
  plt.close()


def plot_param_eff(summary_rows, out_path):
  plt.figure(figsize=(7.5, 5))
  for row in summary_rows:
    marker = "o" if row.get("use_lora", False) else "s"
    plt.scatter(row["trainable_params"], row["best_dev_loss"], s=80, marker=marker)
    plt.annotate(row["run_name"], (row["trainable_params"], row["best_dev_loss"]), fontsize=8)
  plt.xscale("log")
  plt.xlabel("Trainable Parameters (log scale)")
  plt.ylabel("Best Dev Loss")
  plt.title("Parameter Efficiency vs Loss")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(out_path, dpi=220)
  plt.close()


def get_four_way_base_configs():
  return [
    {"use_lora": False, "optim_algorithm": "adam", "use_scheduler": False, "scheduler_type": "linear"},
    {"use_lora": False, "optim_algorithm": "adamw", "use_scheduler": True, "scheduler_type": "cosine"},
    {"use_lora": True, "optim_algorithm": "adam", "use_scheduler": False, "scheduler_type": "linear"},
    {"use_lora": True, "optim_algorithm": "adamw", "use_scheduler": True, "scheduler_type": "cosine"},
  ]


def format_run_name(config):
  if config.get("use_lora", False):
    mode = "lora"
  else:
    mode = "full"
  opt = f'{config["optim_algorithm"]}-cosine' if config["use_scheduler"] else config["optim_algorithm"]
  return f"{mode}-{opt}"


def apply_experiment_defaults(args):
  if args.train_subset_size <= 0:
    args.train_subset_size = 200
  if args.dev_subset_size <= 0:
    args.dev_subset_size = 80
  if args.epochs > 5:
    args.epochs = 5
  return args


def sample_candidates(candidates, max_trials, seed):
  if len(candidates) <= max_trials:
    return candidates
  rng = random.Random(seed)
  return rng.sample(candidates, max_trials)


def build_candidates(args, base_config, method_index):
  if base_config.get("use_lora", False):
    lr_vals = parse_float_list(args.tune_lora_lrs)
    r_vals = parse_int_list(args.tune_lora_rs)
    a_vals = parse_float_list(args.tune_lora_alphas)
    d_vals = parse_float_list(args.tune_lora_dropouts)
  else:
    lr_vals = parse_float_list(args.tune_full_lrs)
    r_vals, a_vals, d_vals = [args.lora_r], [args.lora_alpha], [args.lora_dropout]

  if base_config["optim_algorithm"] == "adamw":
    wd_vals = parse_float_list(args.tune_weight_decays)
    wu_vals = parse_int_list(args.tune_warmup_steps)
  else:
    wd_vals = [args.weight_decay]
    wu_vals = [args.warmup_steps]

  candidates = []
  for lr, wd, wu, r, a, d in itertools.product(lr_vals, wd_vals, wu_vals, r_vals, a_vals, d_vals):
    candidates.append({
      **base_config,
      "lr": lr,
      "weight_decay": wd,
      "warmup_steps": wu,
      "lora_r": r,
      "lora_alpha": a,
      "lora_dropout": d,
    })

  candidates = sample_candidates(candidates, args.max_tuning_trials_per_approach, args.seed + 1000 + method_index)
  for i, c in enumerate(candidates):
    c["run_name"] = f"{format_run_name(base_config)}-trial{i + 1}"
  return candidates


def run_config_set(args, configs, output_dir, run_prefix="run", generate_plots=True):
  os.makedirs(output_dir, exist_ok=True)
  summary_rows, history_rows = [], []

  for i, cfg in enumerate(configs):
    run_args = copy.deepcopy(args)
    run_args.use_lora = cfg.get("use_lora", False)
    run_args.optim_algorithm = cfg["optim_algorithm"]
    run_args.use_scheduler = cfg["use_scheduler"]
    run_args.scheduler_type = cfg["scheduler_type"]
    run_args.lr = float(cfg.get("lr", run_args.lr))
    run_args.weight_decay = float(cfg.get("weight_decay", run_args.weight_decay))
    run_args.warmup_steps = int(cfg.get("warmup_steps", run_args.warmup_steps))
    run_args.lora_r = int(cfg.get("lora_r", run_args.lora_r))
    run_args.lora_alpha = float(cfg.get("lora_alpha", run_args.lora_alpha))
    run_args.lora_dropout = float(cfg.get("lora_dropout", run_args.lora_dropout))

    run_name = cfg.get("run_name", format_run_name(cfg))
    run_args.filepath = os.path.join(output_dir, f"{run_prefix}_{i + 1}_{run_name}.pt")
    seed_everything(run_args.seed)
    print("\n" + "=" * 80)
    print(f"Running {i + 1}/{len(configs)}: {run_name}")
    print("=" * 80)

    train_info = train_one_config(run_args)
    summary_rows.append({
      "run_id": i + 1,
      "run_name": run_name,
      "use_lora": bool(run_args.use_lora),
      "optim_algorithm": run_args.optim_algorithm,
      "use_scheduler": bool(run_args.use_scheduler),
      "scheduler_type": run_args.scheduler_type,
      "train_subset_size": run_args.train_subset_size,
      "dev_subset_size": run_args.dev_subset_size,
      "epochs": run_args.epochs,
      "lr": run_args.lr,
      "weight_decay": run_args.weight_decay,
      "warmup_steps": run_args.warmup_steps,
      "lora_r": run_args.lora_r if run_args.use_lora else "",
      "lora_alpha": run_args.lora_alpha if run_args.use_lora else "",
      "lora_dropout": run_args.lora_dropout if run_args.use_lora else "",
      "best_dev_loss": train_info["best_dev_loss"],
      "best_dev_ppl": train_info["best_dev_ppl"],
      "best_epoch": train_info["best_epoch"],
      "total_params": train_info["total_params"],
      "trainable_params": train_info["trainable_params"],
      "checkpoint_path": run_args.filepath,
    })
    for h in train_info["history"]:
      history_rows.append({"run_id": i + 1, "run_name": run_name, **h})

  write_csv(summary_rows, os.path.join(output_dir, "experiment_summary.csv"))
  write_csv(history_rows, os.path.join(output_dir, "experiment_history.csv"))
  with open(os.path.join(output_dir, "experiment_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)

  if generate_plots and history_rows:
    plot_loss_curves(history_rows, os.path.join(output_dir, "loss_curves.png"))
  if generate_plots and summary_rows:
    plot_performance(summary_rows, os.path.join(output_dir, "performance_comparison.png"))
    plot_param_eff(summary_rows, os.path.join(output_dir, "parameter_efficiency.png"))
  return summary_rows, history_rows


def pick_best_row(rows, metric):
  if metric == "best_dev_ppl":
    return min(rows, key=lambda r: float(r["best_dev_ppl"]))
  return min(rows, key=lambda r: float(r["best_dev_loss"]))


def run_four_way_tuned_then_final(args):
  args = apply_experiment_defaults(args)
  base_configs = get_four_way_base_configs()
  root = os.path.join(args.experiment_dir, "four_way_tuned_then_final")
  tuning_root = os.path.join(root, "phase1_tuning")
  os.makedirs(tuning_root, exist_ok=True)

  tuning_args = copy.deepcopy(args)
  if args.tuning_epochs > 0:
    tuning_args.epochs = args.tuning_epochs

  tuning_rows = []
  best_cfg_map = {}
  for method_idx, base in enumerate(base_configs):
    method_name = format_run_name(base)
    method_dir = os.path.join(tuning_root, f"method_{method_idx + 1}_{method_name}")
    candidates = build_candidates(tuning_args, base, method_idx)
    print(f"\nTuning {method_name}: {len(candidates)} trial(s)")
    summary_rows, _ = run_config_set(tuning_args, candidates, method_dir, run_prefix="trial", generate_plots=False)
    for row in summary_rows:
      row["phase"] = "tuning"
      row["method_name"] = method_name
      tuning_rows.append(row)
    best = pick_best_row(summary_rows, args.tuning_metric)
    best_cfg_map[method_name] = {
      "lr": best["lr"],
      "weight_decay": best["weight_decay"],
      "warmup_steps": best["warmup_steps"],
      "lora_r": best["lora_r"] if best["lora_r"] != "" else args.lora_r,
      "lora_alpha": best["lora_alpha"] if best["lora_alpha"] != "" else args.lora_alpha,
      "lora_dropout": best["lora_dropout"] if best["lora_dropout"] != "" else args.lora_dropout,
      "best_dev_loss": best["best_dev_loss"],
      "best_dev_ppl": best["best_dev_ppl"],
      "run_name": best["run_name"],
    }

  write_csv(tuning_rows, os.path.join(tuning_root, "all_tuning_trials.csv"))
  with open(os.path.join(tuning_root, "best_configs.json"), "w") as f:
    json.dump(best_cfg_map, f, indent=2)

  final_args = copy.deepcopy(args)
  final_args.train_subset_size = 0
  final_args.dev_subset_size = 0
  if args.final_epochs > 0:
    final_args.epochs = args.final_epochs

  final_configs = []
  for base in base_configs:
    method_name = format_run_name(base)
    best = best_cfg_map[method_name]
    final_configs.append({
      **base,
      "run_name": f"{method_name}-final",
      "lr": float(best["lr"]),
      "weight_decay": float(best["weight_decay"]),
      "warmup_steps": int(best["warmup_steps"]),
      "lora_r": int(float(best["lora_r"])) if str(best["lora_r"]) != "" else args.lora_r,
      "lora_alpha": float(best["lora_alpha"]) if str(best["lora_alpha"]) != "" else args.lora_alpha,
      "lora_dropout": float(best["lora_dropout"]) if str(best["lora_dropout"]) != "" else args.lora_dropout,
    })

  final_root = os.path.join(root, "phase2_final_full")
  os.makedirs(final_root, exist_ok=True)
  seeds = parse_int_list(args.final_seeds)
  all_rows = []
  for seed in seeds:
    seed_args = copy.deepcopy(final_args)
    seed_args.seed = seed
    seed_dir = os.path.join(final_root, f"seed_{seed}")
    summary_rows, _ = run_config_set(seed_args, final_configs, seed_dir, run_prefix=f"final_s{seed}", generate_plots=True)
    for row in summary_rows:
      row["seed"] = seed
      row["phase"] = "final_full"
    all_rows.extend(summary_rows)

  write_csv(all_rows, os.path.join(final_root, "final_summary_all_seeds.csv"))
  grouped = {}
  for row in all_rows:
    grouped.setdefault(row["run_name"], []).append(row)
  agg_rows = []
  for run_name, rows in grouped.items():
    losses = [float(r["best_dev_loss"]) for r in rows]
    ppls = [float(r["best_dev_ppl"]) for r in rows]
    agg_rows.append({
      "run_name": run_name,
      "num_seeds": len(rows),
      "mean_best_dev_loss": float(np.mean(losses)),
      "std_best_dev_loss": float(np.std(losses)),
      "mean_best_dev_ppl": float(np.mean(ppls)),
      "std_best_dev_ppl": float(np.std(ppls)),
    })
  agg_rows = sorted(agg_rows, key=lambda r: r["mean_best_dev_loss"])
  write_csv(agg_rows, os.path.join(final_root, "final_summary_aggregated.csv"))

  print("\nFour-way tuning + final artifacts:")
  print(f"- Tuning trials: {os.path.join(tuning_root, 'all_tuning_trials.csv')}")
  print(f"- Tuning best configs: {os.path.join(tuning_root, 'best_configs.json')}")
  print(f"- Final per-seed summary: {os.path.join(final_root, 'final_summary_all_seeds.csv')}")
  print(f"- Final aggregated summary: {os.path.join(final_root, 'final_summary_aggregated.csv')}")


def train(args):
  run_args = copy.deepcopy(args)
  run_args.filepath = f'{run_args.epochs}-{run_args.lr}-sonnet.pt'
  run_args.train_subset_size = 0
  run_args.dev_subset_size = 0
  info = train_one_config(run_args)
  print(f"Best dev loss: {info['best_dev_loss']:.4f}, best dev ppl: {info['best_dev_ppl']:.2f}")
  generate_submission_sonnets(run_args, run_args.filepath)


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--held_out_dev_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--top_p", type=float, default=0.9)

  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--weight_decay", type=float, default=0.0)
  parser.add_argument("--optim_algorithm", type=str, choices=["adam", "adamw"], default="adam")
  parser.add_argument("--optimizer_type", type=str, choices=["custom", "torch"], default="custom")
  parser.add_argument("--use_scheduler", action='store_true')
  parser.add_argument("--scheduler_type", type=str, choices=["linear", "cosine"], default="linear")
  parser.add_argument("--warmup_steps", type=int, default=0)
  parser.add_argument("--early_stop_patience", type=int, default=3)

  parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  parser.add_argument("--use_lora", action='store_true')
  parser.add_argument("--lora_r", type=int, default=8)
  parser.add_argument("--lora_alpha", type=float, default=16.0)
  parser.add_argument("--lora_dropout", type=float, default=0.05)
  parser.add_argument("--lora_target", type=str, default="qv", choices=["q", "k", "v", "qk", "qv", "kv", "qkv"])
  parser.add_argument("--use_flash_attention", action='store_true')

  parser.add_argument("--train_subset_size", type=int, default=0)
  parser.add_argument("--dev_subset_size", type=int, default=0)

  parser.add_argument("--run_four_way_experiments", action='store_true',
                      help="Run 4-way base matrix once with shared hyperparameters.")
  parser.add_argument("--run_four_way_tuned_then_final", action='store_true',
                      help="Tune per method, then final full-data run for tuned-best configs (no phase-2 controlled).")
  parser.add_argument("--experiment_dir", type=str, default="results/sonnet_experiments")

  parser.add_argument("--tuning_metric", type=str, choices=["best_dev_loss", "best_dev_ppl"], default="best_dev_loss")
  parser.add_argument("--max_tuning_trials_per_approach", type=int, default=8)
  parser.add_argument("--tuning_epochs", type=int, default=0)
  parser.add_argument("--final_epochs", type=int, default=20)
  parser.add_argument("--final_seeds", type=str, default="11711")

  parser.add_argument("--tune_full_lrs", type=str, default="1e-5,2e-5,3e-5")
  parser.add_argument("--tune_lora_lrs", type=str, default="5e-5,1e-4,2e-4,3e-4")
  parser.add_argument("--tune_weight_decays", type=str, default="0.0,0.01,0.05")
  parser.add_argument("--tune_warmup_steps", type=str, default="0,10,20,50")
  parser.add_argument("--tune_lora_rs", type=str, default="4,8,16")
  parser.add_argument("--tune_lora_alphas", type=str, default="16,32")
  parser.add_argument("--tune_lora_dropouts", type=str, default="0.0,0.05,0.1")
  return parser.parse_args()


def add_arguments(args):
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
  seed_everything(args.seed)
  if args.run_four_way_tuned_then_final:
    run_four_way_tuned_then_final(args)
  elif args.run_four_way_experiments:
    run_args = apply_experiment_defaults(copy.deepcopy(args))
    run_config_set(run_args, get_four_way_base_configs(), run_args.experiment_dir, run_prefix="fourway", generate_plots=True)
  else:
    train(args)
