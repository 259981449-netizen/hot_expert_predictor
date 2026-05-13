import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data import TraceDataset, load_trace_samples
from src.metrics import evaluate_scores
from src.model import HotExpertPredictor


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate(batch):
    keys = ["input_ids", "attention_mask", "prompt_len", "task_id", "targets"]
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def run_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"], batch["task_id"])
        loss = loss_fn(torch.log_softmax(logits, dim=-1), batch["targets"])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, k_values):
    model.eval()
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    total = 0.0
    all_scores = []
    all_targets = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"], batch["task_id"])
        total += loss_fn(torch.log_softmax(logits, dim=-1), batch["targets"]).item()
        all_scores.append(logits.cpu())
        all_targets.append(batch["targets"].cpu())
    scores = torch.cat(all_scores, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = evaluate_scores(scores, targets, k_values)
    metrics["loss"] = total / max(1, len(loader))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-dir", default="../题目材料")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-samples", type=int, default=None, help="Use a small subset for quick smoke tests.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_trace_samples(args.material_dir)
    if args.limit_samples:
        samples = samples[: args.limit_samples]
    dataset = TraceDataset(samples, max_len=cfg["max_len"], has_labels=True)
    valid_size = max(1, int(len(dataset) * cfg["valid_ratio"]))
    train_size = len(dataset) - valid_size
    train_ds, valid_ds = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)

    device = torch.device(args.device)
    model = HotExpertPredictor(
        vocab_size=cfg["vocab_size"],
        num_layers=cfg["num_layers"],
        num_experts=cfg["num_experts"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_ndcg = -1.0
    stale = 0
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, valid_loader, device, cfg["top_k_values"])
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))

        score = metrics.get("ndcg@16", metrics["loss"] * -1)
        if score > best_ndcg:
            best_ndcg = score
            stale = 0
            torch.save({"config": cfg, "model_state": model.state_dict(), "metrics": metrics}, out_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= cfg["early_stop_patience"]:
                break

    with (out_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
