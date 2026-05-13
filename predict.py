import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import TraceDataset, load_input_samples
from src.model import HotExpertPredictor


def collate(batch):
    tensor_keys = ["input_ids", "attention_mask", "prompt_len", "task_id"]
    result = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    result["request_id"] = [item["request_id"] for item in batch]
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--output-jsonl", default="outputs/submission.jsonl")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = checkpoint["config"]
    top_k = args.top_k or cfg.get("default_output_top_k", cfg["num_experts"])
    top_k = min(top_k, cfg["num_experts"])

    model = HotExpertPredictor(
        vocab_size=cfg["vocab_size"],
        num_layers=cfg["num_layers"],
        num_experts=cfg["num_experts"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    samples = load_input_samples(args.input_jsonl)
    dataset = TraceDataset(samples, max_len=cfg["max_len"], has_labels=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for batch in loader:
            request_ids = batch.pop("request_id")
            batch = {k: v.to(args.device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"], batch["prompt_len"], batch["task_id"])
            rankings = torch.topk(logits, k=top_k, dim=-1).indices.cpu().tolist()
            for request_id, predicted_experts in zip(request_ids, rankings):
                f.write(json.dumps({
                    "request_id": request_id,
                    "predicted_experts": predicted_experts,
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
