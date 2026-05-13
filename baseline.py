import argparse
import json
from pathlib import Path

import numpy as np

from src.data import find_trace_files, load_input_samples, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-dir", default="../题目材料")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="outputs/baseline_submission.jsonl")
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args()

    totals = None
    for path in find_trace_files(args.material_dir):
        for sample in read_jsonl(path):
            counts = np.asarray(sample["raw_counts"], dtype=np.float64)
            totals = counts if totals is None else totals + counts
    if totals is None:
        raise RuntimeError("No trace data loaded")

    ranking = np.argsort(-totals, axis=1)[:, : args.top_k].astype(int).tolist()
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in load_input_samples(args.input_jsonl):
            f.write(json.dumps({
                "request_id": sample["request_id"],
                "predicted_experts": ranking,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
