import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--min-k", type=int, default=16)
    args = parser.parse_args()

    n = 0
    with open(args.submission, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            assert "request_id" in row, f"line {line_no}: missing request_id"
            layers = row.get("predicted_experts")
            assert isinstance(layers, list), f"line {line_no}: predicted_experts must be list"
            assert len(layers) == args.num_layers, f"line {line_no}: wrong layer count"
            for layer_id, experts in enumerate(layers):
                assert len(experts) >= args.min_k, f"line {line_no} layer {layer_id}: too short"
                assert len(experts) == len(set(experts)), f"line {line_no} layer {layer_id}: duplicated expert"
                assert all(isinstance(x, int) and 0 <= x < args.num_experts for x in experts), (
                    f"line {line_no} layer {layer_id}: invalid expert id"
                )
            n += 1
    print(f"OK: {n} lines validated")


if __name__ == "__main__":
    main()
