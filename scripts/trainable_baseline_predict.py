"""Predict/evaluate entrypoint for the classical trainable BSPAR baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainable_baseline_lib import (
    compute_metrics,
    decode_samples,
    golds_to_eval_quads,
    load_dataset,
    load_model_artifacts,
    print_load_stats,
    print_metrics,
    quads_to_jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction with trained baseline artifacts.")
    parser.add_argument("--model_dir", type=Path, required=True, help="Directory containing model.pkl")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset path to predict.")
    parser.add_argument("--gold", type=Path, default=None, help="Optional gold dataset path for evaluation.")
    parser.add_argument("--output", type=Path, default=Path("outputs/trainable_baseline_predictions.jsonl"))
    parser.add_argument("--threshold", type=float, default=None, help="Override decode threshold.")
    parser.add_argument("--top_n_per_sent", type=int, default=None, help="Override decode top-N.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = load_model_artifacts(args.model_dir)

    vectorizer = artifacts["vectorizer"]
    clf = artifacts["classifier"]
    classes = artifacts["classes"]
    asp_lex = set(artifacts["aspect_lexicon"])
    opn_lex = set(artifacts["opinion_lexicon"])
    max_span_len = int(artifacts["max_span_len"])
    threshold = float(args.threshold) if args.threshold is not None else float(artifacts["threshold"])
    top_n = int(args.top_n_per_sent) if args.top_n_per_sent is not None else int(artifacts["top_n_per_sent"])

    input_samples, input_stats = load_dataset(args.input)
    print_load_stats("input", input_stats)

    predictions = decode_samples(
        input_samples,
        vectorizer,
        clf,
        asp_lex,
        opn_lex,
        classes,
        max_span_len,
        threshold,
        top_n,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for sample, sample_preds in zip(input_samples, predictions):
            f.write(
                json.dumps(
                    {
                        "text": sample.text,
                        "tokens": sample.tokens,
                        "pred_quads": quads_to_jsonable(sample_preds),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Saved predictions to: {args.output}")

    gold_path = args.gold if args.gold is not None else args.input
    if gold_path is not None:
        gold_samples, gold_stats = load_dataset(gold_path)
        print_load_stats("gold", gold_stats)
        if len(gold_samples) != len(predictions):
            print("Skip metrics: number of predictions and gold samples differ.")
            return
        golds = golds_to_eval_quads(gold_samples)
        metrics = compute_metrics(predictions, golds)
        print_metrics(metrics)

        metrics_path = args.output.with_suffix(".metrics.json")
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
