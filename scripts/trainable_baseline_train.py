"""Train entrypoint for the classical trainable BSPAR baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainable_baseline_lib import (
    build_phrase_lexicons,
    build_train_instances,
    class_vocab,
    compute_metrics,
    decode_samples,
    golds_to_eval_quads,
    load_dataset,
    print_load_stats,
    print_metrics,
    save_model_artifacts,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classical BSPAR baseline.")
    parser.add_argument("--train", type=Path, required=True, help="Train dataset path.")
    parser.add_argument("--dev", type=Path, default=None, help="Optional dev dataset path.")
    parser.add_argument("--max_span_len", type=int, default=4, help="Max candidate span length.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Class probability threshold.")
    parser.add_argument("--top_n_per_sent", type=int, default=5, help="Top-N decoded quads per sentence.")
    parser.add_argument("--max_iter", type=int, default=400, help="LogisticRegression max_iter.")
    parser.add_argument("--save_dir", type=Path, required=True, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_samples, train_stats = load_dataset(args.train)
    print_load_stats("train", train_stats)

    dev_samples = None
    if args.dev is not None:
        dev_samples, dev_stats = load_dataset(args.dev)
        print_load_stats("dev", dev_stats)

    asp_lex, opn_lex = build_phrase_lexicons(train_samples)
    classes = class_vocab(train_samples)

    print(f"Aspect lexicon size:  {len(asp_lex)}")
    print(f"Opinion lexicon size: {len(opn_lex)}")
    print(f"Label classes:        {len(classes)}")

    train_features, train_targets = build_train_instances(
        train_samples,
        asp_lex,
        opn_lex,
        classes,
        args.max_span_len,
    )
    print(f"Training pair instances: {len(train_features)}")
    print(f"Positive labels:         {int(train_targets.sum())}")

    vectorizer, clf = train_model(train_features, train_targets, max_iter=args.max_iter)
    print("Model training completed.")

    eval_metrics = None
    if dev_samples is not None:
        predictions = decode_samples(
            dev_samples,
            vectorizer,
            clf,
            asp_lex,
            opn_lex,
            classes,
            args.max_span_len,
            args.threshold,
            args.top_n_per_sent,
        )
        golds = golds_to_eval_quads(dev_samples)
        eval_metrics = compute_metrics(predictions, golds)
        print_metrics(eval_metrics)

    save_model_artifacts(
        save_dir=args.save_dir,
        vectorizer=vectorizer,
        clf=clf,
        classes=classes,
        asp_lex=asp_lex,
        opn_lex=opn_lex,
        max_span_len=args.max_span_len,
        threshold=args.threshold,
        top_n_per_sent=args.top_n_per_sent,
        extra_config={
            "train_path": str(args.train),
            "dev_path": str(args.dev) if args.dev else None,
            "max_iter": args.max_iter,
            "train_samples": len(train_samples),
        },
    )

    if eval_metrics is not None:
        with (args.save_dir / "dev_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    with (args.save_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    print(f"Saved artifacts to: {args.save_dir}")


if __name__ == "__main__":
    main()
