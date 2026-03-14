"""Minimal Stage-1 pair-rank run output governance helpers."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from uuid import uuid4


PURPOSES = {"smoke", "baseline", "ablation"}
STAGE1_SUBDIRS = ["smoke", "baseline", "ablation", "interrupted", "archive"]
RUN_INDEX_FIELDS = [
    "run_id",
    "purpose",
    "status",
    "dataset",
    "seed",
    "lambda_pair_rank",
    "pair_rank_margin",
    "quad_f1",
    "a3_count",
    "gold_pair_mean_rank",
    "gold_pair_median_rank",
    "score_topn_minus_gold_mean",
    "score_topn_minus_gold_median",
    "first_outranker_null_ratio",
    "first_outranker_nearmiss_ratio",
    "sample_has_positive_after_retention_ratio",
    "run_dir",
]


def _now_iso():
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _slug(text: str) -> str:
    cleaned = []
    for ch in str(text):
        if ch.isalnum():
            cleaned.append(ch.lower())
        elif ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "run"


def ensure_stage1_pairrank_dirs(output_root: str):
    root = os.path.abspath(output_root)
    os.makedirs(root, exist_ok=True)
    for name in STAGE1_SUBDIRS:
        os.makedirs(os.path.join(root, name), exist_ok=True)
    return root


def infer_purpose(lambda_pair_rank: float):
    return "baseline" if _safe_float(lambda_pair_rank, 0.0) == 0.0 else "smoke"


def normalize_purpose(purpose: str):
    p = str(purpose).strip().lower()
    if p not in PURPOSES:
        raise ValueError(f"Unknown purpose '{purpose}', expected one of {sorted(PURPOSES)}")
    return p


def build_run_id(purpose: str, dataset: str, seed: int, run_tag: str | None = None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:6]
    parts = [ts, normalize_purpose(purpose), _slug(dataset), f"s{_safe_int(seed)}"]
    if run_tag:
        parts.append(_slug(run_tag))
    parts.append(suffix)
    return "_".join(parts)


def _artifact_paths(run_dir: str):
    return {
        "best_ckpt_path": os.path.abspath(os.path.join(run_dir, "best_stage1.pt")),
        "final_ckpt_path": os.path.abspath(os.path.join(run_dir, "final_stage1.pt")),
        "metrics_path": os.path.abspath(os.path.join(run_dir, "stage1_metrics.json")),
        "a3_best_path": os.path.abspath(
            os.path.join(run_dir, "best_stage1_a3_diagnostics.json")
        ),
        "a3_final_path": os.path.abspath(
            os.path.join(run_dir, "final_stage1_a3_diagnostics.json")
        ),
        "log_path": os.path.abspath(os.path.join(run_dir, "train.log")),
    }


def _write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_run_manifest(
    *,
    output_root: str,
    purpose: str,
    dataset: str,
    seed: int,
    lambda_pair_rank: float,
    pair_rank_margin: float,
    retention: str,
    top_n: int,
    config_path: str,
    command: str,
    run_tag: str | None = None,
    run_dir_override: str | None = None,
):
    root = ensure_stage1_pairrank_dirs(output_root)
    purpose = normalize_purpose(purpose)

    run_id = build_run_id(purpose, dataset, seed, run_tag=run_tag)
    if run_dir_override:
        run_dir = os.path.abspath(run_dir_override)
    else:
        run_dir = os.path.abspath(os.path.join(root, purpose, run_id))

    os.makedirs(run_dir, exist_ok=True)
    artifact_paths = _artifact_paths(run_dir)

    manifest = {
        "run_id": run_id,
        "status": "running",
        "purpose": purpose,
        "dataset": str(dataset),
        "seed": _safe_int(seed),
        "lambda_pair_rank": _safe_float(lambda_pair_rank),
        "pair_rank_margin": _safe_float(pair_rank_margin),
        "retention": str(retention),
        "top_n": _safe_int(top_n),
        "config_path": os.path.abspath(config_path),
        "command": str(command),
        "start_time": _now_iso(),
        "end_time": None,
        "run_dir": run_dir,
        "best_ckpt_path": artifact_paths["best_ckpt_path"],
        "final_ckpt_path": artifact_paths["final_ckpt_path"],
        "metrics_path": artifact_paths["metrics_path"],
        "a3_best_path": artifact_paths["a3_best_path"],
        "a3_final_path": artifact_paths["a3_final_path"],
        "log_path": artifact_paths["log_path"],
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def write_metrics_file(metrics_path: str, best_metrics: dict, final_metrics: dict):
    payload = {
        "best_dev_metrics": dict(best_metrics or {}),
        "final_dev_metrics": dict(final_metrics or {}),
    }
    _write_json(metrics_path, payload)
    return payload


def _move_run_to_interrupted(run_dir: str, output_root: str):
    run_dir = os.path.abspath(run_dir)
    interrupted_root = os.path.abspath(os.path.join(output_root, "interrupted"))
    os.makedirs(interrupted_root, exist_ok=True)

    base = os.path.basename(run_dir.rstrip("\\/"))
    target = os.path.join(interrupted_root, base)
    idx = 1
    while os.path.exists(target):
        target = os.path.join(interrupted_root, f"{base}_r{idx}")
        idx += 1

    shutil.move(run_dir, target)
    return os.path.abspath(target)


def finalize_manifest(
    manifest: dict,
    *,
    output_root: str,
    status: str,
    error_message: str | None = None,
):
    final_status = str(status).strip().lower()
    if final_status not in {"finished", "interrupted", "failed"}:
        raise ValueError(f"Unsupported final status: {status}")

    run_dir = os.path.abspath(manifest["run_dir"])
    if final_status in {"interrupted", "failed"}:
        parent = os.path.basename(os.path.dirname(run_dir.rstrip("\\/"))).lower()
        if parent != "interrupted":
            run_dir = _move_run_to_interrupted(run_dir, output_root)

    artifact_paths = _artifact_paths(run_dir)
    manifest.update(
        {
            "status": final_status,
            "end_time": _now_iso(),
            "run_dir": run_dir,
            "best_ckpt_path": artifact_paths["best_ckpt_path"],
            "final_ckpt_path": artifact_paths["final_ckpt_path"],
            "metrics_path": artifact_paths["metrics_path"],
            "a3_best_path": artifact_paths["a3_best_path"],
            "a3_final_path": artifact_paths["a3_final_path"],
            "log_path": artifact_paths["log_path"],
        }
    )
    if error_message:
        manifest["error_message"] = str(error_message)

    manifest_path = os.path.join(run_dir, "manifest.json")
    _write_json(manifest_path, manifest)
    return manifest, manifest_path


def _read_run_index(index_path: str):
    if not os.path.exists(index_path):
        return []
    with open(index_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def update_run_index(index_path: str, row: dict):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    run_id = str(row.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("run_index row must include run_id")

    rows = _read_run_index(index_path)
    replaced = False
    for i, old in enumerate(rows):
        if str(old.get("run_id", "")).strip() == run_id:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with open(index_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_INDEX_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({key: r.get(key, "") for key in RUN_INDEX_FIELDS})


def load_a3_summary(a3_path: str):
    a3 = _read_json(a3_path)
    counts = a3.get("counts", {})
    rank = a3.get("gold_pair_rank", {})
    gap = a3.get("score_topn_minus_gold_pair", {})
    ratios = a3.get("first_outranker_type_ratio", {})
    return {
        "a3_count": _safe_int(counts.get("total_a3_gold_pairs", 0)),
        "gold_pair_mean_rank": _safe_float(rank.get("mean", 0.0)),
        "gold_pair_median_rank": _safe_float(rank.get("median", 0.0)),
        "score_topn_minus_gold_mean": _safe_float(gap.get("mean", 0.0)),
        "score_topn_minus_gold_median": _safe_float(gap.get("median", 0.0)),
        "first_outranker_null_ratio": _safe_float(
            ratios.get("NULL", {}).get("ratio", 0.0)
        ),
        "first_outranker_nearmiss_ratio": _safe_float(
            ratios.get("near_miss", {}).get("ratio", 0.0)
        ),
        "sample_has_positive_after_retention_ratio": _safe_float(
            a3.get("sample_has_positive_after_retention_ratio", 0.0)
        ),
    }


def build_run_index_row(manifest: dict, best_metrics: dict | None = None):
    best_metrics = best_metrics or {}
    row = {
        "run_id": manifest.get("run_id", ""),
        "purpose": manifest.get("purpose", ""),
        "status": manifest.get("status", ""),
        "dataset": manifest.get("dataset", ""),
        "seed": manifest.get("seed", ""),
        "lambda_pair_rank": manifest.get("lambda_pair_rank", ""),
        "pair_rank_margin": manifest.get("pair_rank_margin", ""),
        "quad_f1": _safe_float(best_metrics.get("quad_f1", 0.0)),
        "a3_count": "",
        "gold_pair_mean_rank": "",
        "gold_pair_median_rank": "",
        "score_topn_minus_gold_mean": "",
        "score_topn_minus_gold_median": "",
        "first_outranker_null_ratio": "",
        "first_outranker_nearmiss_ratio": "",
        "sample_has_positive_after_retention_ratio": "",
        "run_dir": manifest.get("run_dir", ""),
    }

    a3_path = str(manifest.get("a3_best_path", ""))
    if os.path.exists(a3_path):
        row.update(load_a3_summary(a3_path))
    return row


def build_command(argv=None):
    args = argv if argv is not None else sys.argv
    return f"python {subprocess.list2cmdline(args)}"


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_to_log(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        out = TeeStream(sys.stdout, log_file)
        err = TeeStream(sys.stderr, log_file)
        with redirect_stdout(out), redirect_stderr(err):
            yield
