"""Controlled 4-seed runner for AGML-BR + A0 + Aff-ACR only.

Pipeline per seed:
1) Stage-1 train (Aff-ACR only)
2) Stage-1 A-breakdown on test split
3) Candidate generation
4) Stage-2 A0 train
5) Stage-2 eval on dev
6) Stage-2 eval on test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aff-ACR A0 controlled multiseed")
    parser.add_argument(
        "--config",
        default="configs/asqp_rest15_agmlbr_a0_affacr.yaml",
        help="Base YAML config path",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Output root (default: outputs/stage2_e2e_agmlbr_a0_affacr_multiseed_<timestamp>)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds; default from config.seeds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if target artifact already exists",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], log_files: list[Path]) -> None:
    for log_file in log_files:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def ensure_testeval_config(base_cfg_path: Path, out_cfg_path: Path) -> None:
    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["dev_file"] = cfg.get("test_file", "test.txt")
    out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with out_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def read_seeds(cfg: dict, arg_seeds: str | None) -> list[int]:
    if arg_seeds:
        return [int(s.strip()) for s in arg_seeds.split(",") if s.strip()]
    seeds = cfg.get("seeds", [42, 123, 456, 3407])
    return [int(s) for s in seeds]


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (PROJECT_ROOT / f"outputs/stage2_e2e_agmlbr_a0_affacr_multiseed_{ts}").resolve()

    seeds = read_seeds(cfg, args.seeds)

    cfg_dir = output_root / "configs"
    notes_dir = output_root / "notes"
    runs_dir = output_root / "runs"
    summary_dir = output_root / "summary"
    stage1_root = output_root / "stage1_runs"
    for p in [cfg_dir, notes_dir, runs_dir, summary_dir, stage1_root]:
        p.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg_dir / "affacr_a0_train.yaml"
    testeval_cfg = cfg_dir / "affacr_a0_testeval.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        train_cfg.write_text(f.read(), encoding="utf-8")
    ensure_testeval_config(train_cfg, testeval_cfg)

    commands_file = notes_dir / "commands_used.txt"
    commands_file.write_text("", encoding="utf-8")

    plan = {
        "config": str(config_path),
        "train_config": str(train_cfg),
        "testeval_config": str(testeval_cfg),
        "seeds": seeds,
        "retention": cfg.get("stage1_pair_retention_strategy", "topn_only"),
        "top_n": cfg.get("stage1_pair_top_n", 20),
        "stage2_use_pair_prior": cfg.get("stage2_use_pair_prior", False),
        "stage2_use_group_loss": cfg.get("stage2_use_group_loss", False),
    }
    (notes_dir / "run_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for seed in seeds:
        run_name = f"A0_seed{seed}_agmlbr_affacr"
        run_dir = runs_dir / run_name
        stage1_dir = stage1_root / run_name
        stage2_dir = run_dir / "stage2"
        cand_dir = run_dir / "candidates"
        run_log = run_dir / "run.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        stage1_ckpt = stage1_dir / "best_stage1.pt"
        a_breakdown_json = stage1_dir / "stage1_a_breakdown_test.json"
        stage2_ckpt = stage2_dir / "best_stage2.pt"
        eval_dev_json = stage2_dir / "eval_best_stage2_dev.json"
        eval_test_json = stage2_dir / "eval_best_stage2_test.json"

        if args.force or not stage1_ckpt.exists():
            cmd = [
                "python", "scripts/train_stage1.py",
                "--config", str(train_cfg),
                "--seed", str(seed),
                "--output_dir", str(stage1_dir),
                "--output_root", str(stage1_root),
                "--purpose", "ablation",
                "--run_tag", "agmlbr_affacr_controlled",
            ]
            run_cmd(cmd, [commands_file, run_log])

        if args.force or not a_breakdown_json.exists():
            cmd = [
                "python", "scripts/analyze_stage1_a_breakdown.py",
                "--config", str(testeval_cfg),
                "--stage1_ckpt", str(stage1_ckpt),
                "--seed", str(seed),
                "--output", str(a_breakdown_json),
            ]
            run_cmd(cmd, [commands_file, run_log])

        if args.force or not (cand_dir / "rerank_train.pt").exists():
            cmd = [
                "python", "scripts/generate_candidates.py",
                "--config", str(train_cfg),
                "--checkpoint", str(stage1_ckpt),
                "--output", str(cand_dir),
            ]
            run_cmd(cmd, [commands_file, run_log])

        if args.force or not stage2_ckpt.exists():
            cmd = [
                "python", "scripts/train_stage2.py",
                "--config", str(train_cfg),
                "--candidates_dir", str(cand_dir),
                "--output_dir", str(stage2_dir),
                "--seed", str(seed),
            ]
            run_cmd(cmd, [commands_file, run_log])

        if args.force or not eval_dev_json.exists():
            cmd = [
                "python", "scripts/eval_stage2_dev.py",
                "--config", str(train_cfg),
                "--stage1_ckpt", str(stage1_ckpt),
                "--stage2_ckpt", str(stage2_ckpt),
                "--seed", str(seed),
                "--output", str(eval_dev_json),
            ]
            run_cmd(cmd, [commands_file, run_log])

        if args.force or not eval_test_json.exists():
            cmd = [
                "python", "scripts/eval_stage2_dev.py",
                "--config", str(testeval_cfg),
                "--stage1_ckpt", str(stage1_ckpt),
                "--stage2_ckpt", str(stage2_ckpt),
                "--seed", str(seed),
                "--output", str(eval_test_json),
            ]
            run_cmd(cmd, [commands_file, run_log])

        run_manifest = {
            "run_name": run_name,
            "seed": seed,
            "stage1_ckpt": str(stage1_ckpt),
            "stage1_a_breakdown_test": str(a_breakdown_json),
            "candidates_dir": str(cand_dir),
            "stage2_ckpt": str(stage2_ckpt),
            "eval_dev_json": str(eval_dev_json),
            "eval_test_json": str(eval_test_json),
            "run_dir": str(run_dir),
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Done. Output root: {output_root}")
    print("Next: run scripts/summarize_affacr_a0_multiseed.py for consolidated tables.")


if __name__ == "__main__":
    main()
