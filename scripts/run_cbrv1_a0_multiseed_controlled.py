"""Controlled 4-seed runner for AGML-BR + A0 + Aff-ACR only + CBR-v1.

Fixed setting (no sweep):
- cbr_v1_lambda = 0.2
- cbr_v1_margin = 0.05
- seeds = 42, 123, 456, 3407 (default)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CBR-v1 A0 controlled 4-seed")
    parser.add_argument(
        "--base_config",
        default="configs/asqp_rest15_agmlbr_a0_affacr_cbrv1.yaml",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Default: outputs/stage2_e2e_agmlbr_a0_cbrv1_multiseed_<timestamp>",
    )
    parser.add_argument(
        "--seeds",
        default="42,123,456,3407",
        help="Comma-separated seeds",
    )
    parser.add_argument("--cbr_lambda", type=float, default=0.2)
    parser.add_argument("--cbr_margin", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], logs: list[Path]) -> None:
    for lp in logs:
        lp.parent.mkdir(parents=True, exist_ok=True)
        with lp.open("a", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def ensure_testeval_config(train_cfg: Path, test_cfg: Path) -> None:
    cfg = yaml.safe_load(train_cfg.read_text(encoding="utf-8"))
    cfg["dev_file"] = cfg.get("test_file", "test.txt")
    test_cfg.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def parse_seeds(raw: str) -> list[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def tagf(v: float) -> str:
    return str(v).replace(".", "p")


def main() -> None:
    args = parse_args()
    base_config_path = (PROJECT_ROOT / args.base_config).resolve()
    base_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = (
            PROJECT_ROOT / f"outputs/stage2_e2e_agmlbr_a0_cbrv1_multiseed_{ts}"
        ).resolve()

    seeds = parse_seeds(args.seeds)
    lam = float(args.cbr_lambda)
    margin = float(args.cbr_margin)
    run_suffix = f"cbrv1_lam{tagf(lam)}_m{tagf(margin)}"

    cfg_dir = output_root / "configs"
    notes_dir = output_root / "notes"
    runs_dir = output_root / "runs"
    stage1_root = output_root / "stage1_runs"
    summary_dir = output_root / "summary"
    for p in [cfg_dir, notes_dir, runs_dir, stage1_root, summary_dir]:
        p.mkdir(parents=True, exist_ok=True)

    commands_file = notes_dir / "commands_used.txt"
    commands_file.write_text("", encoding="utf-8")

    plan = {
        "base_config": str(base_config_path),
        "seeds": seeds,
        "fixed_cbr_v1_lambda": lam,
        "fixed_cbr_v1_margin": margin,
        "cbr_v1_buffer": 3,
        "cbr_v1_detach_cutoff": True,
        "retention": base_cfg.get("stage1_pair_retention_strategy", "topn_only"),
        "top_n": base_cfg.get("stage1_pair_top_n", 20),
        "stage2_use_pair_prior": base_cfg.get("stage2_use_pair_prior", False),
        "stage2_use_group_loss": base_cfg.get("stage2_use_group_loss", False),
    }
    (notes_dir / "run_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for seed in seeds:
        run_name = f"A0_seed{seed}_{run_suffix}"
        run_dir = runs_dir / run_name
        stage1_dir = stage1_root / run_name
        stage2_dir = run_dir / "stage2"
        cand_dir = run_dir / "candidates"
        run_log = run_dir / "run.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg_dir / f"{run_suffix}_seed{seed}_train.yaml"
        testeval_cfg = cfg_dir / f"{run_suffix}_seed{seed}_testeval.yaml"
        cfg = dict(base_cfg)
        cfg["use_cbr_v1_loss"] = True
        cfg["cbr_v1_lambda"] = lam
        cfg["cbr_v1_margin"] = margin
        cfg["cbr_v1_buffer"] = 3
        cfg["cbr_v1_detach_cutoff"] = True
        cfg["seeds"] = [seed]
        train_cfg.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        ensure_testeval_config(train_cfg, testeval_cfg)

        stage1_ckpt = stage1_dir / "best_stage1.pt"
        a_breakdown_json = stage1_dir / "stage1_a_breakdown_test.json"
        a1_split_json = stage1_dir / "a1_span_side_split_test.json"
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
                "--run_tag", run_suffix,
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

        if args.force or not a1_split_json.exists():
            cmd = [
                "python", "scripts/extract_a1_span_side_split.py",
                "--config", str(testeval_cfg),
                "--stage1_ckpt", str(stage1_ckpt),
                "--seed", str(seed),
                "--split", "test",
                "--output", str(a1_split_json),
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

        manifest = {
            "run_name": run_name,
            "seed": seed,
            "cbr_v1_lambda": lam,
            "cbr_v1_margin": margin,
            "cbr_v1_buffer": 3,
            "cbr_v1_detach_cutoff": True,
            "stage1_ckpt": str(stage1_ckpt),
            "stage1_a_breakdown_test": str(a_breakdown_json),
            "a1_span_side_split_test": str(a1_split_json),
            "candidates_dir": str(cand_dir),
            "stage2_ckpt": str(stage2_ckpt),
            "eval_dev_json": str(eval_dev_json),
            "eval_test_json": str(eval_test_json),
            "run_dir": str(run_dir),
            "train_config": str(train_cfg),
            "testeval_config": str(testeval_cfg),
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Done. Output root: {output_root}")
    print("Next: run scripts/summarize_cbrv1_a0_multiseed.py")


if __name__ == "__main__":
    main()

