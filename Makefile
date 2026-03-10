# BSPAR Makefile — Baseline experiments

PYTHON ?= python
SEEDS ?= 42

# ── Single dataset training ─────────────────────────────────────────────────

.PHONY: train-rest15 train-rest16 train-laptop train-restaurant

train-rest15:
	$(PYTHON) scripts/train_stage1.py --config configs/asqp_rest15.yaml --seed $(SEEDS)

train-rest16:
	$(PYTHON) scripts/train_stage1.py --config configs/asqp_rest16.yaml --seed $(SEEDS)

train-laptop:
	$(PYTHON) scripts/train_stage1.py --config configs/acos_laptop.yaml --seed $(SEEDS)

train-restaurant:
	$(PYTHON) scripts/train_stage1.py --config configs/acos_restaurant.yaml --seed $(SEEDS)

# ── Full pipeline (Stage-1 + candidate gen + Stage-2) ────────────────────────

.PHONY: pipeline-rest15 pipeline-rest16 pipeline-laptop pipeline-restaurant

pipeline-rest15:
	$(PYTHON) scripts/run_pipeline.py --config configs/asqp_rest15.yaml --seed $(SEEDS)

pipeline-rest16:
	$(PYTHON) scripts/run_pipeline.py --config configs/asqp_rest16.yaml --seed $(SEEDS)

pipeline-laptop:
	$(PYTHON) scripts/run_pipeline.py --config configs/acos_laptop.yaml --seed $(SEEDS)

pipeline-restaurant:
	$(PYTHON) scripts/run_pipeline.py --config configs/acos_restaurant.yaml --seed $(SEEDS)

# ── Run all baselines ────────────────────────────────────────────────────────

.PHONY: baseline-all
baseline-all: train-rest15 train-rest16 train-laptop train-restaurant

# ── Multi-seed experiments ───────────────────────────────────────────────────

.PHONY: multiseed-rest15
multiseed-rest15:
	@for seed in 42 123 456 789 1024; do \
		echo "=== Rest15 seed=$$seed ==="; \
		$(PYTHON) scripts/train_stage1.py \
			--config configs/asqp_rest15.yaml \
			--seed $$seed \
			--output_dir outputs/asqp_rest15/seed_$$seed; \
	done

# ── Utilities ────────────────────────────────────────────────────────────────

.PHONY: install check-data clean

install:
	pip install -r requirements.txt

check-data:
	@echo "=== Data Statistics ==="
	@wc -l data/asqp_rest15/*.txt
	@wc -l data/asqp_rest16/*.txt
	@wc -l data/acos_laptop/*.tsv
	@wc -l data/acos_restaurant/*.tsv

clean:
	rm -rf outputs/ logs/ __pycache__/ bspar/__pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
