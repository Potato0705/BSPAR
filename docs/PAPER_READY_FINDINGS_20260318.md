# PAPER READY FINDINGS (2026-03-18)

## 1) Verified Findings

### V1. Current strongest verified trunk
- **Setting**: `AGML-BR + A0 + Aff-ACR only`
- **4-seed verified (42/123/456/3407)**:
  - Dev Quad-F1: **0.190838 ± 0.001215**
  - Test Quad-F1: **0.170541 ± 0.000960**
- This is the current most reliable and reproducible trunk.

### V2. Verified gain over AGML-BR + A0 baseline (4-seed)
- Relative to `AGML-BR + A0`, `+Aff-ACR only` shows stable improvements:
  - Test Quad-F1: **0.167343 -> 0.170541** (delta **+0.003198**)
  - A_total: **219.00 -> 215.75** (delta **-3.25**)
  - A1: **28.50 -> 27.50** (delta **-1.00**)
  - A1 opinion_only_miss: **19.00 -> 16.25** (delta **-2.75**)
  - A3: **125.50 -> 120.00** (delta **-5.50**)
  - A3 cat_aff_not_materialized: **79.75 -> 75.75** (delta **-4.00**)
  - score@top_n - gold score mean: **0.201604 -> 0.169811** (delta **-0.031793**)

## 2) Preliminary Findings

### P1. CBR-v1 is boundary-positive but transfer-unstable
- **Setting**: `AGML-BR + A0 + Aff-ACR only + CBR-v1 (lambda=0.2, margin=0.05)`
- 4-seed summary:
  - Dev Quad-F1: **0.191044 ± 0.001363**
  - Test Quad-F1: **0.170556 ± 0.002276**
- Diagnostic conclusion (current level):
  - `A3_topn_drop` improves on average (delta **-2.25** vs Aff-ACR only)
  - Boundary-to-E2E transfer is not stable (`continue_ok_ratio = 2/4`)
  - Therefore CBR-v1 should be presented as a **preliminary boundary-positive direction**, not a verified refinement.

### P2. Retained-only oracle + frozen-probe feasibility is promising but not yet method-verified
- Oracle upper-bound indicates non-trivial retained-only repair headroom, mostly in `A3_cat_aff_not_materialized`.
- Mean oracle-fixable A1 opinion-only miss is **0.0** under retained-only constraint.
- Frozen probe separability is high (`probe2_auc` around **0.91**), but this is feasibility evidence, not a verified method gain.

## 3) Negative / Stopped Findings

### N1. RPH-v1 current instantiation is a negative smoke
- In seed42 smoke, RPH-v1 underperforms strongest Aff-ACR baseline on test:
  - strongest Aff-ACR test: **0.169101**
  - RPH-v1 test: **0.166748** (delta **-0.002353**)
- Failure localization indicates this is not a clean retained-only benefit realization.
- Current interpretation for paper: **negative smoke for this instantiation**, not a global rejection of retained-signal hypothesis.

### N2. Stopped direction families
- ROMR/HOMR family: stopped.
- Previously validated dead-end families remain out of scope for current paper trunk.

## 4) Ongoing Diagnostic-only Findings

### D1. Offline replay (fixed checkpoint, fixed retained set) confirms strict invariants
- Retained set invariants are preserved:
  - retained exact match = 1.0
  - Jaccard = 1.0
  - sample_has_positive_after_retention_ratio delta = 0
  - A3_topn_drop delta = 0

### D2. Probe signal conversion is not yet demonstrated on target bucket
- Frozen probe offline replay can produce non-negative F1 movement in specific setups,
  but **does not yet reduce** `A3_cat_aff_not_materialized` in current fixed-checkpoint replays.
- This remains an ongoing diagnostic result and should not be written as a verified module.

---

## Writing Rule for Main Paper
- Verified: can appear in main result table and main claim text.
- Preliminary: can appear in analysis/ablation with explicit caveat.
- Negative/stopped: can appear as controlled failure evidence.
- Ongoing diagnostic-only: should appear as future-work/ongoing diagnosis, not as final method claim.
