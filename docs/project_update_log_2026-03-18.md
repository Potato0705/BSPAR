# BSPAR 更新日志（2026-03-17 ~ 2026-03-18）

## 1. 本轮目标与边界
- 目标：在 `AGML-BR + A0 + Aff-ACR only` strongest trunk 上验证 `CBR-v1` 是否能稳定把 Stage-1 boundary 改善转化为 end-to-end 收益。
- 严格边界：
  - 不改 Stage-2 主逻辑
  - 不改 retention baseline（固定 `topn_only + top_n=20`）
  - 不继续扩 `early-interaction 3A`
  - 不做新方法家族发散

## 2. 已完成工作

### 2.1 CBR-v1 最小实现落地
- 在 Stage-1 增加 `CBR-v1`（Cutoff-aware Boundary Ranking）loss，接入总损失：
  - `L_total += cbr_v1_lambda * L_boundary`
- 关键实现点：
  - 使用与真实 retention 一致的 candidate universe（`pair_valid_mask`）
  - 使用与真实 retention 一致的排序路径（`sigmoid(pair_scores)` 后同一路径排序）
  - `K=20`，`b=3`，`detach_cutoff=True`
- 新增日志：
  - `boundary_active_positive_ratio`
  - `avg_cutoff_gap`
  - `num_samples_with_active_boundary_loss`

### 2.2 seed42 小 sweep（4 点）完成
- sweep: `lambda_b in {0.1, 0.2}`, `m_b in {0.03, 0.05}`
- 结论：
  - 最优点为 `lambda_b=0.2, m_b=0.05`
  - 该点在 seed42 上同时改善了：
    - dev/test Quad-F1
    - A3 `topn_drop`
    - gold pair mean/median rank
    - A_total
  - `sample_has_positive_after_retention_ratio` 持平（不伤）

### 2.3 固定最优点后 4-seed 受控确认完成
- 固定配置：
  - `AGML-BR + A0 + Aff-ACR only + CBR-v1(lambda=0.2, margin=0.05)`
  - seeds: `42, 123, 456, 3407`
- 对照组：
  - `AGML-BR + A0 + Aff-ACR only`（current strongest verified baseline）

### 2.4 boundary->E2E 转化断点诊断完成
- 新做了 baseline vs affacr vs cbr 的 4-seed 诊断汇总。
- 重点结论：
  - CBR-v1 在 boundary 侧有稳定正信号（`A3_topn_drop`、rank、coverage）
  - 但 test 侧跨 seed 转化不稳定（仅 1/4 seed test 正增益）
  - 已定位典型“boundary-positive but test-nonpositive” seed：`123, 3407`

## 3. 关键结果（4-seed）

### 3.1 Aff-ACR only（对照）
- dev Quad-F1: `0.190838 ± 0.001215`
- test Quad-F1: `0.170541 ± 0.000960`
- A3_topn_drop mean: `44.25`
- A3_cat_aff_not_materialized mean: `75.75`
- A_total mean: `215.75`

### 3.2 CBR-v1（lambda=0.2, margin=0.05）
- dev Quad-F1: `0.191044 ± 0.001363`
- test Quad-F1: `0.170556 ± 0.002276`
- A3_topn_drop mean: `42.00`（相对对照 `-2.25`）
- A3_cat_aff_not_materialized mean: `74.50`（相对对照 `-1.25`）
- A_total mean: `209.75`（相对对照 `-6.00`）
- sample_has_positive_after_retention_ratio mean: `0.925481`（相对对照 `+0.004808`）

### 3.3 断点信号
- 相对 Aff-ACR only 的 test 平均增量：`+0.000015`（近似持平）
- `continue` 硬判定（逐 seed）：
  - `continue_ok = 2/4`（50%）
- boundary-positive 但 test 非正的 seed：
  - `123`, `3407`

## 4. 当前结论分级
- 当前 strongest verified setting：**仍为 `AGML-BR + A0 + Aff-ACR only`**。
- `CBR-v1` 当前分级：**preliminary boundary-positive direction**（不是 verified refinement，也不是负结果）。
- 含义：
  - CBR-v1 真实命中了 Stage-1 top-20 cutoff boundary
  - 但 boundary 收益向 E2E 的稳定转化尚未成立

## 5. 本轮新增/更新文件

### 5.1 配置
- `configs/asqp_rest15_agmlbr_a0_affacr_cbrv1.yaml`

### 5.2 运行与汇总脚本
- `scripts/run_cbrv1_a0_seed42_sweep.py`
- `scripts/summarize_cbrv1_a0_seed42_sweep.py`
- `scripts/run_cbrv1_a0_multiseed_controlled.py`
- `scripts/summarize_cbrv1_a0_multiseed.py`
- `scripts/summarize_cbr_transfer_failure.py`

### 5.3 关键输出目录
- seed42 sweep:
  - `outputs/stage2_e2e_agmlbr_a0_cbrv1_seed42_sweep_20260317_221033/summary/`
- 4-seed controlled:
  - `outputs/stage2_e2e_agmlbr_a0_cbrv1_multiseed_20260318_082343/summary/`
- transfer 诊断：
  - `cbr_vs_affacr_transfer_diagnosis_per_seed.csv`
  - `cbr_vs_affacr_transfer_diagnosis_mean_std.json`
  - `cbr_vs_affacr_transfer_takeaways.md`

## 6. 下一步建议（仅方向，不在本轮执行）
- 在不追加 3A 预算前提下，优先沿“转化断点”做更窄的 materialization 侧验证，而不是继续扩 early-interaction 分支。
