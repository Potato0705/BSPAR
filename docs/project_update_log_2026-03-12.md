# BSPAR 排查与更新日志（2026-03-11 ~ 2026-03-12）

## 1. 目标与范围
- 目标：修通 Stage-1/Stage-2 候选流，定位 0 分原因，建立稳定 baseline，并验证最小 Stage-2 校准增强。
- 约束：不改 Stage-1 主架构；外部 pipeline 接口保持不变；先修 pipeline/pruning/decode，再做最小训练增强。

## 2. 关键排查结论

### 2.1 Stage-1 主断点
- 初始主断点不是 span 覆盖本身，而是 pair candidate flow 被 gate/阈值路径截断。
- 修复前后验证显示：pair gate 路径统一后，正例候选进入 Stage-2 明显恢复。

### 2.2 Stage-1 retention baseline 固化
- 通过 Phase-2 ablation 固化默认策略：
  - `stage1_pair_retention_strategy: topn_only`
  - `stage1_pair_top_n: 20`
  - decode 不额外阈值门控（`stage1_decode_pair_score_threshold: null`）

### 2.3 Stage-1 pair calibration 增强（最小改动）
- 加入 difficulty-aware pair loss（easy/span-near-miss/category-confused）与 pair focal。
- 结果：pair score 分离度和正例进入率提升，Stage-1 从“可跑”进入“可工作”。

### 2.4 Stage-2 `L_group` 实验结论
- 实现了 Pair-Group Listwise Consistency Loss 并完成 A0/A1/A2。
- 结论：当前数据上 `L_group` 未带来净收益，不是当前最高优先级。

### 2.5 Stage-2 `+pair-prior`（当前保留方案）
- 最小增强：`S_final = S_quad + alpha * S_pair`，并加 `L_pair_prior`（pair-level BCE）。
- 单 seed sweep 结论：`alpha=1.0, lambda_pair=0.3` 最优。
- 多 seed 结论：A4 相对 A0 的 Quad-F1 均值稳定正增益，且 D 类错误平均下降。

## 3. 多 seed 结果（最新）

### 3.1 Stage-1 多 seed（固定 retention baseline）
- 文件：`outputs/asqp_rest15_phase4_multiseed_v2/multiseed_summary.json`
- Quad-F1: `0.0663 ± 0.0129`
- Span-F1: `0.3804 ± 0.0328`

### 3.2 Stage-2 A0 vs A4（`alpha=1.0, lambda_pair=0.3`）
- 文件：`outputs/asqp_rest15_stage2_pairprior_multiseed/a0_vs_a4_multiseed_summary.json`
- A0 Quad-F1: `0.1176 ± 0.0007`
- A4 Quad-F1: `0.1194 ± 0.0010`
- Mean delta: `+0.00185`
- D 类错误平均变化: `-0.75`
- top-k 正确 pair 覆盖率平均变化: `+0.00147`
- 4/4 seeds 均为非负收益。

## 4. A 类错误分解（当前瓶颈）
- 文件：`outputs/asqp_rest15_stage1_A_breakdown_seed123.json`
- 总 A miss: `189`
- A1（gold span 未进 top-k）: `61`（`32.28%`）
- A2（gold spans 在但 pair 不在 pair-space）: `0`（`0.00%`）
- A3（gold pair 在但未进入 retained candidates）: `96`（`50.79%`）
  - 其中 top-n drop: `87`
  - cat/aff 未 materialize: `9`
- A4（implicit/NULL miss）: `25`（`13.23%`）
- A5（max_span_length 限制）: `7`（`3.70%`）

结论：下一优先级为 `candidate retention policy -> span proposal -> implicit handling`，而不是 pair-space construction（A2=0）。

## 5. 主要代码更新清单
- Stage-1
  - `bspar/models/bspar_stage1.py`
  - `bspar/training/stage1_trainer.py`
  - `bspar/training/candidate_generator.py`
  - `bspar/config.py`
- Stage-2
  - `bspar/models/bspar_stage2.py`
  - `bspar/training/stage2_trainer.py`
  - `bspar/config.py`
- 新增分析/评测脚本
  - `scripts/diagnose_stage1_pair_scores.py`
  - `scripts/eval_stage2_dev.py`
  - `scripts/analyze_stage2_error_sources.py`
  - `scripts/analyze_stage1_a_breakdown.py`
- 新增实验配置
  - `configs/asqp_rest15_phase3_paircal.yaml`
  - `configs/asqp_rest15_stage2_a*.yaml`
  - `configs/asqp_rest15_stage2_pairprior_*.yaml`

## 6. 结果索引（便于后查）
- Stage-1 Phase3 对比：
  - `outputs/asqp_rest15/phase3_paircal_comparison_20260311.json`
- Stage-1 多 seed：
  - `outputs/asqp_rest15_phase4_multiseed_v2/multiseed_summary.json`
- Stage-2 `L_group` ablation：
  - `outputs/asqp_rest15_stage2_ablation_seed123/ablation_summary.json`
- Stage-2 `+pair-prior` 单 seed sweep：
  - `outputs/asqp_rest15_stage2_pairprior_ablation_seed123/pairprior_ablation_summary.json`
- Stage-2 `A0 vs A4` 多 seed：
  - `outputs/asqp_rest15_stage2_pairprior_multiseed/a0_vs_a4_multiseed_summary.json`
- Stage-1 A 类拆分：
  - `outputs/asqp_rest15_stage1_A_breakdown_seed123.json`

## 7. 复现实验命令（核心）
```bash
# Stage-1 训练
python scripts/train_stage1.py --config configs/asqp_rest15_phase3_paircal.yaml --seed 123 --output_dir outputs/asqp_rest15_phase4_multiseed_v2/seed123

# 生成 Stage-2 候选
python scripts/generate_candidates.py --config configs/asqp_rest15_phase3_paircal.yaml --checkpoint outputs/asqp_rest15_phase4_multiseed_v2/seed123/best_stage1.pt --output outputs/asqp_rest15_stage2_candidates_seed123

# Stage-2 训练（A0）
python scripts/train_stage2.py --config configs/asqp_rest15_stage2_pairprior_a0.yaml --candidates_dir outputs/asqp_rest15_stage2_candidates_seed123 --output_dir outputs/asqp_rest15_stage2_pairprior_multiseed/seed123/a0 --seed 123

# Stage-2 训练（A4: pair-prior）
python scripts/train_stage2.py --config configs/asqp_rest15_stage2_pairprior_a4_alpha1p0_lam0p3.yaml --candidates_dir outputs/asqp_rest15_stage2_candidates_seed123 --output_dir outputs/asqp_rest15_stage2_pairprior_multiseed/seed123/a4 --seed 123

# Stage-2 误差分析
python scripts/analyze_stage2_error_sources.py --config configs/asqp_rest15_stage2_pairprior_a4_alpha1p0_lam0p3.yaml --stage1_ckpt outputs/asqp_rest15_phase4_multiseed_v2/seed123/best_stage1.pt --stage2_ckpt outputs/asqp_rest15_stage2_pairprior_multiseed/seed123/a4/best_stage2.pt --output outputs/asqp_rest15_stage2_pairprior_multiseed/seed123/a4/error_analysis.json

# Stage-1 A 类拆分
python scripts/analyze_stage1_a_breakdown.py --config configs/asqp_rest15_phase3_paircal.yaml --stage1_ckpt outputs/asqp_rest15_phase4_multiseed_v2/seed123/best_stage1.pt --output outputs/asqp_rest15_stage1_A_breakdown_seed123.json
```
