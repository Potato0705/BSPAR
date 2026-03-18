# PROJECT STATUS (2026-03-18)

## 1. 项目定位
- 目标定位：论文版 BSPAR，不做 leaderboard trick。
- 方法主线固定：
  - Unified explicit-implicit candidate generation
  - Real-candidate reranking
  - Quad-aware reasoning
- 当前工程策略：先做受控诊断与最小可验证改动，再决定是否扩展训练预算。

## 2. 当前 strongest verified setting
- **主干设定**：`AGML-BR + A0 + Aff-ACR only`
- **4-seed verified（seed=42/123/456/3407）**：
  - Dev Quad-F1: **0.190838 ± 0.001215**
  - Test Quad-F1: **0.170541 ± 0.000960**
- 这是当前最可信、可复现的系统主干。

## 3. 已确认结论

### 3.1 Verified findings
- `AGML-BR + A0 + Aff-ACR only` 在 4-seed 下相对 `AGML-BR + A0` 有稳定正收益（已完成多 seed 对照）。
- 当前剩余主问题主要位于 Stage-1 retained 后 materialization 侧，而不是简单的 retention 覆盖不足。

### 3.2 Preliminary findings
- **CBR-v1**：boundary-positive 但 transfer 不稳，当前分级为 preliminary。
  - 4-seed `continue_ok_ratio = 2/4`
  - 对 `A3_topn_drop` 有改善信号，但 test 端转化不稳定。
- **retained oracle / frozen-probe feasibility**：
  - retained-only 理论可修复空间主要在 `A3_cat_aff_not_materialized`
  - `A1 opinion_only_miss` 在 retained-only 口径下理论可修复量为 0
  - `pair_reprs.detach()` 上 probe separability 强（AUC 约 0.91），但“可分”不等于“已转化”。

### 3.3 Negative findings / stopped findings
- **RPH-v1（当前 instantiation）**：seed42 smoke 为负向（相对 strongest baseline test 下降），并且 failure localization 指向链路漂移/目标错配风险，不作为当前主线。
- **ROMR/HOMR family**：停止。

## 4. 已停止或不要回到的方向
- retention/gate 旧家族（old gate / old marginal-gain / pair-floor）
- A4 作为默认增益主线（当前主线已切 A0）
- Cat-MBL / Cat-ACR only
- ROMR / HOMR
- 旧 weak-logit RIC
- 旧 3A early-interaction prior 融合版
- LLM / ICL / one-shot 重写主干

## 5. 当前唯一主任务
- 在 **fixed Aff-ACR only checkpoint** 上做 **frozen-feature offline probe replay**，目标是验证：
  - probe signal 本身能否降低 `A3_cat_aff_not_materialized`
- 这一步不是直接做 RPH-v2，也不是直接进 4-seed 训练扩展。

## 6. 分析实验必须同时看的指标
- `A1 / A3 / A_total`
- `A1 opinion_only_miss`
- `A3 topn_drop / cat_aff_not_materialized`
- `gold pair mean / median rank`
- `score@top_n - gold_pair_score`
- `first outranker type ratio`
- `sample_has_positive_after_retention_ratio`

## 7. 最近一轮工作记录
- 读取与整理：
  - strongest/cbr/rph/oracle/probe 各目录 summary 与诊断文件
  - Stage-1 核心代码与训练日志入口
- 新增脚本（诊断/回放）：
  - `scripts/diagnose_retained_calibration_feasibility.py`
  - `scripts/diagnose_retained_oracle_probe.py`
  - `scripts/audit_rphv1_failure_localization.py`
  - `scripts/replay_frozen_probe_offline.py`
  - `scripts/replay_frozen_probe_cataff_decode_offline.py`
- 关键输出目录：
  - `outputs/stage1_retained_oracle_probe_20260318_155552/`
  - `outputs/stage1_frozen_probe_replay_20260318_200800/`
  - `outputs/stage1_frozen_probe_cataff_replay_20260318_202210/`
- 本轮关键结论：
  - retained-only 上限存在，但主要针对 `A3_cat_aff_not_materialized`
  - fixed-retained replay 下，invariant 可严格保持（retained exact/jaccard=1.0）
  - 当前 probe 使用位置仍未形成明确的 materialization 桶下降证据（需继续严谨验证）

## 8. 给后续协作者/新对话的注意事项
- 不要从零开始重讲历史；直接以本文件 +对应 summary 为上下文。
- 不要把 preliminary 写成 verified。
- 不要默认 Stage-2 是主战场；当前主问题仍在 Stage-1 retained 后 materialization。
- 不要重复已证伪方向。
- 当前只允许 retained-only 且以 `A3_cat_aff_not_materialized` 为主目标进行验证。
