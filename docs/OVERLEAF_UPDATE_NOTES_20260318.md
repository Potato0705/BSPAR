# OVERLEAF UPDATE NOTES (2026-03-18)

## 1. Method 部分应如何表述当前方法链
- 主方法链建议写为：
  1. Stage-1：AGML-BR trunk + Aff-ACR only（当前 strongest verified trunk）
  2. Stage-2：A0 reranker（固定，不作为本轮创新点）
- 建议强调：
  - 论文核心不是 one-shot 重写，而是在 BSPAR 两阶段框架内做可验证增强。
  - 当前贡献重心是 Stage-1 候选质量与 materialization 可靠性诊断闭环。

## 2. 哪些结果应写成 verified findings
- `AGML-BR + A0 + Aff-ACR only` 的 4-seed dev/test 主结果（mean/std）。
- 相对 `AGML-BR + A0` 的 4-seed稳定增益（含 A 桶分解与 rank/gap 诊断）。
- 以上可以进入主结果与主消融表。

## 3. 哪些结果只能写成 preliminary findings
- `CBR-v1`：boundary-positive，但 4-seed transfer 不稳定（continue 2/4）。
- retained oracle / frozen-probe feasibility：
  - 证明“有信号、可能有上限”，但尚未形成稳定端到端收益。
- 这些结果建议放在分析/讨论或 appendix 的“ongoing directions”。

## 4. RPH-v1 当前该如何表述
- 不应写成“retained probe hypothesis 被彻底否定”。
- 推荐写法：
  - **RPH-v1 current instantiation is a negative smoke**。
  - failure localization 指向 **selection drift + target misalignment**，而非“retained-only signal 不存在”。

## 5. 下一步 planned experiment 应如何表述
- 计划实验：**frozen-feature offline probe replay**
  - fixed Aff-ACR checkpoint
  - fixed retained set（严格不扰动 retention）
  - 目标：验证 signal 可转化性（尤其 `A3_cat_aff_not_materialized`）
- 明确不是：
  - 直接扩 RPH-v2 训练
  - 新 trunk 重训
  - 新方法家族发散

## 6. 建议可写入论文的段落/小节标题
- `Stage-1 Trunk Stabilization with AGML-BR and Aff-ACR`
- `Verified Multi-seed Gains under Fixed A0 Reranking`
- `Boundary-Positive but Transfer-Unstable: Evidence from CBR-v1`
- `Retained-only Repair Feasibility: Oracle and Frozen-Probe Analysis`
- `Negative Smoke, Not Hypothesis Collapse: Interpreting RPH-v1`
- `Planned Validation: Fixed-Checkpoint Offline Replay for Signal Convertibility`
