# SYNC BRIEF (2026-03-18)

## 1. 当前最强 verified setting
- 当前最可信主干：`AGML-BR + A0 + Aff-ACR only`
- 4-seed（42/123/456/3407）结果：
  - Dev Quad-F1: **0.190838 ± 0.001215**
  - Test Quad-F1: **0.170541 ± 0.000960**
- 相对 `AGML-BR + A0`：test 平均提升 **+0.003198**，并伴随 A_total / A3 / A3_cat_aff_not_materialized 下降。

## 2. 当前已收口方向
- `CBR-v1`：当前定级为 **preliminary boundary-positive direction**。
  - 优点：boundary 指标有改善（如 A3_topn_drop）。
  - 问题：跨 seed 向最终 test 的转化不稳定（continue 2/4）。
- `RPH-v1` current instantiation：**negative smoke**，不作为主线推进。
- `ROMR/HOMR`：停止。

## 3. 当前唯一主任务
- fixed Aff-ACR checkpoint 下的 **frozen-feature offline replay**。
- 目标：验证 retained-only signal 能否在不扰动 retained set 的前提下，真正降低 `A3_cat_aff_not_materialized`。

## 4. 为什么下一步不是继续乱试新方法
- 现阶段主要风险不是“有没有新想法”，而是“信号是否可转化”。
- 先做 fixed-checkpoint replay 可把“信号本身可用”与“训练插入导致 checkpoint 漂移”分离。
- 在该因果链未钉死前，继续堆新模块会放大不确定性，降低论文可信度。

## 5. 论文当前最应该怎么写
- 主结果：写 `AGML-BR + A0 + Aff-ACR only` 的 4-seed verified 结果。
- CBR-v1：写为 preliminary，不写成 verified refinement。
- RPH-v1：写为 current instantiation negative smoke，并说明 failure localization 指向 selection drift + target misalignment。
- offline replay：写为 ongoing diagnostic，不写成已验证方法模块。
