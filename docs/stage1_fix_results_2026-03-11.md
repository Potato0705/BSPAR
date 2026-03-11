# Stage-1 Fix Results (2026-03-11)

## Context
- Branch: `claude/review-repo-contents-Hb1gS`
- Dataset: `ASQP Rest15`
- Goal: verify non-zero score after Stage-1 decode/pruning fixes

## Code changes included in this run
- Added configurable Stage-1 decode threshold: `stage1_pair_score_threshold`
- Changed Stage-1 eval decode to use pair-score-first ranking + configurable threshold
- Forced gold span inclusion in Stage-1 training-time pruning (fixed-size replacement)
- Made Stage-1 checkpoint loading compatible with PyTorch 2.6 (`weights_only=False`)

## Stage-1 Dev Results (best checkpoint)
- Seed `42`: Quad-F1 `0.007432`, Quad-P `0.004331`, Quad-R `0.026163`, Span-F1 `0.197966`
- Seed `123`: Quad-F1 `0.009747`, Quad-P `0.005750`, Quad-R `0.031977`, Span-F1 `0.153483`
- Seed `456`: Quad-F1 `0.008584`, Quad-P `0.005263`, Quad-R `0.023256`, Span-F1 `0.133634`

Average Quad-F1 over 3 seeds: `0.008588`
