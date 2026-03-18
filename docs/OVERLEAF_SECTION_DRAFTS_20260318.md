# OVERLEAF SECTION DRAFTS (2026-03-18)

## 1) Method / Framework (Draft)
BSPAR is implemented as a two-stage framework rather than a one-shot reformulation. Stage-1 performs unified explicit-implicit candidate generation and ranking, while Stage-2 reranks real candidates for final quadruple prediction. In the current strongest verified system, Stage-1 uses an AGML-BR trunk with Aff-ACR-only refinement, and Stage-2 uses the A0 reranker under a fixed retention baseline (`topn_only`, `top_n=20`). This setup is intentionally controlled to preserve comparability across ablations and to isolate where gains are produced.

## 2) Main Experimental Findings (Draft)
Under a four-seed controlled evaluation (42/123/456/3407), `AGML-BR + A0 + Aff-ACR only` is the current strongest verified trunk. It achieves 0.1908 +/- 0.0012 on dev Quad-F1 and 0.1705 +/- 0.0010 on test Quad-F1. Relative to `AGML-BR + A0`, it yields consistent improvements in both final score and error-budget metrics, including reductions in A_total, A3, and A3 cat/aff not-materialized cases. These gains are treated as verified findings because they are supported by full multi-seed coverage.

## 3) Boundary-focused Diagnostics (CBR-v1) (Draft)
CBR-v1 was introduced to target top-20 boundary behavior in Stage-1. In multi-seed analysis, CBR-v1 shows a clear boundary-positive signal (e.g., lower A3 topn-drop and improved boundary-side indicators). However, the boundary improvements do not consistently transfer to end-to-end test gains across seeds. Therefore, CBR-v1 is currently classified as a preliminary boundary-positive direction rather than a verified refinement.

## 4) Retained-only Diagnostics (RPH / Oracle / Frozen Replay) (Draft)
Retained-only oracle analysis indicates non-trivial headroom concentrated in A3 cat/aff materialization errors, while A1 opinion-only miss has near-zero theoretical fixability under the retained-only constraint. Frozen-probe feasibility diagnostics further show strong separability from detached pair representations. Nevertheless, the current RPH-v1 instantiation is a negative smoke result. Failure localization suggests selection drift and target-misalignment risk in the current insertion. Hence, retained-only diagnostics should be presented as evidence of potential, not as a confirmed method module.

## 5) Current Next-step Paragraph (Draft)
The immediate next step is a fixed-checkpoint, fixed-retained-set offline replay to test signal convertibility without trunk drift. This protocol explicitly prevents retention-path perturbation and isolates whether the retained-only signal can reduce A3 cat/aff not-materialized errors. Until this conversion is verified, retained-only components should remain diagnostic rather than promoted into the main model line.

---

## Suggested Section/Subsection Titles
- `Overall Framework and Controlled Optimization Scope`
- `Verified Multi-seed Improvements of the AGML-BR + A0 + Aff-ACR Trunk`
- `Boundary-positive but Transfer-unstable Behavior of CBR-v1`
- `Retained-only Headroom: Oracle and Frozen-Probe Diagnostics`
- `From Signal Separability to Practical Convertibility: Current Status`

## Cautionary Wording Templates
- Verified wording: "We observe a consistent multi-seed improvement..."
- Preliminary wording: "We observe boundary-side improvements, but transfer remains unstable..."
- Diagnostic wording: "These results suggest potential separability, but do not yet establish a deployable module..."
