# BSPAR: Method Overview (Paper-Oriented)

## One-Sentence Definition

BSPAR is a bi-stage span-based framework for aspect sentiment quadruple prediction that unifies explicit and implicit element proposal, constructs aspect-opinion pairs with quad-aware joint scoring, and trains a reranker on real generated candidates to close the training-inference gap.

## Three Core Contributions

### C1: Unified Explicit-Implicit Span Proposal
- Span enumeration with learnable, context-conditioned NULL prototypes
- Unifies explicit span extraction and implicit element modeling in one framework
- Supports Claim 1 (span-based > token-only) & Claim 2 (implicit modeling improves robustness)

### C2: Quad-Aware Joint Scoring
- Category and affective predictions participate in quad-level scoring, not post-hoc classification
- Ensures consistency across all four elements of the quadruple
- Supports Claim 3 (quad-aware > pair-only)

### C3: Real-Candidate Reranking with Training-Inference Consistency
- Reranker trains on Stage-1 actual outputs, not oracle/gold candidates
- Narrows distribution gap between training and inference
- Supports Claim 4 (real-candidate training improves generalization)

## Architecture Overview

```
Input: x = (w₁, w₂, ..., wₙ)
         │
         ▼
┌─────────────────────────┐
│   Shared Encoder        │  → H = (h₁, ..., hₙ) ∈ ℝⁿˣᵈ
│   (RoBERTa/XLM-R/etc.)  │
└────────┬────────────────┘
         │
═════════╪═══════════════════════
 STAGE 1 │
═════════╪═══════════════════════
         ▼
┌─────────────────────────┐
│ Module A: Span Proposal  │
│ • Enumerate s_(i,j)     │
│ • Compute r_(i,j)       │
│ • Unary scores φ_asp,   │
│   φ_opn                 │
│ • Prune top-K           │
│ • NULL prototypes       │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│ Module B: Pair + MTL     │
│ • Pair repr r_pair      │
│ • P_pair, P(c|a,o),     │
│   P(y|a,o)              │
│ • Hard-negative training │
└────────┬────────────────┘
         │
═════════╪═══════════════════════
 STAGE 2 │
═════════╪═══════════════════════
         ▼
┌─────────────────────────┐
│ Module C: Quad Reranker  │
│ • Real candidate quads  │
│ • Quad repr r_quad      │
│ • Score S(q)            │
│ • Margin ranking loss   │
│ • NMS dedup + decode    │
└─────────────────────────┘
         │
         ▼
Output: {(a₁,o₁,c₁,y₁), ...}
```

## Key Mathematical Definitions

### Task Formalization
- Input: x = (w₁, ..., wₙ)
- Output: Q = {q₁, ..., qₘ}, qₖ = (aₖ, oₖ, cₖ, yₖ)
- aₖ ∈ S ∪ {NULL}, oₖ ∈ S ∪ {NULL}
- cₖ ∈ C (predefined categories)
- yₖ ∈ {POS, NEG, NEU} (ASQP) or (v, ar) ∈ [1,5]² (dimABSA)

### Span Representation
r_(i,j) = [hᵢ; hⱼ; AttnPool(hᵢ..hⱼ); e_width(j−i+1)]

### NULL Prototypes (Context-Conditioned)
r_A_NULL = FFN_null-a(h̄), r_O_NULL = FFN_null-o(h̄)
where h̄ = mean(h₁, ..., hₙ)

### Pair Representation
r_pair = [rₐ; rₒ; rₐ⊙rₒ; |rₐ−rₒ|; e_dist; e_ord]

### Quad Score
r_quad = [r_pair; eₓ; e_y; f_meta]
S(q) = FFN_rank(r_quad)

### Training Objective
L = λ₁·L_span + λ₂·L_pair + λ₃·L_cat + λ₄·L_aff + λ₅·L_rank

## Design Decisions (Main Line)

| Decision | Main | Alternative | Rationale |
|----------|------|-------------|-----------|
| Span scoring | Dual binary heads | 3-way softmax | Independent heads for aspect/opinion roles |
| NULL prototype | Context-conditioned | Fixed learnable vector | Captures context-dependent implicit semantics |
| Pair loss | BCE + hard negatives | Contrastive | Stable; hard neg provides discrimination |
| Ranking loss | Pairwise margin | Pointwise/Listwise | Clear gradient signal |
| Affective (ASQP) | Sentiment CE | Ordinal regression | Standard, simple |
| Affective (dimABSA) | Smooth-L1 | Uncertainty-aware | Smooth-L1 as stable baseline |
| Reranker training | Real candidates | Oracle candidates | Core contribution — not replaceable |

## Differences from Baseline Categories

| Type | Representative | BSPAR Difference |
|------|---------------|-------------------|
| Pipeline tagging | BIO → pair → classify | Span-granularity; joint quad scoring |
| Generative | GAS, Paraphrase | Structured reasoning; no generation order dependency |
| Span pair-only | SpanASTE | Quad-aware scoring beyond pair; implicit modeling |
| MRC-based | BMRC | No query templates; no multi-round queries |
| Table-filling | GTS | Span-level (not token grid); quad-level interaction |
