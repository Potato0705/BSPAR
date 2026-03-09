# BSPAR: Engineering Implementation Blueprint

## 1. Project Structure

```
BSPAR/
├── bspar/
│   ├── __init__.py
│   ├── config.py                  # Dataclass-based unified config
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py             # Shared contextual encoder wrapper
│   │   ├── span_proposal.py       # Module A: span repr + unary scoring + NULL prototypes
│   │   ├── pair_module.py         # Module B: pair construction + category + affective heads
│   │   ├── quad_reranker.py       # Module C: quad-aware reranker
│   │   ├── bspar_stage1.py        # Stage-1 model (encoder + A + B)
│   │   └── bspar_stage2.py        # Stage-2 model (reranker, shares or reloads encoder)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py              # Core data structures (Span, Pair, Quad, Example)
│   │   ├── dataset.py             # Stage-1 dataset: tokenization + span label assignment
│   │   ├── rerank_dataset.py      # Stage-2 dataset: real candidate quad loading
│   │   ├── preprocessor.py        # Raw data → unified format converter
│   │   ├── span_utils.py          # Span enumeration, width bucket, distance bucket
│   │   └── hard_negatives.py      # Hard negative pair/quad construction logic
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── span_loss.py           # L_span: focal BCE for dual binary heads
│   │   ├── pair_loss.py           # L_pair: BCE with hard-negative weighting
│   │   ├── category_loss.py       # L_cat: CE conditioned on valid pairs
│   │   ├── affective_loss.py      # L_aff: sentiment CE or Smooth-L1 regression
│   │   ├── ranking_loss.py        # L_rank: pairwise margin ranking loss
│   │   └── multitask.py           # Weighted combination: L = Σ λᵢ Lᵢ
│   │
│   ├── decode/
│   │   ├── __init__.py
│   │   ├── span_pruner.py         # Top-K span pruning logic
│   │   ├── pair_decoder.py        # Pair candidate construction + filtering
│   │   ├── quad_decoder.py        # Quad candidate expansion + NMS dedup
│   │   └── output_formatter.py    # Final quad → text span recovery + format
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── stage1_trainer.py      # Stage-1 training loop (encoder + span + pair + MTL)
│   │   ├── stage2_trainer.py      # Stage-2 training loop (reranker on real candidates)
│   │   ├── candidate_generator.py # Run Stage-1 inference → dump real candidates for Stage-2
│   │   └── scheduler.py           # LR scheduler, warmup, etc.
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Quad-level F1 (exact match), pair F1, span F1
│   │   ├── analysis.py            # Implicit-case, multi-pair, span-length breakdown
│   │   └── error_taxonomy.py      # Error categorization for error analysis
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py                  # File I/O, checkpoint save/load
│       ├── logging.py             # Structured logging
│       └── seed.py                # Reproducibility: seed setting
│
├── configs/
│   ├── asqp_rest15.yaml           # ASQP dataset-specific config
│   ├── asqp_rest16.yaml
│   ├── asqp_lapt14.yaml
│   └── dimabsa.yaml               # dimABSA config (continuous affective)
│
├── scripts/
│   ├── train_stage1.py            # Entry: python scripts/train_stage1.py --config ...
│   ├── generate_candidates.py     # Entry: dump Stage-1 candidates for Stage-2
│   ├── train_stage2.py            # Entry: train reranker
│   ├── predict.py                 # Entry: full pipeline inference
│   └── run_experiment.py          # Multi-seed experiment runner
│
├── tests/
│   ├── test_span_proposal.py
│   ├── test_pair_module.py
│   ├── test_quad_reranker.py
│   ├── test_decode.py
│   └── test_metrics.py
│
├── docs/
│   ├── method_overview.md
│   └── engineering_blueprint.md   # This document
│
└── requirements.txt
```

### Design Principles
- **One module = one paper section**: each model file corresponds to a section in the Method
- **Stage separation is explicit**: Stage-1 and Stage-2 are separate model classes with separate trainers
- **Data structures are shared**: `schema.py` defines the canonical data types used everywhere
- **Losses are modular**: each loss in its own file, combined in `multitask.py`
- **Decode is separate from model**: model outputs logits/scores; decode handles thresholding, pruning, NMS

---

## 2. Core Data Structures (`bspar/data/schema.py`)

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Span:
    """A text span or NULL indicator."""
    start: int          # token index, -1 for NULL
    end: int            # token index (inclusive), -1 for NULL
    text: str           # surface form, "NULL" for implicit
    is_null: bool = False

    @property
    def length(self) -> int:
        if self.is_null:
            return 0
        return self.end - self.start + 1

    @staticmethod
    def null(role: str = "") -> "Span":
        return Span(start=-1, end=-1, text="NULL", is_null=True)


@dataclass
class Quad:
    """A single aspect sentiment quadruple."""
    aspect: Span
    opinion: Span
    category: str
    sentiment: Optional[str] = None       # POS/NEG/NEU for ASQP
    valence: Optional[float] = None       # for dimABSA
    arousal: Optional[float] = None       # for dimABSA


@dataclass
class Example:
    """A single training/inference example."""
    id: str
    text: str
    tokens: list[str] = field(default_factory=list)
    token_offsets: list[tuple[int, int]] = field(default_factory=list)
    quads: list[Quad] = field(default_factory=list)  # gold quads (empty at inference)


@dataclass
class SpanCandidate:
    """A scored span candidate from Module A."""
    span: Span
    repr: object        # tensor, filled at runtime
    asp_score: float    # φ_asp(s)
    opn_score: float    # φ_opn(s)
    role: str           # "aspect" | "opinion" | "null_aspect" | "null_opinion"


@dataclass
class PairCandidate:
    """A scored pair candidate from Module B."""
    aspect: SpanCandidate
    opinion: SpanCandidate
    pair_repr: object           # tensor
    pair_score: float           # P_pair(a,o)
    category_probs: object      # tensor [|C|]
    category_pred: str
    affective_pred: object      # sentiment label or (v, ar)


@dataclass
class QuadCandidate:
    """A fully expanded quad candidate for reranking."""
    pair: PairCandidate
    category: str
    affective: object           # label or (v, ar)
    quad_score: float = 0.0     # S(q), filled by reranker
    # Meta features for reranker input
    asp_unary: float = 0.0
    opn_unary: float = 0.0
    pair_validity: float = 0.0
    cat_prob: float = 0.0
    has_null_aspect: bool = False
    has_null_opinion: bool = False

    def to_quad(self) -> Quad:
        """Convert to final Quad output."""
        return Quad(
            aspect=self.pair.aspect.span,
            opinion=self.pair.opinion.span,
            category=self.category,
            sentiment=self.affective if isinstance(self.affective, str) else None,
            valence=self.affective[0] if isinstance(self.affective, tuple) else None,
            arousal=self.affective[1] if isinstance(self.affective, tuple) else None,
        )
```

---

## 3. Model Class Design

### 3.1 Encoder (`bspar/models/encoder.py`)

```python
class SharedEncoder(nn.Module):
    """Wraps a pretrained transformer as the shared backbone."""

    def __init__(self, model_name: str, finetune: bool = True):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size
        if not finetune:
            for p in self.transformer.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Returns:
            H: (batch, seq_len, hidden_size) — contextualized token representations
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.last_hidden_state  # H
```

### 3.2 Span Proposal (`bspar/models/span_proposal.py`)

```python
class SpanProposal(nn.Module):
    """
    Module A: Structured Span Proposal with NULL Prototypes.

    Responsibilities:
    1. Enumerate all spans up to max_span_length
    2. Compute span representations via boundary + attn-pool + width embedding
    3. Score each span as aspect candidate and opinion candidate (dual binary heads)
    4. Generate context-conditioned NULL prototypes
    """

    def __init__(self, hidden_size: int, max_span_length: int,
                 span_repr_size: int, width_embedding_dim: int):
        super().__init__()
        self.max_span_length = max_span_length

        # Width embedding
        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)

        # Attention pooling
        self.attn_proj = nn.Linear(hidden_size, 1)

        # Span representation projection
        # Input: [h_start; h_end; attn_pool; e_width]
        raw_dim = hidden_size * 3 + width_embedding_dim
        self.span_proj = nn.Sequential(
            nn.Linear(raw_dim, span_repr_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.span_repr_size = span_repr_size

        # Dual binary scoring heads
        self.asp_scorer = nn.Linear(span_repr_size, 1)
        self.opn_scorer = nn.Linear(span_repr_size, 1)

        # NULL prototype generators (context-conditioned)
        self.null_asp_proj = nn.Sequential(
            nn.Linear(hidden_size, span_repr_size),
            nn.Tanh(),
        )
        self.null_opn_proj = nn.Sequential(
            nn.Linear(hidden_size, span_repr_size),
            nn.Tanh(),
        )

    def _enumerate_spans(self, seq_len: int, device):
        """Generate all (start, end) pairs with length <= max_span_length."""
        spans = []
        for i in range(seq_len):
            for j in range(i, min(i + self.max_span_length, seq_len)):
                spans.append((i, j))
        return spans  # list of (start, end)

    def _compute_span_repr(self, H, spans):
        """
        H: (batch, seq_len, hidden)
        spans: list of (start, end) — shared across batch
        Returns: (batch, num_spans, span_repr_size)
        """
        batch_size = H.size(0)
        span_reprs = []

        for (i, j) in spans:
            h_start = H[:, i, :]                          # (batch, hidden)
            h_end = H[:, j, :]                            # (batch, hidden)

            # Attention pooling over span tokens
            span_tokens = H[:, i:j+1, :]                  # (batch, span_len, hidden)
            attn_weights = self.attn_proj(span_tokens)     # (batch, span_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            attn_pooled = (attn_weights * span_tokens).sum(dim=1)  # (batch, hidden)

            # Width embedding
            width = j - i + 1
            e_w = self.width_embedding(
                torch.tensor(width, device=H.device)
            ).unsqueeze(0).expand(batch_size, -1)          # (batch, width_dim)

            raw = torch.cat([h_start, h_end, attn_pooled, e_w], dim=-1)
            span_reprs.append(raw)

        span_reprs = torch.stack(span_reprs, dim=1)        # (batch, num_spans, raw_dim)
        return self.span_proj(span_reprs)                   # (batch, num_spans, span_repr_size)

    def forward(self, H, attention_mask):
        """
        Args:
            H: (batch, seq_len, hidden) from encoder
            attention_mask: (batch, seq_len)
        Returns:
            span_reprs: (batch, num_spans, span_repr_size)
            asp_scores: (batch, num_spans) — aspect unary scores
            opn_scores: (batch, num_spans) — opinion unary scores
            null_asp_repr: (batch, span_repr_size) — NULL aspect prototype
            null_opn_repr: (batch, span_repr_size) — NULL opinion prototype
            span_indices: list of (start, end) tuples
        """
        seq_len = H.size(1)
        span_indices = self._enumerate_spans(seq_len, H.device)

        # Span representations
        span_reprs = self._compute_span_repr(H, span_indices)

        # Dual unary scores
        asp_scores = self.asp_scorer(span_reprs).squeeze(-1)   # (batch, num_spans)
        opn_scores = self.opn_scorer(span_reprs).squeeze(-1)   # (batch, num_spans)

        # NULL prototypes — conditioned on sentence-level mean
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        h_mean = (H * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths  # (batch, hidden)
        null_asp_repr = self.null_asp_proj(h_mean)   # (batch, span_repr_size)
        null_opn_repr = self.null_opn_proj(h_mean)   # (batch, span_repr_size)

        return {
            "span_reprs": span_reprs,
            "asp_scores": asp_scores,
            "opn_scores": opn_scores,
            "null_asp_repr": null_asp_repr,
            "null_opn_repr": null_opn_repr,
            "span_indices": span_indices,
        }
```

### 3.3 Pair Module (`bspar/models/pair_module.py`)

```python
class PairModule(nn.Module):
    """
    Module B: Pair Construction with Multi-Task Prediction Heads.

    Takes pruned aspect/opinion span representations (including NULLs),
    constructs pair representations, and predicts:
    1. Pair validity (binary)
    2. Category distribution
    3. Affective output (sentiment polarity or valence-arousal)
    """

    def __init__(self, span_repr_size: int, num_categories: int,
                 num_sentiments: int = 3, dist_buckets: int = 16,
                 order_types: int = 3, task_type: str = "asqp"):
        super().__init__()
        self.task_type = task_type

        # Distance and order embeddings
        self.dist_embedding = nn.Embedding(dist_buckets, 32)
        self.order_embedding = nn.Embedding(order_types, 16)

        # Pair representation: [r_a; r_o; r_a⊙r_o; |r_a-r_o|; e_dist; e_order]
        pair_input_dim = span_repr_size * 4 + 32 + 16
        pair_hidden = span_repr_size * 2

        self.pair_proj = nn.Sequential(
            nn.Linear(pair_input_dim, pair_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pair_repr_size = pair_hidden

        # Prediction heads
        self.pair_scorer = nn.Linear(pair_hidden, 1)         # P_pair
        self.cat_head = nn.Linear(pair_hidden, num_categories)  # P(c|a,o)

        if task_type == "asqp":
            self.aff_head = nn.Linear(pair_hidden, num_sentiments)  # P(y|a,o)
        else:  # dimabsa
            self.aff_head = nn.Linear(pair_hidden, 2)  # (valence, arousal)

    def _compute_distance_bucket(self, asp_span, opn_span):
        """Compute bucketed distance between aspect and opinion spans."""
        # Returns bucket index; NULL spans get a special bucket
        ...

    def _compute_order(self, asp_span, opn_span):
        """0: aspect before opinion, 1: opinion before aspect, 2: involves NULL."""
        ...

    def forward(self, asp_reprs, opn_reprs, asp_spans, opn_spans,
                dist_ids, order_ids):
        """
        Args:
            asp_reprs:  (batch, num_pairs, span_repr_size) — pruned aspect reprs
            opn_reprs:  (batch, num_pairs, span_repr_size) — matched opinion reprs
            dist_ids:   (batch, num_pairs) — distance bucket indices
            order_ids:  (batch, num_pairs) — order type indices
        Returns:
            pair_reprs:  (batch, num_pairs, pair_repr_size)
            pair_scores: (batch, num_pairs)     — P_pair
            cat_logits:  (batch, num_pairs, |C|) — category logits
            aff_output:  (batch, num_pairs, ?)   — sentiment logits or (v, ar)
        """
        # Element-wise interaction
        hadamard = asp_reprs * opn_reprs
        abs_diff = (asp_reprs - opn_reprs).abs()
        e_dist = self.dist_embedding(dist_ids)
        e_order = self.order_embedding(order_ids)

        # Concatenate
        pair_input = torch.cat([
            asp_reprs, opn_reprs, hadamard, abs_diff, e_dist, e_order
        ], dim=-1)

        pair_reprs = self.pair_proj(pair_input)

        # Predictions
        pair_scores = self.pair_scorer(pair_reprs).squeeze(-1)
        cat_logits = self.cat_head(pair_reprs)
        aff_output = self.aff_head(pair_reprs)

        return {
            "pair_reprs": pair_reprs,
            "pair_scores": pair_scores,
            "cat_logits": cat_logits,
            "aff_output": aff_output,
        }
```

### 3.4 Quad Reranker (`bspar/models/quad_reranker.py`)

```python
class QuadReranker(nn.Module):
    """
    Module C: Quad-Aware Reranker.

    Takes candidate quads (from Stage-1 real outputs), computes a unified
    quad-level score S(q) that integrates pair representation, category
    embedding, affective embedding, and meta features.

    Key design: trained on real Stage-1 candidates, NOT oracle candidates.
    """

    def __init__(self, pair_repr_size: int, cat_embedding_dim: int,
                 aff_embedding_dim: int, num_categories: int,
                 num_meta_features: int, task_type: str = "asqp"):
        super().__init__()

        # Category and affective embeddings for quad-level reasoning
        self.cat_embedding = nn.Embedding(num_categories, cat_embedding_dim)

        if task_type == "asqp":
            self.aff_embedding = nn.Embedding(3, aff_embedding_dim)  # POS/NEG/NEU
        else:
            self.aff_proj = nn.Linear(2, aff_embedding_dim)  # (v, ar) → embedding

        # Meta features: asp_score, opn_score, pair_score, cat_prob,
        #   cat_entropy, has_null_a, has_null_o, asp_len, opn_len
        self.meta_proj = nn.Linear(num_meta_features, 32)

        # Quad scorer
        quad_input_dim = pair_repr_size + cat_embedding_dim + aff_embedding_dim + 32
        self.quad_scorer = nn.Sequential(
            nn.Linear(quad_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, pair_reprs, cat_ids, aff_input, meta_features):
        """
        Args:
            pair_reprs:    (batch, num_cands, pair_repr_size)
            cat_ids:       (batch, num_cands) — predicted category indices
            aff_input:     (batch, num_cands) for ASQP or (batch, num_cands, 2) for dimABSA
            meta_features: (batch, num_cands, num_meta_features)
        Returns:
            quad_scores:   (batch, num_cands) — S(q) for each candidate
        """
        e_cat = self.cat_embedding(cat_ids)
        if hasattr(self, 'aff_embedding'):
            e_aff = self.aff_embedding(aff_input)
        else:
            e_aff = self.aff_proj(aff_input)
        e_meta = self.meta_proj(meta_features)

        quad_repr = torch.cat([pair_reprs, e_cat, e_aff, e_meta], dim=-1)
        quad_scores = self.quad_scorer(quad_repr).squeeze(-1)

        return quad_scores
```

### 3.5 Stage-1 Composite Model (`bspar/models/bspar_stage1.py`)

```python
class BSPARStage1(nn.Module):
    """
    Stage-1: Encoder + Span Proposal + Pair Module.

    This is the main model for Stage-1 training.
    At inference time, it produces candidate quads for Stage-2.
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = SharedEncoder(config.model_name)
        self.span_proposal = SpanProposal(
            hidden_size=self.encoder.hidden_size,
            max_span_length=config.max_span_length,
            span_repr_size=config.span_repr_size,
            width_embedding_dim=config.width_embedding_dim,
        )
        self.pair_module = PairModule(
            span_repr_size=config.span_repr_size,
            num_categories=config.num_categories,
            num_sentiments=config.num_sentiments,
            task_type=config.task_type,
        )
        self.config = config

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                gold_quads=None, mode="train"):
        """
        Training mode:  returns losses dict
        Inference mode: returns candidate quads with scores
        """
        # Step 1: Encode
        H = self.encoder(input_ids, attention_mask, token_type_ids)

        # Step 2: Span proposal
        proposal_out = self.span_proposal(H, attention_mask)

        # Step 3: Prune spans → build pair candidates
        pruned_aspects, pruned_opinions = self._prune_spans(proposal_out)

        # Step 4: Construct pairs (Cartesian product of pruned sets)
        pair_inputs = self._construct_pairs(pruned_aspects, pruned_opinions)

        # Step 5: Pair + MTL prediction
        pair_out = self.pair_module(**pair_inputs)

        if mode == "train":
            return self._compute_losses(proposal_out, pair_out, gold_quads)
        else:
            return self._build_candidates(proposal_out, pair_out)

    def _prune_spans(self, proposal_out):
        """Keep top-K_a aspects and top-K_o opinions by unary score."""
        asp_scores = proposal_out["asp_scores"]
        opn_scores = proposal_out["opn_scores"]
        span_reprs = proposal_out["span_reprs"]

        # Top-K for aspects
        K_a = min(self.config.top_k_aspects, asp_scores.size(1))
        asp_topk = torch.topk(asp_scores, K_a, dim=1)

        # Top-K for opinions
        K_o = min(self.config.top_k_opinions, opn_scores.size(1))
        opn_topk = torch.topk(opn_scores, K_o, dim=1)

        # Gather pruned span representations
        pruned_asp_reprs = torch.gather(
            span_reprs, 1,
            asp_topk.indices.unsqueeze(-1).expand(-1, -1, span_reprs.size(-1))
        )
        pruned_opn_reprs = torch.gather(
            span_reprs, 1,
            opn_topk.indices.unsqueeze(-1).expand(-1, -1, span_reprs.size(-1))
        )

        # Append NULL prototypes
        null_asp = proposal_out["null_asp_repr"].unsqueeze(1)  # (batch, 1, dim)
        null_opn = proposal_out["null_opn_repr"].unsqueeze(1)
        pruned_asp_reprs = torch.cat([pruned_asp_reprs, null_asp], dim=1)
        pruned_opn_reprs = torch.cat([pruned_opn_reprs, null_opn], dim=1)

        return pruned_asp_reprs, pruned_opn_reprs

    def _construct_pairs(self, asp_reprs, opn_reprs):
        """Cartesian product of aspect × opinion candidates, excluding NULL×NULL."""
        ...

    def _compute_losses(self, proposal_out, pair_out, gold_quads):
        """Compute L_span + L_pair + L_cat + L_aff."""
        ...

    def _build_candidates(self, proposal_out, pair_out):
        """Build QuadCandidate list for Stage-2 reranking."""
        ...
```

### 3.6 Stage-2 Composite Model (`bspar/models/bspar_stage2.py`)

```python
class BSPARStage2(nn.Module):
    """
    Stage-2: Quad-Aware Reranker.

    Input:  pre-computed candidate quads from Stage-1 (real candidates)
    Output: reranked quad scores

    Training: uses real candidates dumped by Stage-1, NOT oracle candidates.
    """

    def __init__(self, config):
        super().__init__()
        self.reranker = QuadReranker(
            pair_repr_size=config.pair_repr_size,
            cat_embedding_dim=config.cat_embedding_dim,
            aff_embedding_dim=config.aff_embedding_dim,
            num_categories=config.num_categories,
            num_meta_features=config.num_meta_features,
            task_type=config.task_type,
        )

    def forward(self, pair_reprs, cat_ids, aff_input, meta_features,
                labels=None, mode="train"):
        """
        Training: returns ranking loss
        Inference: returns quad scores for final decode
        """
        quad_scores = self.reranker(pair_reprs, cat_ids, aff_input, meta_features)

        if mode == "train" and labels is not None:
            loss = self._compute_ranking_loss(quad_scores, labels)
            return {"loss": loss, "quad_scores": quad_scores}
        return {"quad_scores": quad_scores}

    def _compute_ranking_loss(self, scores, labels):
        """
        Pairwise margin ranking loss.
        labels: (batch, num_cands) — 1 for gold-matched, 0 for negative
        """
        ...
```

---

## 4. Forward Flow (End-to-End)

### 4.1 Stage-1 Training Forward

```
input_ids, attention_mask, gold_quads
        │
        ▼
┌─ SharedEncoder ─────────────────────┐
│  H = encoder(input_ids, attn_mask)  │
└──────────┬──────────────────────────┘
           │
           ▼
┌─ SpanProposal ──────────────────────────────────────────┐
│  1. Enumerate spans (i,j) with j-i+1 ≤ L               │
│  2. Compute span_reprs via [h_i; h_j; attn_pool; e_w]  │
│  3. Score: asp_scores, opn_scores (dual binary)         │
│  4. Generate null_asp_repr, null_opn_repr               │
│  5. L_span = focal_bce(asp_scores, asp_labels)          │
│           + focal_bce(opn_scores, opn_labels)           │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─ Span Pruning ──────────────────────────────────────────┐
│  • During training: use gold spans + top-K predicted    │
│    (teacher-forced augmented with predicted for         │
│     consistency — see §4.3 Training Tricks)             │
│  • During inference: use top-K predicted only           │
│  • Always append NULL prototypes                        │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─ PairModule ────────────────────────────────────────────┐
│  1. Construct pairs: aspect × opinion (excl NULL×NULL)  │
│  2. Compute pair_reprs                                  │
│  3. Predict: pair_scores, cat_logits, aff_output        │
│  4. L_pair = bce_hard_neg(pair_scores, pair_labels)     │
│  5. L_cat  = ce(cat_logits[valid], cat_labels[valid])   │
│  6. L_aff  = ce_or_smooth_l1(aff_out[valid], aff_lbl)  │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
   L_stage1 = λ₁·L_span + λ₂·L_pair + λ₃·L_cat + λ₄·L_aff
```

### 4.2 Candidate Generation (Bridge Between Stages)

```
Trained Stage-1 model
        │
        ▼
For each example in train+dev set:
  1. Run Stage-1 in inference mode
  2. Collect all candidate quads with:
     - pair_repr (detached tensor, saved to disk)
     - predicted category, affective output
     - meta features: asp_score, opn_score, pair_score,
       cat_prob, cat_entropy, has_null_a, has_null_o, ...
  3. Label each candidate:
     - positive: exact match with some gold quad
     - negative: partial match or no match
  4. Save to disk as .pt / .jsonl files

Output: rerank_train.pt, rerank_dev.pt
```

**Critical**: this step runs **Stage-1 in inference mode on training data** to generate realistic candidates. The distribution of these candidates (noise level, error patterns) is what the reranker will learn from.

### 4.3 Stage-2 Training Forward

```
pair_reprs, cat_ids, aff_input, meta_features, labels
        │
        ▼
┌─ QuadReranker ──────────────────────────────────────────┐
│  1. Embed category: e_cat                               │
│  2. Embed affective: e_aff                              │
│  3. Project meta: e_meta                                │
│  4. quad_repr = [pair_repr; e_cat; e_aff; e_meta]       │
│  5. quad_score = FFN(quad_repr)                         │
│  6. L_rank = margin_ranking(scores, labels)             │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
   L_stage2 = L_rank
```

### 4.4 Full Inference Pipeline

```
input text
    │
    ▼
Tokenize → input_ids, attention_mask
    │
    ▼
Stage-1 model.forward(mode="inference")
    │
    ├── span_proposal → top-K aspects + opinions + NULLs
    ├── pair_module → candidate pairs with cat/aff predictions
    └── expand → candidate quads with meta features
    │
    ▼
Stage-2 model.forward(mode="inference")
    │
    ├── quad_scores for each candidate
    └── sort by S(q) descending
    │
    ▼
Decode:
    1. Sort candidates by quad_score
    2. Apply score threshold: keep S(q) > τ
    3. NMS deduplication: remove quads that overlap
       with a higher-scored quad (same aspect span or
       same opinion span with same category)
    4. Output remaining quads
    │
    ▼
Format output → list of (aspect_text, opinion_text, category, sentiment/VA)
```

---

## 5. Training Schedule

### Recommended: Sequential Two-Phase (Stable, Clear for Paper)

```
Phase A: Stage-1 Training
├── Train encoder + SpanProposal + PairModule jointly
├── Loss: L_stage1 = λ₁·L_span + λ₂·L_pair + λ₃·L_cat + λ₄·L_aff
├── Optimizer: AdamW, lr=2e-5 (encoder), lr=1e-4 (task heads)
├── Epochs: ~20, with early stopping on dev quad F1
├── Span pruning during training: gold spans ∪ top-K predicted (mixed)
└── Output: best_stage1.pt

Phase A→B Bridge: Candidate Generation
├── Load best_stage1.pt
├── Run inference on train set → dump real candidates
├── Run inference on dev set → dump real candidates
├── Label candidates against gold quads
└── Output: rerank_train.pt, rerank_dev.pt

Phase B: Stage-2 Training (Reranker)
├── Train QuadReranker on real candidates from Phase A output
├── Loss: L_rank (pairwise margin)
├── Optimizer: AdamW, lr=5e-4
├── Epochs: ~10, with early stopping on dev quad F1
├── Encoder is FROZEN (reranker only uses pre-computed pair_reprs)
└── Output: best_stage2.pt

Phase C (Optional): Joint Fine-Tuning
├── Load best_stage1.pt + best_stage2.pt
├── End-to-end fine-tune with L_stage1 + λ₅·L_rank
├── Lower LR: 5e-6 (encoder), 2e-5 (heads), 1e-4 (reranker)
├── Epochs: ~5
└── NOTE: Report as optional; main results use Phase A+B
```

**Why this schedule:**
- **Stability**: Stage-2 sees stable candidates from a converged Stage-1
- **Ablation-friendly**: Can ablate Stage-2 by skipping Phase B entirely
- **Reproducibility**: Clear separation, deterministic candidate dump
- **Paper clarity**: Each phase corresponds to a clear methodological component

---

## 6. Loss Computation Interface

### `bspar/losses/multitask.py`

```python
class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss combiner for Stage-1."""

    def __init__(self, lambda_span=1.0, lambda_pair=1.0,
                 lambda_cat=1.0, lambda_aff=0.5):
        super().__init__()
        self.span_loss = SpanFocalLoss(alpha=0.25, gamma=2.0)
        self.pair_loss = PairBCELoss(hard_neg_weight=3.0)
        self.cat_loss = CategoryCELoss(label_smoothing=0.1)
        self.aff_loss = None  # set based on task_type
        self.lambdas = {
            "span": lambda_span,
            "pair": lambda_pair,
            "cat": lambda_cat,
            "aff": lambda_aff,
        }

    def forward(self, predictions, targets):
        """
        predictions: dict with asp_scores, opn_scores, pair_scores,
                     cat_logits, aff_output
        targets: dict with asp_labels, opn_labels, pair_labels,
                 cat_labels, aff_labels, valid_pair_mask
        """
        losses = {}
        losses["span"] = self.span_loss(
            predictions["asp_scores"], targets["asp_labels"],
            predictions["opn_scores"], targets["opn_labels"],
        )
        losses["pair"] = self.pair_loss(
            predictions["pair_scores"], targets["pair_labels"],
            targets.get("hard_neg_mask"),
        )
        # Category and affective losses only on valid (gold-positive) pairs
        valid = targets["valid_pair_mask"]
        if valid.any():
            losses["cat"] = self.cat_loss(
                predictions["cat_logits"][valid],
                targets["cat_labels"][valid],
            )
            losses["aff"] = self.aff_loss(
                predictions["aff_output"][valid],
                targets["aff_labels"][valid],
            )
        else:
            losses["cat"] = torch.tensor(0.0, device=valid.device)
            losses["aff"] = torch.tensor(0.0, device=valid.device)

        total = sum(self.lambdas[k] * losses[k] for k in losses)
        losses["total"] = total
        return losses
```

---

## 7. Reranker Real-Candidate Training Flow (Detailed)

This is the **most methodologically important** engineering component, as it directly
supports Contribution C3.

### 7.1 Candidate Generation Pipeline (`bspar/training/candidate_generator.py`)

```python
class CandidateGenerator:
    """
    Generates real candidate quads from a trained Stage-1 model.

    This is the bridge between Stage-1 and Stage-2.
    Key principle: candidates reflect the ACTUAL noise distribution
    of Stage-1, not an idealized oracle.
    """

    def __init__(self, stage1_model, config):
        self.model = stage1_model
        self.config = config

    def generate(self, dataloader, gold_examples):
        """
        For each example:
        1. Run Stage-1 inference → get candidate quads
        2. Match each candidate against gold quads
        3. Assign labels: 1 (exact match) or 0 (no match)
        4. Collect pair_reprs + meta features

        Returns: list of RerankExample
        """
        all_rerank_examples = []

        for batch, golds in zip(dataloader, gold_examples):
            with torch.no_grad():
                outputs = self.model(**batch, mode="inference")

            for i, (cands, gold) in enumerate(zip(outputs["candidates"], golds)):
                rerank_ex = RerankExample(example_id=gold.id)

                for cand in cands:
                    label = self._match_gold(cand, gold.quads)
                    rerank_ex.add_candidate(
                        pair_repr=cand.pair_repr.detach().cpu(),
                        cat_id=cand.category_id,
                        aff_input=cand.affective_pred,
                        meta=self._extract_meta(cand),
                        label=label,
                    )

                # Ensure at least one positive and one negative per example
                if rerank_ex.has_positive and rerank_ex.has_negative:
                    all_rerank_examples.append(rerank_ex)

        return all_rerank_examples

    def _match_gold(self, candidate, gold_quads):
        """Exact match: aspect span + opinion span + category + sentiment."""
        for gq in gold_quads:
            if (candidate.aspect_span == (gq.aspect.start, gq.aspect.end) and
                candidate.opinion_span == (gq.opinion.start, gq.opinion.end) and
                candidate.category == gq.category and
                self._aff_match(candidate.affective_pred, gq)):
                return 1
        return 0

    def _extract_meta(self, cand):
        """Extract meta features for quad-level reasoning."""
        return [
            cand.asp_unary_score,
            cand.opn_unary_score,
            cand.pair_score,
            cand.cat_prob,           # P(predicted_cat | a, o)
            cand.cat_entropy,        # entropy of category distribution
            float(cand.has_null_aspect),
            float(cand.has_null_opinion),
            cand.asp_length,
            cand.opn_length,
        ]
```

### 7.2 What Makes This Different from Oracle Reranking

| Aspect | Oracle Reranking | Real-Candidate Reranking (Ours) |
|--------|-----------------|----------------------------------|
| Positive candidates | Constructed from gold spans + gold labels | Stage-1 predictions that happen to match gold |
| Negative candidates | Random or heuristic pairs from gold spans | Stage-1 predictions that don't match gold |
| Noise distribution | Clean, unrealistic | Matches actual test-time noise |
| Boundary errors | Not present | Present (Stage-1 may predict near-miss spans) |
| Category errors | Not present | Present (Stage-1 may predict wrong category) |
| Training signal | Learns to distinguish clean candidates | Learns to distinguish noisy, realistic candidates |

### 7.3 Hard Negative Construction for Reranker

Within the real candidates, we further categorize negatives by difficulty:

```python
class HardNegativeLabeler:
    """Categorize negative candidates by overlap with gold."""

    def label_difficulty(self, candidate, gold_quads):
        """
        Returns:
            0: easy negative (no element matches any gold)
            1: partial match (1-2 elements match some gold quad)
            2: hard negative (3 elements match, 1 wrong)
        """
        max_overlap = 0
        for gq in gold_quads:
            overlap = 0
            if self._span_match(candidate.aspect, gq.aspect): overlap += 1
            if self._span_match(candidate.opinion, gq.opinion): overlap += 1
            if candidate.category == gq.category: overlap += 1
            if self._aff_match(candidate, gq): overlap += 1
            max_overlap = max(max_overlap, overlap)

        if max_overlap >= 3:
            return 2  # hard
        elif max_overlap >= 1:
            return 1  # partial
        else:
            return 0  # easy
```

---

## 8. Decode Logic (`bspar/decode/`)

### 8.1 Span Pruning (`span_pruner.py`)

```
Input:  asp_scores (num_spans,), opn_scores (num_spans,)
Params: K_a, K_o, score_threshold

1. Filter by threshold: keep spans with score > threshold
2. Take top-K_a for aspects, top-K_o for opinions
3. Append NULL prototypes
4. Return: pruned_aspects (≤ K_a+1), pruned_opinions (≤ K_o+1)
```

### 8.2 Pair Construction (`pair_decoder.py`)

```
Input: pruned_aspects, pruned_opinions
Rules:
1. Cartesian product: every aspect × every opinion
2. EXCLUDE: NULL_aspect × NULL_opinion pairs
3. EXCLUDE: self-pairing (same span as both aspect and opinion)
4. Return: candidate_pairs (≤ (K_a+1)×(K_o+1) - 1)
```

### 8.3 Quad Expansion (`quad_decoder.py`)

```
Input: candidate_pairs with cat_logits, aff_output
Params: top_c (number of category candidates per pair)

1. For each pair, take top-c categories by probability
2. For ASQP: take argmax sentiment per pair-category combo
   For dimABSA: take predicted (v, ar) values
3. Expand: each pair × top-c categories = candidate quads
4. Attach meta features
5. Return: candidate_quads for reranker
```

### 8.4 NMS Deduplication

```
Input: reranked quads sorted by S(q) descending
Params: overlap_threshold

1. Initialize: selected = []
2. For each quad q in sorted order:
   a. Check overlap with all quads in selected:
      - Same aspect span AND same category → suppress
      - Same opinion span AND same category → suppress
   b. If not suppressed: add q to selected
3. Return: selected quads as final output
```

### 8.5 Output Formatting (`output_formatter.py`)

```
Input: selected QuadCandidates, original tokens, token-to-char offsets

1. For each quad:
   a. Aspect text: tokens[start:end+1] joined, or "NULL"
   b. Opinion text: tokens[start:end+1] joined, or "NULL"
   c. Category: string label
   d. Sentiment: POS/NEG/NEU or (v, ar)
   e. For dimABSA: clip v, ar to [1.0, 5.0]
2. Return: list of formatted quads
```

---

## 9. Configuration Design (`bspar/config.py`)

```python
from dataclasses import dataclass

@dataclass
class BSPARConfig:
    # === Task ===
    task_type: str = "asqp"             # "asqp" or "dimabsa"
    num_categories: int = 13            # dataset-dependent
    num_sentiments: int = 3             # POS, NEG, NEU (ASQP only)

    # === Encoder ===
    model_name: str = "roberta-base"
    finetune_encoder: bool = True
    hidden_size: int = 768              # auto-detected from model

    # === Span Proposal ===
    max_span_length: int = 8
    span_repr_size: int = 256
    width_embedding_dim: int = 32
    top_k_aspects: int = 20
    top_k_opinions: int = 20
    span_score_threshold: float = 0.0

    # === Pair Module ===
    dist_buckets: int = 16
    order_types: int = 3                # asp_first, opn_first, has_null
    top_c_categories: int = 3           # categories per pair for quad expansion

    # === Quad Reranker ===
    cat_embedding_dim: int = 32
    aff_embedding_dim: int = 32
    num_meta_features: int = 9
    pair_repr_size: int = 512           # = span_repr_size * 2 from pair_proj

    # === Loss Weights ===
    lambda_span: float = 1.0
    lambda_pair: float = 1.0
    lambda_cat: float = 1.0
    lambda_aff: float = 0.5
    lambda_rank: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    hard_neg_weight: float = 3.0
    ranking_margin: float = 1.0

    # === Training ===
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    reranker_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    stage1_epochs: int = 20
    stage2_epochs: int = 10
    batch_size: int = 16
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    patience: int = 5                   # early stopping patience

    # === Decode ===
    quad_score_threshold: float = 0.0
    nms_overlap_suppress: bool = True

    # === Reproducibility ===
    seeds: list = None                  # e.g., [42, 123, 456, 789, 1024]

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1024]
        self.pair_repr_size = self.span_repr_size * 2
```

---

## 10. Potential Engineering Pitfalls

| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Span enumeration OOM for long sequences | O(n·L) spans × batch → GPU memory explosion | Cap n at 128 tokens; use efficient batched indexing |
| Pair Cartesian product explosion | K_a × K_o can be large | Keep K_a, K_o ≤ 20; filter by score threshold first |
| Stage-1 candidate quality too low on first epochs | Reranker gets garbage input if Stage-1 not converged | Only generate candidates from best Stage-1 checkpoint |
| Label leakage in candidate generation | If gold spans are mixed in during candidate gen for train | Candidate generation must use pure inference mode |
| Imbalanced pos/neg in reranker training | Vast majority of candidates are negative | Subsample negatives per example (e.g., max 50); weight hard negatives higher |
| NULL prototype gradient vanishing | NULL branch sees fewer examples | Ensure NULL-involved pairs are not dropped during sampling |
| Tokenizer offset mismatch | Subword tokenization shifts span boundaries | Maintain strict token-to-char offset mapping; test with edge cases |
