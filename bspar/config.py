"""Unified configuration for BSPAR."""

from dataclasses import dataclass, field


@dataclass
class BSPARConfig:
    """All hyperparameters and settings for BSPAR."""

    # === Task ===
    task_type: str = "asqp"             # "asqp" or "dimabsa"
    dataset_name: str = "asqp_rest15"   # asqp_rest15/rest16, acos_laptop/restaurant
    num_categories: int = 12            # dataset-dependent, auto-set from dataset
    num_sentiments: int = 3             # POS, NEG, NEU (ASQP only)

    # === Encoder ===
    model_name: str = "roberta-base"
    finetune_encoder: bool = True
    hidden_size: int = 768              # auto-detected from model

    # === Span Proposal (Module A) ===
    max_span_length: int = 8
    span_repr_size: int = 256
    width_embedding_dim: int = 32
    top_k_aspects: int = 20
    top_k_opinions: int = 20
    span_score_threshold: float = 0.0

    # === Pair Module (Module B) ===
    dist_buckets: int = 16
    order_types: int = 3                # asp_first, opn_first, has_null
    top_c_categories: int = 3           # categories per pair for quad expansion

    # === Quad Reranker (Module C) ===
    cat_embedding_dim: int = 32
    aff_embedding_dim: int = 32
    num_meta_features: int = 9
    pair_repr_size: int = 512           # auto-computed: span_repr_size * 2

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
    seeds: list = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    def __post_init__(self):
        self.pair_repr_size = self.span_repr_size * 2
