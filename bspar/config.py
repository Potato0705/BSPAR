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
    pair_pos_weight: float = 8.0
    pair_easy_neg_weight: float = 1.0
    pair_span_nearmiss_weight: float = 2.0
    pair_cat_confused_weight: float = 3.0
    pair_focal_gamma: float = 1.0
    lambda_pair_rank: float = 0.0
    pair_rank_margin: float = 0.1

    # === Training ===
    encoder_lr: float = 2e-5
    head_lr: float = 5e-5
    reranker_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    stage1_epochs: int = 20
    stage2_epochs: int = 10
    batch_size: int = 16
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    patience: int = 8                   # early stopping patience
    stage2_use_group_loss: bool = False
    stage2_group_loss_lambda: float = 0.3
    stage2_group_tau: float = 1.0
    stage2_use_pair_prior: bool = False
    stage2_pair_prior_alpha: float = 0.3
    stage2_pair_prior_lambda: float = 0.3
    stage2_pair_prior_pos_weight: float = 1.0
    stage1_ckpt_metric: str = "composite"
    stage1_ckpt_quad_weight: float = 0.6
    stage1_ckpt_pair_recall_weight: float = 0.25
    stage1_ckpt_pos_ratio_weight: float = 0.15

    # === Teacher Forcing ===
    gold_injection_start: float = 1.0   # gold span injection prob at epoch 1
    gold_injection_end: float = 0.0     # gold span injection prob at final epoch
    gold_injection_warmup: int = 2      # keep full injection for N epochs before decay

    # === Decode ===
    quad_score_threshold: float = 0.0
    stage1_pair_retention_strategy: str = "topn_only"  # topn_only | pair_gate_only | pair_gate_topn
    stage1_pair_top_n: int = 20
    stage1_pair_score_threshold: float = 0.01
    stage1_decode_pair_score_threshold: float | None = None
    nms_overlap_suppress: bool = True

    # === Reproducibility ===
    seeds: list = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    def __post_init__(self):
        self.pair_repr_size = self.span_repr_size * 2
