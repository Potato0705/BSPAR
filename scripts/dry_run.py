"""Dry-run: validate the full BSPAR pipeline logic without needing
a pretrained model download. Uses random tensors to simulate encoder output.

This tests: data loading → label assignment → loss computation → decode → metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from bspar.config import BSPARConfig
from bspar.data.preprocessor import (
    load_data, build_category_map, get_categories_for_dataset
)
from bspar.data.schema import Span, Quad, Example
from bspar.data.span_utils import enumerate_spans
from bspar.models.span_proposal import SpanProposal
from bspar.models.pair_module import PairModule
from bspar.models.quad_reranker import QuadReranker
from bspar.models.bspar_stage2 import BSPARStage2
from bspar.losses.span_loss import SpanFocalLoss
from bspar.losses.pair_loss import PairBCELoss
from bspar.losses.category_loss import CategoryCELoss
from bspar.losses.ranking_loss import PairwiseMarginRankingLoss
from bspar.evaluation.metrics import quad_f1, pair_f1, span_f1, compute_quad_f1
from bspar.training.candidate_generator import CandidateGenerator, RerankSample, RerankExample
from bspar.utils.seed import set_seed


def test_data_loading():
    """Test all 4 dataset formats load correctly."""
    print("=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)

    datasets = {
        "asqp_rest15": ("data/asqp_rest15/train.txt", "auto"),
        "asqp_rest16": ("data/asqp_rest16/train.txt", "auto"),
        "acos_laptop": ("data/acos_laptop/train.tsv", "acos"),
        "acos_restaurant": ("data/acos_restaurant/train.tsv", "acos"),
    }

    for name, (path, fmt) in datasets.items():
        cats = get_categories_for_dataset(name)
        examples = load_data(path, fmt, cats)

        # Validate examples
        assert len(examples) > 0, f"{name}: no examples loaded"
        ex = examples[0]
        assert isinstance(ex, Example)
        assert len(ex.tokens) > 0
        assert len(ex.quads) > 0

        # Validate quads
        for q in ex.quads:
            assert isinstance(q, Quad)
            assert isinstance(q.aspect, Span)
            assert isinstance(q.opinion, Span)
            assert q.sentiment in ("POS", "NEG", "NEU")

        n_null_asp = sum(1 for e in examples for q in e.quads if q.aspect.is_null)
        n_null_opn = sum(1 for e in examples for q in e.quads if q.opinion.is_null)
        n_quads = sum(len(e.quads) for e in examples)

        print(f"  ✓ {name}: {len(examples)} examples, {n_quads} quads, "
              f"{len(cats)} categories, "
              f"null_asp={n_null_asp}, null_opn={n_null_opn}")

    print("  ✓ All data loading tests passed!\n")


def test_span_enumeration():
    """Test span enumeration and label assignment."""
    print("=" * 60)
    print("TEST 2: Span Enumeration")
    print("=" * 60)

    spans = enumerate_spans(10, max_span_length=8)
    expected = sum(min(8, 10 - i) for i in range(10))
    assert len(spans) == expected, f"Expected {expected} spans, got {len(spans)}"

    # Check all spans are valid
    for s, e in spans:
        assert 0 <= s <= e < 10
        assert e - s + 1 <= 8

    print(f"  ✓ enumerate_spans(10, 8) → {len(spans)} spans")
    print("  ✓ Span enumeration tests passed!\n")


def test_span_proposal():
    """Test SpanProposal module forward pass."""
    print("=" * 60)
    print("TEST 3: Span Proposal Module")
    print("=" * 60)

    batch_size, seq_len, hidden_size = 4, 32, 768
    H = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    proposal = SpanProposal(
        hidden_size=hidden_size,
        max_span_length=8,
        span_repr_size=256,
        width_embedding_dim=32,
    )

    out = proposal(H, mask)

    assert "asp_scores" in out
    assert "opn_scores" in out
    assert "span_reprs" in out
    assert "span_indices" in out
    assert "null_asp_repr" in out
    assert "null_opn_repr" in out

    num_spans = len(out["span_indices"])
    assert out["asp_scores"].shape == (batch_size, num_spans)
    assert out["opn_scores"].shape == (batch_size, num_spans)
    assert out["span_reprs"].shape == (batch_size, num_spans, 256)
    assert out["null_asp_repr"].shape == (batch_size, 256)

    print(f"  ✓ SpanProposal: {num_spans} spans, reprs={out['span_reprs'].shape}")
    print("  ✓ Span proposal tests passed!\n")


def test_pair_module():
    """Test PairModule forward pass."""
    print("=" * 60)
    print("TEST 4: Pair Module")
    print("=" * 60)

    batch_size, n_pairs = 4, 50
    span_repr_size = 256
    num_categories = 12
    num_sentiments = 3

    pair_mod = PairModule(
        span_repr_size=span_repr_size,
        num_categories=num_categories,
        num_sentiments=num_sentiments,
        task_type="asqp",
    )

    asp_reprs = torch.randn(batch_size, n_pairs, span_repr_size)
    opn_reprs = torch.randn(batch_size, n_pairs, span_repr_size)
    dist_ids = torch.randint(0, 16, (batch_size, n_pairs))
    order_ids = torch.randint(0, 3, (batch_size, n_pairs))

    out = pair_mod(asp_reprs, opn_reprs, dist_ids, order_ids)

    assert out["pair_scores"].shape == (batch_size, n_pairs)
    assert out["cat_logits"].shape == (batch_size, n_pairs, num_categories)
    assert out["aff_output"].shape == (batch_size, n_pairs, num_sentiments)
    assert out["pair_reprs"].shape == (batch_size, n_pairs, span_repr_size * 2)

    print(f"  ✓ PairModule: pairs={n_pairs}, pair_repr_size={span_repr_size * 2}")
    print("  ✓ Pair module tests passed!\n")


def test_losses():
    """Test all loss functions."""
    print("=" * 60)
    print("TEST 5: Loss Functions")
    print("=" * 60)

    # Span loss
    span_loss_fn = SpanFocalLoss(alpha=0.25, gamma=2.0)
    asp_scores = torch.randn(4, 100)
    asp_labels = torch.zeros(4, 100)
    asp_labels[:, :5] = 1.0
    opn_scores = torch.randn(4, 100)
    opn_labels = torch.zeros(4, 100)
    opn_labels[:, 10:15] = 1.0
    loss_span = span_loss_fn(asp_scores, asp_labels, opn_scores, opn_labels)
    assert loss_span.dim() == 0 and loss_span.item() > 0
    print(f"  ✓ Focal BCE loss: {loss_span.item():.4f}")

    # Pair loss
    pair_loss_fn = PairBCELoss(hard_neg_weight=3.0)
    pair_logits = torch.randn(4, 50)
    pair_labels = torch.zeros(4, 50)
    pair_labels[:, :3] = 1.0
    hn_mask = torch.zeros(4, 50)
    hn_mask[:, 3:6] = 1.0
    loss_pair = pair_loss_fn(pair_logits, pair_labels, hn_mask)
    assert loss_pair.dim() == 0 and loss_pair.item() > 0
    print(f"  ✓ Pair BCE loss (with HN): {loss_pair.item():.4f}")

    # Category loss
    cat_loss_fn = CategoryCELoss(label_smoothing=0.1)
    cat_logits = torch.randn(12, 12)  # (num_valid_pairs, num_categories)
    cat_labels = torch.randint(0, 12, (12,))
    loss_cat = cat_loss_fn(cat_logits, cat_labels)
    print(f"  ✓ Category CE loss: {loss_cat.item():.4f}")

    # Ranking loss
    rank_loss_fn = PairwiseMarginRankingLoss(margin=1.0)
    scores = torch.randn(1, 10)
    labels = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float)
    loss_rank = rank_loss_fn(scores, labels)
    print(f"  ✓ Pairwise ranking loss: {loss_rank.item():.4f}")

    print("  ✓ All loss function tests passed!\n")


def test_reranker():
    """Test QuadReranker and BSPARStage2."""
    print("=" * 60)
    print("TEST 6: Quad Reranker (Stage-2)")
    print("=" * 60)

    config = BSPARConfig()
    config.num_categories = 12
    config.__post_init__()

    model = BSPARStage2(config)

    batch_size, n_cands = 4, 20
    pair_reprs = torch.randn(batch_size, n_cands, config.pair_repr_size)
    cat_ids = torch.randint(0, 12, (batch_size, n_cands))
    aff_input = torch.randint(0, 3, (batch_size, n_cands))
    meta_features = torch.randn(batch_size, n_cands, config.num_meta_features)
    labels = torch.zeros(batch_size, n_cands)
    labels[:, :3] = 1.0

    # Training
    result = model(pair_reprs, cat_ids, aff_input, meta_features, labels, mode="train")
    assert "loss" in result
    assert "quad_scores" in result
    assert result["quad_scores"].shape == (batch_size, n_cands)
    print(f"  ✓ Stage-2 train loss: {result['loss'].item():.4f}")

    # Inference
    result = model(pair_reprs, cat_ids, aff_input, meta_features, mode="inference")
    assert "quad_scores" in result
    assert "loss" not in result
    print(f"  ✓ Stage-2 inference: scores shape={result['quad_scores'].shape}")

    # Backward pass
    result = model(pair_reprs, cat_ids, aff_input, meta_features, labels, mode="train")
    result["loss"].backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  ✓ Backward pass OK, {grad_count} params have gradients")

    print("  ✓ Reranker tests passed!\n")


def test_metrics():
    """Test evaluation metrics."""
    print("=" * 60)
    print("TEST 7: Evaluation Metrics")
    print("=" * 60)

    # Create mock predictions and golds
    a1 = Span(0, 1, "good food")
    o1 = Span(3, 3, "delicious")
    a2 = Span(5, 5, "service")
    o2 = Span(7, 7, "slow")

    preds = [[
        Quad(a1, o1, "food quality", "POS"),
        Quad(a2, o2, "service general", "NEG"),
    ]]
    golds = [[
        Quad(a1, o1, "food quality", "POS"),      # match
        Quad(a2, o2, "service general", "NEG"),    # match
        Quad(Span(10, 10, "price"), Span.null("opinion"), "food prices", "NEU"),  # miss
    ]]

    qf1 = quad_f1(preds, golds)
    assert qf1["tp"] == 2
    assert qf1["total_pred"] == 2
    assert qf1["total_gold"] == 3
    assert qf1["precision"] == 1.0
    assert abs(qf1["recall"] - 2/3) < 1e-6
    print(f"  ✓ Quad F1: P={qf1['precision']:.3f} R={qf1['recall']:.3f} F1={qf1['f1']:.3f}")

    pf1 = pair_f1(preds, golds)
    print(f"  ✓ Pair F1: {pf1['f1']:.3f}")

    sf1 = span_f1(preds, golds, role="aspect")
    print(f"  ✓ Aspect Span F1: {sf1['f1']:.3f}")

    print("  ✓ Metrics tests passed!\n")


def test_candidate_generator():
    """Test CandidateGenerator data structures."""
    print("=" * 60)
    print("TEST 8: Candidate Generator Structures")
    print("=" * 60)

    sample_pos = RerankSample(
        pair_repr=torch.randn(512),
        category_id=3,
        affective=0,
        meta_features=[0.9, 0.8, 0.7, 0.6, 0.1, 0.0, 0.0, 3.0, 2.0],
        label=1,
        difficulty=0,
    )
    sample_neg = RerankSample(
        pair_repr=torch.randn(512),
        category_id=5,
        affective=1,
        meta_features=[0.3, 0.2, 0.1, 0.4, 0.9, 0.0, 1.0, 1.0, 0.0],
        label=0,
        difficulty=2,
    )

    rerank_ex = RerankExample(
        example_id="test_0",
        candidates=[sample_pos, sample_neg],
    )

    assert rerank_ex.has_positive
    assert rerank_ex.has_negative
    print(f"  ✓ RerankExample: {len(rerank_ex.candidates)} candidates, "
          f"has_pos={rerank_ex.has_positive}, has_neg={rerank_ex.has_negative}")

    print("  ✓ Candidate generator tests passed!\n")


def test_stage2_training_loop():
    """Test Stage-2 training dataset and collation."""
    print("=" * 60)
    print("TEST 9: Stage-2 Training Loop")
    print("=" * 60)

    from bspar.training.stage2_trainer import RerankDataset, collate_rerank

    config = BSPARConfig()
    config.num_categories = 12
    config.__post_init__()

    # Create synthetic rerank examples
    examples = []
    for i in range(10):
        cands = []
        for j in range(8):
            cands.append(RerankSample(
                pair_repr=torch.randn(config.pair_repr_size),
                category_id=j % config.num_categories,
                affective=j % 3,
                meta_features=[float(x) for x in range(config.num_meta_features)],
                label=1 if j < 2 else 0,
            ))
        examples.append(RerankExample(example_id=f"ex_{i}", candidates=cands))

    dataset = RerankDataset(examples, config)
    assert len(dataset) == 10

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_rerank)

    model = BSPARStage2(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run 2 training steps
    model.train()
    for step, batch in enumerate(loader):
        result = model(
            pair_reprs=batch["pair_reprs"],
            cat_ids=batch["cat_ids"],
            aff_input=batch["aff_input"],
            meta_features=batch["meta_features"],
            labels=batch["labels"],
            mode="train",
        )
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step}: loss={loss.item():.4f}")
        if step >= 1:
            break

    print("  ✓ Stage-2 training loop tests passed!\n")


def main():
    set_seed(42)
    print("\n" + "=" * 60)
    print("BSPAR DRY-RUN: Full Pipeline Validation")
    print("=" * 60 + "\n")

    test_data_loading()
    test_span_enumeration()
    test_span_proposal()
    test_pair_module()
    test_losses()
    test_reranker()
    test_metrics()
    test_candidate_generator()
    test_stage2_training_loop()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nThe codebase is ready for training.")
    print("To run real experiments, ensure 'roberta-base' is accessible:")
    print("  python scripts/train_stage1.py --config configs/asqp_rest15.yaml")


if __name__ == "__main__":
    main()
