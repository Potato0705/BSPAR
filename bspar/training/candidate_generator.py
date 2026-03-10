"""Bridge between Stage-1 and Stage-2: generate real candidates.

Runs a trained Stage-1 model in inference mode on train/dev data to produce
realistic candidate quads. These candidates (with their noise patterns)
become the training data for the Stage-2 reranker.

This is the engineering realization of Contribution C3.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class RerankSample:
    """A single candidate quad with features and label for reranker training."""
    pair_repr: torch.Tensor         # (pair_repr_size,)
    category_id: int
    affective: object               # int (ASQP) or tuple (dimABSA)
    meta_features: list[float]
    label: int                      # 1=gold match, 0=negative
    difficulty: int = 0             # 0=easy, 1=partial, 2=hard negative


@dataclass
class RerankExample:
    """All candidate quads for one input example."""
    example_id: str
    candidates: list[RerankSample] = field(default_factory=list)

    @property
    def has_positive(self):
        return any(c.label == 1 for c in self.candidates)

    @property
    def has_negative(self):
        return any(c.label == 0 for c in self.candidates)


class CandidateGenerator:
    """Generate real candidate quads from a trained Stage-1 model."""

    def __init__(self, stage1_model, config, category_map):
        self.model = stage1_model
        self.config = config
        self.category_map = category_map

    @torch.no_grad()
    def generate(self, dataloader, gold_examples):
        """Run Stage-1 inference and label candidates against gold.

        Args:
            dataloader: DataLoader yielding tokenized batches
            gold_examples: list of Example with gold quads

        Returns:
            list of RerankExample
        """
        self.model.eval()
        all_rerank = []
        example_idx = 0

        for batch in dataloader:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                mode="inference",
            )

            batch_size = batch["input_ids"].size(0)
            for b in range(batch_size):
                if example_idx >= len(gold_examples):
                    break

                gold = gold_examples[example_idx]
                rerank_ex = RerankExample(example_id=gold.id)

                # Extract candidates for this example from batch outputs
                candidates = self._extract_batch_candidates(outputs, b)

                for cand in candidates:
                    label = self._match_gold(cand, gold.quads)
                    difficulty = self._assess_difficulty(cand, gold.quads)

                    sample = RerankSample(
                        pair_repr=cand["pair_repr"].cpu(),
                        category_id=cand["category_id"],
                        affective=cand["affective"],
                        meta_features=cand["meta_features"],
                        label=label,
                        difficulty=difficulty,
                    )
                    rerank_ex.candidates.append(sample)

                # Only keep examples with at least one pos and one neg
                if rerank_ex.has_positive and rerank_ex.has_negative:
                    all_rerank.append(rerank_ex)

                example_idx += 1

        return all_rerank

    def _extract_batch_candidates(self, outputs, batch_idx):
        """Extract candidate quads for a single example from batch output.

        The Stage-1 model's _build_candidates() returns:
            outputs["candidates"]: list[list[dict]] — candidates per example
        Each candidate dict has keys: pair_repr, pair_score, asp_span, opn_span,
            category_id, affective, meta_features, etc.
        """
        candidates = outputs.get("candidates", [])
        if batch_idx >= len(candidates):
            return []
        return candidates[batch_idx]

    def _match_gold(self, candidate, gold_quads):
        """Exact match: aspect span + opinion span + category + sentiment."""
        for gq in gold_quads:
            asp_match = (
                (candidate["asp_span"] == (-1, -1) and gq.aspect.is_null) or
                (candidate["asp_span"] == (gq.aspect.start, gq.aspect.end))
            )
            opn_match = (
                (candidate["opn_span"] == (-1, -1) and gq.opinion.is_null) or
                (candidate["opn_span"] == (gq.opinion.start, gq.opinion.end))
            )
            cat_match = (
                self.category_map.get(candidate["category_id"]) == gq.category
            )
            aff_match = self._affective_match(candidate["affective"], gq)

            if asp_match and opn_match and cat_match and aff_match:
                return 1
        return 0

    def _affective_match(self, pred_aff, gold_quad):
        """Check affective match (exact for ASQP, threshold for dimABSA)."""
        if self.config.task_type == "asqp":
            sentiment_map = {0: "POS", 1: "NEG", 2: "NEU"}
            return sentiment_map.get(pred_aff) == gold_quad.sentiment
        else:
            # For dimABSA, exact match is not meaningful for real values
            # Use threshold-based matching for labeling
            v_pred, ar_pred = pred_aff
            v_diff = abs(v_pred - (gold_quad.valence or 0))
            ar_diff = abs(ar_pred - (gold_quad.arousal or 0))
            return v_diff < 0.5 and ar_diff < 0.5

    def _assess_difficulty(self, candidate, gold_quads):
        """Categorize negative difficulty: 0=easy, 1=partial, 2=hard."""
        max_overlap = 0
        for gq in gold_quads:
            overlap = 0
            asp_match = (
                (candidate["asp_span"] == (-1, -1) and gq.aspect.is_null) or
                (candidate["asp_span"] == (gq.aspect.start, gq.aspect.end))
            )
            opn_match = (
                (candidate["opn_span"] == (-1, -1) and gq.opinion.is_null) or
                (candidate["opn_span"] == (gq.opinion.start, gq.opinion.end))
            )
            cat_match = (
                self.category_map.get(candidate["category_id"]) == gq.category
            )
            if asp_match:
                overlap += 1
            if opn_match:
                overlap += 1
            if cat_match:
                overlap += 1
            max_overlap = max(max_overlap, overlap)

        if max_overlap >= 3:
            return 2  # hard: 3 elements match but affective wrong
        elif max_overlap >= 1:
            return 1  # partial
        return 0      # easy
