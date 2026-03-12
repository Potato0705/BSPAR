"""Stage-1 training loop: Encoder + SpanProposal + PairModule joint training."""

import os
import time
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from ..models.bspar_stage1 import BSPARStage1
from ..data.dataset import BSPARStage1Dataset, collate_stage1
from ..evaluation.metrics import compute_quad_f1


class Stage1Trainer:
    """Handles the complete Stage-1 training pipeline."""

    def __init__(self, config, train_examples, dev_examples,
                 cat_to_id, id_to_cat):
        self.config = config
        self.cat_to_id = cat_to_id
        self.id_to_cat = id_to_cat

        # Build datasets
        print("Building Stage-1 datasets...")
        self.train_dataset = BSPARStage1Dataset(
            train_examples, config.model_name,
            max_length=128, max_span_length=config.max_span_length,
        )
        self.dev_dataset = BSPARStage1Dataset(
            dev_examples, config.model_name,
            max_length=128, max_span_length=config.max_span_length,
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size,
            shuffle=True, collate_fn=collate_stage1,
        )
        self.dev_loader = DataLoader(
            self.dev_dataset, batch_size=config.batch_size,
            shuffle=False, collate_fn=collate_stage1,
        )

        # Build model
        print("Building Stage-1 model...")
        self.model = BSPARStage1(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer with differential LR
        encoder_params = list(self.model.encoder.parameters())
        head_params = (
            list(self.model.span_proposal.parameters()) +
            list(self.model.pair_module.parameters())
        )
        self.optimizer = AdamW([
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": head_params, "lr": config.head_lr},
        ], weight_decay=config.weight_decay)

        # Scheduler with warmup
        total_steps = len(self.train_loader) * config.stage1_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay_scheduler = LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.0,
            total_iters=total_steps - warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )

        # Tracking
        self.best_f1 = 0.0
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.patience_counter = 0

    def train(self, output_dir: str):
        """Run full Stage-1 training with early stopping."""
        os.makedirs(output_dir, exist_ok=True)

        for epoch in range(1, self.config.stage1_epochs + 1):
            # Compute scheduled gold injection probability
            gold_inj_prob = self._get_gold_injection_prob(epoch)

            # Train
            train_loss = self._train_epoch(epoch, gold_inj_prob)

            # Evaluate
            dev_metrics = self._evaluate()
            dev_f1 = dev_metrics.get("quad_f1", 0.0)
            ckpt_score = self._checkpoint_selection_score(dev_metrics)

            print(f"Epoch {epoch}/{self.config.stage1_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Dev Quad-F1: {dev_f1:.4f} | "
                  f"Span-F1: {dev_metrics.get('span_f1', 0.0):.4f} | "
                  f"PairSpaceR: {dev_metrics.get('gold_pair_recall_pair_space', 0.0):.4f} | "
                  f"PosRetainR: {dev_metrics.get('sample_has_positive_after_retention_ratio', 0.0):.4f} | "
                  f"CkptScore: {ckpt_score:.4f} | "
                  f"GoldInj: {gold_inj_prob:.2f}")

            # Early stopping
            if ckpt_score > self.best_score:
                self.best_f1 = dev_f1
                self.best_score = ckpt_score
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(output_dir, "best_stage1.pt")
                print(f"  -> New best! Saved checkpoint.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"  -> Early stopping at epoch {epoch}")
                    break

        # Save final
        self._save_checkpoint(output_dir, "final_stage1.pt")
        print(
            f"Stage-1 training complete. Best dev F1: {self.best_f1:.4f} | "
            f"Best ckpt score: {self.best_score:.4f} | "
            f"Best epoch: {self.best_epoch}"
        )
        return self.best_f1

    def _checkpoint_selection_score(self, metrics):
        """Compute scalar score for selecting best Stage-1 checkpoint."""
        metric = getattr(self.config, "stage1_ckpt_metric", "composite")
        quad = metrics.get("quad_f1", 0.0)
        pair_recall = metrics.get("gold_pair_recall_after_gate", 0.0)
        pos_ratio = metrics.get("sample_has_positive_after_retention_ratio", 0.0)

        if metric == "quad_f1":
            return quad
        if metric == "pair_recall_after_retention":
            return pair_recall
        if metric == "pos_candidate_ratio":
            return pos_ratio

        # Default: readiness composite under fixed retention baseline.
        wq = getattr(self.config, "stage1_ckpt_quad_weight", 0.6)
        wr = getattr(self.config, "stage1_ckpt_pair_recall_weight", 0.25)
        wp = getattr(self.config, "stage1_ckpt_pos_ratio_weight", 0.15)
        return (wq * quad) + (wr * pair_recall) + (wp * pos_ratio)

    def _get_gold_injection_prob(self, epoch):
        """Compute gold injection probability for scheduled teacher forcing.

        Returns 1.0 during warmup epochs, then linearly decays to end value.
        """
        cfg = self.config
        start = getattr(cfg, 'gold_injection_start', 1.0)
        end = getattr(cfg, 'gold_injection_end', 0.0)
        warmup = getattr(cfg, 'gold_injection_warmup', 2)

        if epoch <= warmup:
            return start

        total_decay_epochs = cfg.stage1_epochs - warmup
        if total_decay_epochs <= 0:
            return start

        progress = (epoch - warmup) / total_decay_epochs
        return start + (end - start) * min(progress, 1.0)

    def _train_epoch(self, epoch, gold_injection_prob=1.0):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            gold_quads = batch["gold_quads"]
            word_to_subword = batch["word_to_subword"]

            losses = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gold_quads=gold_quads,
                cat_to_id=self.cat_to_id,
                word_to_subword=word_to_subword,
                mode="train",
                gold_injection_prob=gold_injection_prob,
            )

            loss = losses["loss_total"]

            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                avg = total_loss / num_batches
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | "
                      f"Loss: {avg:.4f} | "
                      f"span: {losses['loss_span']:.3f} "
                      f"pair: {losses['loss_pair']:.3f} "
                      f"cat: {losses['loss_cat']:.3f} "
                      f"aff: {losses['loss_aff']:.3f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self):
        """Evaluate on dev set - collect predictions and compute metrics."""
        self.model.eval()
        all_preds = []
        all_golds = []
        pair_thr = getattr(self.config, "stage1_pair_score_threshold", 0.01)
        pair_strategy = getattr(self.config, "stage1_pair_retention_strategy", "topn_only")
        pair_top_n = getattr(self.config, "stage1_pair_top_n", 20)
        pair_flow = self._init_pair_flow_stats(pair_thr, pair_strategy, pair_top_n)

        for batch in self.dev_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_to_subword=batch["word_to_subword"],
                mode="inference",
                pair_score_threshold=pair_thr,
                pair_retention_strategy=pair_strategy,
                pair_top_n=pair_top_n,
            )

            # Decode candidates into predictions
            for b_idx, cands in enumerate(outputs["candidates"]):
                # Simple greedy decode: take top candidate per unique (asp, opn)
                preds = self._greedy_decode(cands)
                all_preds.append(preds)
                all_golds.append(batch["gold_quads"][b_idx])
                self._update_pair_flow_stats(pair_flow, outputs, batch, b_idx)

        metrics = compute_quad_f1(all_preds, all_golds, self.id_to_cat,
                                  self.cat_to_id)
        pair_flow_metrics = self._finalize_pair_flow_stats(pair_flow)
        metrics.update(pair_flow_metrics)

        print(
            "  [PairFlow] "
            f"pair-space recall: {pair_flow_metrics['gold_pair_recall_pair_space']:.4f} | "
            f"after-retention recall: {pair_flow_metrics['gold_pair_recall_after_gate']:.4f} | "
            f"sample +retain ratio: {pair_flow_metrics['sample_has_positive_after_retention_ratio']:.4f} | "
            f"avg +pairs into Stage-2: {pair_flow_metrics['avg_positive_pairs_into_stage2']:.2f} | "
            f"avg pairs into Stage-2: {pair_flow_metrics['avg_pairs_into_stage2']:.2f} | "
            f"avg cands into Stage-2: {pair_flow_metrics['avg_candidates_into_stage2']:.2f} | "
            f"pos q50/q90: {pair_flow_metrics['pos_pair_score_q50']:.4f}/{pair_flow_metrics['pos_pair_score_q90']:.4f} | "
            f"neg q50/q90: {pair_flow_metrics['neg_pair_score_q50']:.4f}/{pair_flow_metrics['neg_pair_score_q90']:.4f}"
        )

        return metrics

    def _greedy_decode(self, candidates):
        """Simple greedy decoding from candidates.

        Sort by pair_score first, then cat_prob, take top predictions,
        apply simple dedup.
        """
        if not candidates:
            return []

        decode_pair_thr = getattr(self.config, "stage1_decode_pair_score_threshold", None)
        scored = sorted(
            candidates,
            key=lambda c: (c["pair_score"], c["cat_prob"]),
            reverse=True,
        )

        selected = []
        seen = set()
        for c in scored:
            if decode_pair_thr is not None and c["pair_score"] < decode_pair_thr:
                break
            key = (c["asp_span"], c["opn_span"], c["category_id"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(c)
            if len(selected) >= 10:  # max predictions per example
                break

        return selected

    @staticmethod
    def _score_quantile(values, q):
        """Compute q-quantile for a Python list without numpy dependency."""
        if not values:
            return 0.0
        vals = sorted(values)
        idx = int(math.floor((len(vals) - 1) * q))
        idx = max(0, min(len(vals) - 1, idx))
        return float(vals[idx])

    def _init_pair_flow_stats(self, pair_thr, pair_strategy, pair_top_n):
        return {
            "pair_thr": pair_thr,
            "pair_strategy": pair_strategy,
            "pair_top_n": pair_top_n,
            "num_examples": 0,
            "gold_pair_total": 0,
            "gold_pair_hit_space": 0,
            "gold_pair_hit_gate": 0,
            "positive_pairs_into_stage2": 0,
            "pairs_into_stage2": 0,
            "candidates_into_stage2": 0,
            "sample_with_gold_pair": 0,
            "sample_has_positive_after_retention": 0,
            "pos_scores": [],
            "neg_scores": [],
        }

    def _gold_pairs_subword(self, gold_quads, w2s):
        """Build gold pair set in subword coordinates for one example."""
        gold_pairs = set()
        for q in gold_quads:
            if q.aspect.is_null:
                a_span = (-1, -1)
            else:
                a_span = self.model._word_span_to_subword(
                    q.aspect.start, q.aspect.end, w2s
                )
                if a_span is None:
                    continue

            if q.opinion.is_null:
                o_span = (-1, -1)
            else:
                o_span = self.model._word_span_to_subword(
                    q.opinion.start, q.opinion.end, w2s
                )
                if o_span is None:
                    continue

            gold_pairs.add((a_span, o_span))
        return gold_pairs

    def _update_pair_flow_stats(self, stats, outputs, batch, b_idx):
        """Accumulate pair-space and pair-gate diagnostics for one example."""
        pair_map = outputs.get("pair_map", [])
        asp_indices_batch = outputs.get("asp_indices", [])
        opn_indices_batch = outputs.get("opn_indices", [])
        if b_idx >= len(asp_indices_batch) or b_idx >= len(opn_indices_batch):
            return

        gold_quads = batch["gold_quads"][b_idx]
        w2s = batch["word_to_subword"][b_idx]
        gold_pairs = self._gold_pairs_subword(gold_quads, w2s)

        pair_scores = torch.sigmoid(outputs["pair_scores"][b_idx]).detach().cpu().tolist()
        selected_pair_ids = outputs.get("selected_pair_ids", [])
        if b_idx < len(selected_pair_ids):
            selected_pair_ids = selected_pair_ids[b_idx]
        else:
            selected_pair_ids = []
        selected_pair_ids = set(selected_pair_ids)

        pair_space = set()
        after_gate = set()

        for p, (ai, oi) in enumerate(pair_map):
            pair_key = (asp_indices_batch[b_idx][ai], opn_indices_batch[b_idx][oi])
            pair_space.add(pair_key)
            score = pair_scores[p]
            if p in selected_pair_ids:
                after_gate.add(pair_key)

            if pair_key in gold_pairs:
                stats["pos_scores"].append(score)
            else:
                stats["neg_scores"].append(score)

        stats["num_examples"] += 1
        stats["pairs_into_stage2"] += len(after_gate)
        stats["positive_pairs_into_stage2"] += len(gold_pairs & after_gate)
        stats["candidates_into_stage2"] += len(outputs["candidates"][b_idx])
        stats["gold_pair_total"] += len(gold_pairs)
        stats["gold_pair_hit_space"] += len(gold_pairs & pair_space)
        stats["gold_pair_hit_gate"] += len(gold_pairs & after_gate)
        if len(gold_pairs) > 0:
            stats["sample_with_gold_pair"] += 1
            if len(gold_pairs & after_gate) > 0:
                stats["sample_has_positive_after_retention"] += 1

    def _finalize_pair_flow_stats(self, stats):
        gold_total = max(stats["gold_pair_total"], 1)
        num_examples = max(stats["num_examples"], 1)
        sample_with_gold = max(stats["sample_with_gold_pair"], 1)
        return {
            "gold_pair_recall_pair_space": stats["gold_pair_hit_space"] / gold_total,
            "gold_pair_recall_after_gate": stats["gold_pair_hit_gate"] / gold_total,
            "avg_positive_pairs_into_stage2": (
                stats["positive_pairs_into_stage2"] / num_examples
            ),
            "avg_pairs_into_stage2": (
                stats["pairs_into_stage2"] / num_examples
            ),
            "avg_candidates_into_stage2": (
                stats["candidates_into_stage2"] / num_examples
            ),
            "sample_has_positive_after_retention_ratio": (
                stats["sample_has_positive_after_retention"] / sample_with_gold
            ),
            "pos_pair_score_q50": self._score_quantile(stats["pos_scores"], 0.5),
            "pos_pair_score_q90": self._score_quantile(stats["pos_scores"], 0.9),
            "neg_pair_score_q50": self._score_quantile(stats["neg_scores"], 0.5),
            "neg_pair_score_q90": self._score_quantile(stats["neg_scores"], 0.9),
        }

    def _save_checkpoint(self, output_dir, filename):
        """Save model checkpoint."""
        path = os.path.join(output_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_f1": self.best_f1,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "stage1_ckpt_metric": getattr(self.config, "stage1_ckpt_metric", "composite"),
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_f1 = ckpt.get("best_f1", 0.0)
        self.best_score = ckpt.get("best_score", float("-inf"))
        self.best_epoch = ckpt.get("best_epoch", 0)
