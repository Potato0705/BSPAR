"""Stage-1 training loop: Encoder + SpanProposal + PairModule joint training."""

import os
import time
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

            print(f"Epoch {epoch}/{self.config.stage1_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Dev Quad-F1: {dev_f1:.4f} | "
                  f"Span-F1: {dev_metrics.get('span_f1', 0.0):.4f} | "
                  f"GoldInj: {gold_inj_prob:.2f}")

            # Early stopping
            if dev_f1 > self.best_f1:
                self.best_f1 = dev_f1
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
        print(f"Stage-1 training complete. Best dev F1: {self.best_f1:.4f}")
        return self.best_f1

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

        for batch in self.dev_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_to_subword=batch["word_to_subword"],
                mode="inference",
            )

            # Decode candidates into predictions
            for b_idx, cands in enumerate(outputs["candidates"]):
                # Simple greedy decode: take top candidate per unique (asp, opn)
                preds = self._greedy_decode(cands)
                all_preds.append(preds)
                all_golds.append(batch["gold_quads"][b_idx])

        return compute_quad_f1(all_preds, all_golds, self.id_to_cat,
                               self.cat_to_id)

    def _greedy_decode(self, candidates):
        """Simple greedy decoding from candidates.

        Sort by pair_score first, then cat_prob, take top predictions,
        apply simple dedup.
        """
        if not candidates:
            return []

        pair_thr = getattr(self.config, "stage1_pair_score_threshold", 0.01)
        scored = sorted(
            candidates,
            key=lambda c: (c["pair_score"], c["cat_prob"]),
            reverse=True,
        )

        selected = []
        seen = set()
        for c in scored:
            # Sorted by pair_score desc, so we can stop once below threshold.
            if c["pair_score"] < pair_thr:
                break
            key = (c["asp_span"], c["opn_span"], c["category_id"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(c)
            if len(selected) >= 10:  # max predictions per example
                break

        return selected

    def _save_checkpoint(self, output_dir, filename):
        """Save model checkpoint."""
        path = os.path.join(output_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_f1": self.best_f1,
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_f1 = ckpt.get("best_f1", 0.0)
