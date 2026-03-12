"""Stage-2 training loop: Quad-Aware Reranker on real candidates."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from ..models.bspar_stage2 import BSPARStage2


def roc_auc_binary(scores, labels):
    """Binary ROC-AUC without sklearn dependency."""
    pos = [(s, y) for s, y in zip(scores, labels) if y == 1]
    neg = [(s, y) for s, y in zip(scores, labels) if y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum_pos = 0.0
    for rank, (_, y) in enumerate(sorted_pairs, start=1):
        if y == 1:
            rank_sum_pos += rank

    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc_binary(scores, labels):
    """Average precision (PR-AUC approximation) without sklearn dependency."""
    n_pos = int(sum(1 for y in labels if y == 1))
    if n_pos == 0:
        return 0.0

    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0

    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return float(ap)


class RerankDataset(Dataset):
    """Dataset of real candidate quads from Stage-1 for reranker training.

    Each item is one example with multiple candidate quads,
    each labeled as positive (gold match) or negative.
    """

    def __init__(self, rerank_examples, config):
        self.examples = rerank_examples
        self.config = config
        self.max_cands = 64  # max candidates per example

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        candidates = ex.candidates[:self.max_cands]

        pair_reprs = torch.stack([c.pair_repr for c in candidates])
        cat_ids = torch.tensor([c.category_id for c in candidates], dtype=torch.long)
        labels = torch.tensor([c.label for c in candidates], dtype=torch.float)
        meta = torch.tensor([c.meta_features for c in candidates], dtype=torch.float)
        asp_spans = torch.tensor(
            [list(getattr(c, "asp_span", (-1, -1))) for c in candidates],
            dtype=torch.long,
        )
        opn_spans = torch.tensor(
            [list(getattr(c, "opn_span", (-1, -1))) for c in candidates],
            dtype=torch.long,
        )

        # Affective input
        if self.config.task_type == "asqp":
            aff_input = torch.tensor(
                [c.affective if isinstance(c.affective, int) else 0
                 for c in candidates],
                dtype=torch.long
            )
        else:
            aff_input = torch.tensor(
                [list(c.affective) if isinstance(c.affective, (tuple, list))
                 else [0.0, 0.0] for c in candidates],
                dtype=torch.float
            )

        return {
            "pair_reprs": pair_reprs,
            "cat_ids": cat_ids,
            "aff_input": aff_input,
            "meta_features": meta,
            "labels": labels,
            "asp_spans": asp_spans,
            "opn_spans": opn_spans,
            "num_cands": len(candidates),
        }


def collate_rerank(batch):
    """Collate rerank examples with padding to max num_cands in batch."""
    max_cands = max(item["num_cands"] for item in batch)
    batch_size = len(batch)

    pair_dim = batch[0]["pair_reprs"].size(-1)
    meta_dim = batch[0]["meta_features"].size(-1)

    pair_reprs = torch.zeros(batch_size, max_cands, pair_dim)
    cat_ids = torch.zeros(batch_size, max_cands, dtype=torch.long)
    labels = torch.full((batch_size, max_cands), -1, dtype=torch.float)
    meta_features = torch.zeros(batch_size, max_cands, meta_dim)
    asp_spans = torch.full((batch_size, max_cands, 2), -1, dtype=torch.long)
    opn_spans = torch.full((batch_size, max_cands, 2), -1, dtype=torch.long)
    cand_mask = torch.zeros(batch_size, max_cands, dtype=torch.bool)

    if batch[0]["aff_input"].dim() == 1:
        aff_input = torch.zeros(batch_size, max_cands, dtype=torch.long)
    else:
        aff_input = torch.zeros(batch_size, max_cands, 2)

    for i, item in enumerate(batch):
        n = item["num_cands"]
        pair_reprs[i, :n] = item["pair_reprs"]
        cat_ids[i, :n] = item["cat_ids"]
        aff_input[i, :n] = item["aff_input"]
        meta_features[i, :n] = item["meta_features"]
        labels[i, :n] = item["labels"]
        asp_spans[i, :n] = item["asp_spans"]
        opn_spans[i, :n] = item["opn_spans"]
        cand_mask[i, :n] = True

    return {
        "pair_reprs": pair_reprs,
        "cat_ids": cat_ids,
        "aff_input": aff_input,
        "meta_features": meta_features,
        "labels": labels,
        "asp_spans": asp_spans,
        "opn_spans": opn_spans,
        "cand_mask": cand_mask,
    }


class Stage2Trainer:
    """Handles Stage-2 reranker training on real candidates."""

    def __init__(self, config, train_rerank_examples, dev_rerank_examples):
        self.config = config

        # Datasets
        self.train_dataset = RerankDataset(train_rerank_examples, config)
        self.dev_dataset = RerankDataset(dev_rerank_examples, config)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size,
            shuffle=True, collate_fn=collate_rerank,
        )
        self.dev_loader = DataLoader(
            self.dev_dataset, batch_size=config.batch_size,
            shuffle=False, collate_fn=collate_rerank,
        )

        # Model
        self.model = BSPARStage2(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.reranker_lr,
            weight_decay=config.weight_decay,
        )

        total_steps = len(self.train_loader) * config.stage2_epochs
        self.scheduler = LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1,
            total_iters=total_steps,
        )

        self.best_loss = float("inf")
        self.patience_counter = 0

    def train(self, output_dir: str):
        """Run Stage-2 reranker training."""
        os.makedirs(output_dir, exist_ok=True)

        for epoch in range(1, self.config.stage2_epochs + 1):
            train_stats = self._train_epoch(epoch)
            dev_stats = self._evaluate()

            print(
                f"Epoch {epoch}/{self.config.stage2_epochs} | "
                f"Train Loss: {train_stats['loss']:.4f} "
                f"(rank: {train_stats['loss_rank']:.4f}, "
                f"group: {train_stats['loss_group']:.4f}, "
                f"pair: {train_stats['loss_pair_prior']:.4f}) | "
                f"Dev Loss: {dev_stats['loss']:.4f} "
                f"(rank: {dev_stats['loss_rank']:.4f}, "
                f"group: {dev_stats['loss_group']:.4f}, "
                f"pair: {dev_stats['loss_pair_prior']:.4f}) | "
                f"PairAUC: {dev_stats['pair_prior_auc']:.4f} | "
                f"PairPRAUC: {dev_stats['pair_prior_pr_auc']:.4f} | "
                f"GroupAcc: {dev_stats['group_accuracy']:.4f} | "
                f"ConflictAcc: {dev_stats['conflict_group_accuracy']:.4f} | "
                f"MRR: {dev_stats['group_mrr']:.4f} | "
                f"Hits@1: {dev_stats['group_hits1']:.4f}"
            )

            if dev_stats["loss"] < self.best_loss:
                self.best_loss = dev_stats["loss"]
                self.patience_counter = 0
                self._save_checkpoint(output_dir, "best_stage2.pt")
                print("  -> New best! Saved.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"  -> Early stopping at epoch {epoch}")
                    break

        self._save_checkpoint(output_dir, "final_stage2.pt")
        print(f"Stage-2 training complete. Best dev loss: {self.best_loss:.4f}")

    def _train_epoch(self, epoch):
        del epoch
        self.model.train()
        stats = self._init_stats()

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            result = self.model(
                pair_reprs=batch["pair_reprs"],
                cat_ids=batch["cat_ids"],
                aff_input=batch["aff_input"],
                meta_features=batch["meta_features"],
                labels=batch["labels"],
                asp_spans=batch["asp_spans"],
                opn_spans=batch["opn_spans"],
                mode="train",
            )

            loss = result["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self._update_stats(stats, result)

        return self._finalize_stats(stats)

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        stats = self._init_stats()

        for batch in self.dev_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            result = self.model(
                pair_reprs=batch["pair_reprs"],
                cat_ids=batch["cat_ids"],
                aff_input=batch["aff_input"],
                meta_features=batch["meta_features"],
                labels=batch["labels"],
                asp_spans=batch["asp_spans"],
                opn_spans=batch["opn_spans"],
                mode="train",  # need labels for loss computation
            )

            self._update_stats(stats, result)

        return self._finalize_stats(stats)

    @staticmethod
    def _init_stats():
        return {
            "num_batches": 0,
            "loss_sum": 0.0,
            "loss_rank_sum": 0.0,
            "loss_group_sum": 0.0,
            "loss_pair_prior_sum": 0.0,
            "group_count_all": 0,
            "group_count_conflict": 0,
            "group_acc_correct_all": 0.0,
            "group_acc_correct_conflict": 0.0,
            "group_hits1_conflict": 0.0,
            "group_rr_sum_conflict": 0.0,
            "pair_prior_scores": [],
            "pair_prior_labels": [],
        }

    @staticmethod
    def _update_stats(stats, result):
        stats["num_batches"] += 1
        stats["loss_sum"] += float(result["loss"].item())
        stats["loss_rank_sum"] += float(result.get("loss_rank", 0.0).item())
        stats["loss_group_sum"] += float(result.get("loss_group", 0.0).item())
        pair_loss_val = result.get("loss_pair_prior", 0.0)
        if hasattr(pair_loss_val, "item"):
            pair_loss_val = float(pair_loss_val.item())
        else:
            pair_loss_val = float(pair_loss_val)
        stats["loss_pair_prior_sum"] += pair_loss_val

        stats["group_count_all"] += int(result.get("group_count_all", 0))
        stats["group_count_conflict"] += int(result.get("group_count_conflict", 0))
        stats["group_acc_correct_all"] += float(result.get("group_acc_correct_all_raw", 0.0))
        stats["group_acc_correct_conflict"] += float(
            result.get("group_acc_correct_conflict_raw", 0.0)
        )
        stats["group_hits1_conflict"] += float(result.get("group_hits1_conflict_raw", 0.0))
        stats["group_rr_sum_conflict"] += float(result.get("group_rr_sum_conflict_raw", 0.0))

        pair_logits = result.get("pair_prior_group_logits", None)
        pair_labels = result.get("pair_prior_group_labels", None)
        if pair_logits is not None and pair_labels is not None:
            logits_cpu = pair_logits.detach().cpu().tolist()
            labels_cpu = pair_labels.detach().cpu().tolist()
            stats["pair_prior_scores"].extend(float(v) for v in logits_cpu)
            stats["pair_prior_labels"].extend(int(v) for v in labels_cpu)

    @staticmethod
    def _finalize_stats(stats):
        n = max(stats["num_batches"], 1)
        n_group_all = max(stats["group_count_all"], 1)
        n_group_conflict = max(stats["group_count_conflict"], 1)
        pair_scores = stats["pair_prior_scores"]
        pair_labels = stats["pair_prior_labels"]
        return {
            "loss": stats["loss_sum"] / n,
            "loss_rank": stats["loss_rank_sum"] / n,
            "loss_group": stats["loss_group_sum"] / n,
            "loss_pair_prior": stats["loss_pair_prior_sum"] / n,
            "group_accuracy": stats["group_acc_correct_all"] / n_group_all,
            "conflict_group_accuracy": (
                stats["group_acc_correct_conflict"] / n_group_conflict
            ),
            "group_hits1": stats["group_hits1_conflict"] / n_group_conflict,
            "group_mrr": stats["group_rr_sum_conflict"] / n_group_conflict,
            "group_count_all": stats["group_count_all"],
            "group_count_conflict": stats["group_count_conflict"],
            "pair_prior_auc": roc_auc_binary(pair_scores, pair_labels),
            "pair_prior_pr_auc": pr_auc_binary(pair_scores, pair_labels),
            "pair_prior_group_count": len(pair_labels),
        }

    def _save_checkpoint(self, output_dir, filename):
        path = os.path.join(output_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_loss": self.best_loss,
        }, path)
