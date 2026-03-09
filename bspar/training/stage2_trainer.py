"""Stage-2 training loop: Quad-Aware Reranker on real candidates."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from ..models.bspar_stage2 import BSPARStage2


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
        cand_mask[i, :n] = True

    return {
        "pair_reprs": pair_reprs,
        "cat_ids": cat_ids,
        "aff_input": aff_input,
        "meta_features": meta_features,
        "labels": labels,
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
            train_loss = self._train_epoch(epoch)
            dev_loss = self._evaluate()

            print(f"Epoch {epoch}/{self.config.stage2_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")

            if dev_loss < self.best_loss:
                self.best_loss = dev_loss
                self.patience_counter = 0
                self._save_checkpoint(output_dir, "best_stage2.pt")
                print(f"  → New best! Saved.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"  → Early stopping at epoch {epoch}")
                    break

        self._save_checkpoint(output_dir, "final_stage2.pt")
        print(f"Stage-2 training complete. Best dev loss: {self.best_loss:.4f}")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            result = self.model(
                pair_reprs=batch["pair_reprs"],
                cat_ids=batch["cat_ids"],
                aff_input=batch["aff_input"],
                meta_features=batch["meta_features"],
                labels=batch["labels"],
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

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in self.dev_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            result = self.model(
                pair_reprs=batch["pair_reprs"],
                cat_ids=batch["cat_ids"],
                aff_input=batch["aff_input"],
                meta_features=batch["meta_features"],
                labels=batch["labels"],
                mode="train",
            )

            total_loss += result["loss"].item()
            n += 1

        return total_loss / max(n, 1)

    def _save_checkpoint(self, output_dir, filename):
        path = os.path.join(output_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_loss": self.best_loss,
        }, path)
