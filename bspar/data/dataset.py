"""PyTorch Dataset and collation for BSPAR Stage-1 training.

Handles tokenization with subword alignment, span label assignment,
and batch collation with padding.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .schema import Example
from .span_utils import enumerate_spans, compute_distance_bucket, compute_order


class BSPARStage1Dataset(Dataset):
    """Dataset for Stage-1 training.

    Each item contains:
    - Tokenized input (subword tokens with alignment to word tokens)
    - Span labels for aspect/opinion identification
    - Gold quads for pair/category/affective label assignment
    """

    def __init__(self, examples: list[Example], tokenizer_name: str,
                 max_length: int = 128, max_span_length: int = 8):
        self.examples = examples
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except (OSError, Exception):
            # Offline fallback: create a minimal word-level tokenizer
            from bspar.data._offline_tokenizer import OfflineTokenizer
            print(f"Warning: cannot load '{tokenizer_name}', "
                  f"using offline word tokenizer (dev/test only)")
            self.tokenizer = OfflineTokenizer(max_length=max_length)
        self.max_length = max_length
        self.max_span_length = max_span_length

        # Pre-tokenize all examples
        self.encoded = []
        for ex in examples:
            enc = self._encode_example(ex)
            if enc is not None:
                self.encoded.append(enc)

    def _encode_example(self, example: Example):
        """Tokenize and align word tokens to subword tokens.

        We use word-level spans, so we need a mapping from word index
        to subword token range.
        """
        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            example.text,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"]

        # Build word_to_subword mapping
        # offset_mapping gives (char_start, char_end) for each subword
        word_to_subword = self._align_words_to_subwords(
            example.tokens, example.token_offsets, offset_mapping
        )

        if not word_to_subword:
            return None

        num_words = len(example.tokens)

        # Remap gold quads from word-level to subword-level spans
        # For span enumeration, we work at WORD level (simpler & consistent)
        # The span representation uses subword tokens from the encoder

        return {
            "example_id": example.id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "num_words": num_words,
            "word_to_subword": word_to_subword,  # list of (sub_start, sub_end)
            "quads": example.quads,
            "tokens": example.tokens,
        }

    def _align_words_to_subwords(self, words, word_offsets, offset_mapping):
        """Map each word to its subword token range.

        Returns list of (subword_start, subword_end) for each word.
        subword_end is inclusive.
        """
        mapping = []

        for word_idx, (w_start, w_end) in enumerate(word_offsets):
            sub_start = None
            sub_end = None
            for sub_idx, (s_start, s_end) in enumerate(offset_mapping):
                if s_start == 0 and s_end == 0:
                    continue  # special token
                if s_start >= w_start and s_end <= w_end:
                    if sub_start is None:
                        sub_start = sub_idx
                    sub_end = sub_idx
                elif s_start >= w_end:
                    break

            if sub_start is not None:
                mapping.append((sub_start, sub_end))
            else:
                # Fallback: try overlapping match
                for sub_idx, (s_start, s_end) in enumerate(offset_mapping):
                    if s_end > w_start and s_start < w_end:
                        if sub_start is None:
                            sub_start = sub_idx
                        sub_end = sub_idx
                if sub_start is not None:
                    mapping.append((sub_start, sub_end))
                else:
                    mapping.append((0, 0))  # fallback

        return mapping

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def collate_stage1(batch, max_span_length=8):
    """Collate a batch of Stage-1 examples with padding.

    Returns a dict ready for BSPARStage1.forward().
    """
    batch_size = len(batch)

    # Pad input_ids and attention_mask
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(item["input_ids"])
        attention_mask[i, :seq_len] = torch.tensor(item["attention_mask"])

    # Collect gold quads and metadata
    gold_quads = [item["quads"] for item in batch]
    word_to_subword = [item["word_to_subword"] for item in batch]
    num_words = [item["num_words"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    example_ids = [item["example_id"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "gold_quads": gold_quads,
        "word_to_subword": word_to_subword,
        "num_words": num_words,
        "tokens": tokens,
        "example_ids": example_ids,
    }
