"""Minimal offline tokenizer for development/testing when HuggingFace
models cannot be downloaded.

This tokenizer splits text on whitespace, assigns each word a unique ID
(with a fixed vocab), and produces offset_mapping compatible with the
BSPARStage1Dataset word-to-subword alignment logic.

NOT for production use — use a real pretrained tokenizer for real experiments.
"""


class OfflineTokenizer:
    """Word-level tokenizer that mimics HuggingFace tokenizer interface."""

    def __init__(self, max_length: int = 128, vocab_size: int = 50265):
        self.max_length = max_length
        self.vocab_size = vocab_size
        # Reserve special tokens: 0=<pad>, 1=<s>, 2=</s>, 3=<unk>
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self._word2id = {}
        self._next_id = 4

    def _get_id(self, word: str) -> int:
        if word not in self._word2id:
            if self._next_id < self.vocab_size:
                self._word2id[word] = self._next_id
                self._next_id += 1
            else:
                return self.unk_token_id
        return self._word2id[word]

    def __call__(self, text: str, return_offsets_mapping: bool = False,
                 max_length: int = None, truncation: bool = False,
                 padding: bool = False, return_tensors=None, **kwargs):
        """Tokenize text, mimicking HuggingFace tokenizer output."""
        max_len = max_length or self.max_length

        words = text.split()
        # Build offset mapping (char_start, char_end) for each word
        offsets = []
        pos = 0
        for w in words:
            start = text.index(w, pos)
            end = start + len(w)
            offsets.append((start, end))
            pos = end

        # Truncate if needed (leave room for <s> and </s>)
        if truncation and len(words) > max_len - 2:
            words = words[:max_len - 2]
            offsets = offsets[:max_len - 2]

        # Build input_ids: <s> word1 word2 ... </s>
        input_ids = [self.bos_token_id]
        offset_mapping = [(0, 0)]  # <s> has no char span

        for word, (cs, ce) in zip(words, offsets):
            input_ids.append(self._get_id(word.lower()))
            offset_mapping.append((cs, ce))

        input_ids.append(self.eos_token_id)
        offset_mapping.append((0, 0))  # </s>

        attention_mask = [1] * len(input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_offsets_mapping:
            result["offset_mapping"] = offset_mapping

        return result
