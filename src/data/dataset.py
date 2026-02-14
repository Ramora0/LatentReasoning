import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class BigMathDataset(Dataset):
    """Dataset for Big-Math-RL-Verified with separate question/answer tokenization."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_question_tokens: int = 512,
        max_answer_tokens: int = 64,
        split: str = "train",
        data_dir: str = None,
    ):
        self.tokenizer = tokenizer
        self.max_question_tokens = max_question_tokens
        self.max_answer_tokens = max_answer_tokens
        self._total_seen = 0
        self._discarded = 0

        if data_dir is not None:
            from datasets import load_from_disk
            ds = load_from_disk(data_dir)
            # load_from_disk may return DatasetDict or Dataset
            self.data = ds[split] if hasattr(ds, "keys") else ds
        else:
            from datasets import load_dataset
            self.data = load_dataset(
                "SynthLabsAI/Big-Math-RL-Verified", split=split
            )

    def __len__(self) -> int:
        return len(self.data)

    @property
    def discard_rate(self) -> float:
        """Fraction of samples discarded due to question length."""
        if self._total_seen == 0:
            return 0.0
        return self._discarded / self._total_seen

    def __getitem__(self, idx: int) -> dict | None:
        item = self.data[idx]

        question = item["problem"]
        answer = item["answer"]

        self._total_seen += 1

        question_enc = self.tokenizer(
            question,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        question_ids = question_enc["input_ids"].squeeze(0)

        # Discard questions that exceed the max length rather than truncating
        if len(question_ids) > self.max_question_tokens:
            self._discarded += 1
            return None

        answer_enc = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_answer_tokens,
            add_special_tokens=False,
            return_tensors="pt",
        )
        answer_ids = answer_enc["input_ids"].squeeze(0)

        # Append EOS token to answer
        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=answer_ids.dtype)
        answer_ids = torch.cat([answer_ids, eos])

        return {
            "question_ids": question_ids,
            "answer_ids": answer_ids,
            "question_len": len(question_ids),
            "answer_len": len(answer_ids),
        }
