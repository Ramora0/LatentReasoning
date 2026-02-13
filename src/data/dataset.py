import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class BigMathDataset(Dataset):
    """Dataset for Big-Math-RL-Verified with separate question/answer tokenization."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_question_tokens: int = 512,
        max_answer_tokens: int = 64,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_question_tokens = max_question_tokens
        self.max_answer_tokens = max_answer_tokens

        self.data = load_dataset(
            "SynthLabsAI/Big-Math-RL-Verified", split=split
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        question = item["problem"]
        answer = item["answer"]

        question_enc = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_question_tokens,
            add_special_tokens=True,
            return_tensors="pt",
        )
        question_ids = question_enc["input_ids"].squeeze(0)

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
