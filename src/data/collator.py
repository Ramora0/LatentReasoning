import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


@dataclass
class LatentReasoningCollator:
    """Pads question and answer sequences independently for latent reasoning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, batch: list[dict]) -> dict:
        # Find max lengths in this batch
        max_q_len = max(item["question_len"] for item in batch)
        max_a_len = max(item["answer_len"] for item in batch)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        question_ids_list = []
        answer_ids_list = []
        question_mask_list = []
        answer_mask_list = []
        labels_list = []

        for item in batch:
            q_ids = item["question_ids"]
            a_ids = item["answer_ids"]

            # Pad questions (right-pad)
            q_pad_len = max_q_len - len(q_ids)
            q_padded = torch.cat([q_ids, torch.full((q_pad_len,), pad_id, dtype=q_ids.dtype)])
            q_mask = torch.cat([torch.ones(len(q_ids), dtype=torch.long),
                                torch.zeros(q_pad_len, dtype=torch.long)])

            # Pad answers (right-pad)
            a_pad_len = max_a_len - len(a_ids)
            a_padded = torch.cat([a_ids, torch.full((a_pad_len,), pad_id, dtype=a_ids.dtype)])
            a_mask = torch.cat([torch.ones(len(a_ids), dtype=torch.long),
                                torch.zeros(a_pad_len, dtype=torch.long)])

            # Labels: answer tokens with padding set to -100
            labels = a_padded.clone()
            labels[a_mask == 0] = -100

            question_ids_list.append(q_padded)
            answer_ids_list.append(a_padded)
            question_mask_list.append(q_mask)
            answer_mask_list.append(a_mask)
            labels_list.append(labels)

        return {
            "question_ids": torch.stack(question_ids_list),
            "question_mask": torch.stack(question_mask_list),
            "answer_ids": torch.stack(answer_ids_list),
            "answer_mask": torch.stack(answer_mask_list),
            "labels": torch.stack(labels_list),
        }
