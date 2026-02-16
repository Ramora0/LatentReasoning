import os

import torch
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from .metrics import check_answer


class Evaluator:
    """Evaluate latent reasoning model on GSM8K benchmark."""

    def __init__(self, config, tokenizer: PreTrainedTokenizer, device: torch.device,
                 eval_data_dir: str = None):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = config.eval.max_new_tokens
        self.benchmarks = config.eval.benchmarks
        self.eval_data_dir = eval_data_dir

        self.use_xla = getattr(config.distributed, "backend", "cuda") == "xla"
        if self.use_xla:
            import torch_xla.core.xla_model as xm
            self._xm = xm

    def evaluate(self, model, K: int, p: float, q_visibility: float = 1.0) -> dict:
        """Run evaluation on all configured benchmarks."""
        results = {}
        for benchmark in self.benchmarks:
            if benchmark == "gsm8k":
                acc, correct, total = self._eval_gsm8k(model, K, p, q_visibility)
                results["gsm8k_accuracy"] = acc
                results["gsm8k_correct"] = correct
                results["gsm8k_total"] = total
        return results

    def _load_dataset(self, local_subdir, hf_name, hf_config=None, split="test"):
        """Load a dataset from local disk (if available) or HuggingFace Hub."""
        if self.eval_data_dir:
            local_path = os.path.join(self.eval_data_dir, local_subdir)
            if os.path.isdir(local_path):
                from datasets import load_from_disk
                ds = load_from_disk(local_path)
                return ds[split] if hasattr(ds, "keys") else ds
        from datasets import load_dataset
        return load_dataset(hf_name, hf_config, split=split)

    def _eval_gsm8k(self, model, K: int, p: float, q_visibility: float = 1.0) -> float:
        """Evaluate on GSM8K test set."""
        dataset = self._load_dataset("gsm8k", "gsm8k", hf_config="main")
        return self._eval_dataset(
            model, dataset, K, p,
            question_key="question",
            answer_key="answer",
            benchmark_name="GSM8K",
            q_visibility=q_visibility,
        )

    def _eval_dataset(
        self,
        model,
        dataset,
        K: int,
        p: float,
        question_key: str,
        answer_key: str,
        benchmark_name: str,
        q_visibility: float = 1.0,
    ) -> tuple[float, int, int]:
        """Evaluate model on a dataset of math problems.

        Returns:
            (accuracy, correct_count, total_count)
        """
        model.eval()
        correct = 0
        total = 0

        for item in tqdm(dataset, desc=f"Eval {benchmark_name}", leave=False):
            question = item[question_key]
            gold_text = item[answer_key]

            # Tokenize question
            enc = self.tokenizer(
                question,
                truncation=True,
                max_length=self.config.data.max_question_tokens,
                add_special_tokens=True,
                return_tensors="pt",
            )
            question_ids = enc["input_ids"].to(self.device)
            question_mask = enc["attention_mask"].to(self.device)

            # Generate answer
            with torch.no_grad():
                # Handle FSDP-wrapped models
                gen_model = model
                if hasattr(model, "module"):
                    gen_model = model.module

                generated_ids = gen_model.generate_answer(
                    question_ids=question_ids,
                    question_mask=question_mask,
                    K=K,
                    p=p,
                    max_new_tokens=self.max_new_tokens,
                    q_visibility=q_visibility,
                )

            # On XLA, mark_step() flushes the lazy graph so it actually executes
            # (same pattern as training's mark_step after each micro-batch).
            if self.use_xla:
                self._xm.mark_step()

            pred_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            if check_answer(pred_text, gold_text):
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        return accuracy, correct, total
