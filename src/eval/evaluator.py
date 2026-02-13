import os

import torch
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from .metrics import extract_answer, numerical_compare


class Evaluator:
    """Evaluate latent reasoning model on GSM8K and MATH benchmarks."""

    def __init__(self, config, tokenizer: PreTrainedTokenizer, device: torch.device,
                 eval_data_dir: str = None):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = config.eval.max_new_tokens
        self.benchmarks = config.eval.benchmarks
        self.eval_data_dir = eval_data_dir

    def evaluate(self, model, K: int, p: float) -> dict:
        """Run evaluation on all configured benchmarks."""
        results = {}
        for benchmark in self.benchmarks:
            if benchmark == "gsm8k":
                acc = self._eval_gsm8k(model, K, p)
                results["gsm8k_accuracy"] = acc
            elif benchmark == "math":
                acc = self._eval_math(model, K, p)
                results["math_accuracy"] = acc
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

    def _eval_gsm8k(self, model, K: int, p: float) -> float:
        """Evaluate on GSM8K test set."""
        dataset = self._load_dataset("gsm8k", "gsm8k", hf_config="main")
        return self._eval_dataset(
            model, dataset, K, p,
            question_key="question",
            answer_key="answer",
            benchmark_name="GSM8K",
        )

    def _eval_math(self, model, K: int, p: float) -> float:
        """Evaluate on MATH test set."""
        dataset = self._load_dataset(
            "competition_math", "hendrycks/competition_math"
        )
        return self._eval_dataset(
            model, dataset, K, p,
            question_key="problem",
            answer_key="solution",
            benchmark_name="MATH",
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
    ) -> float:
        """Evaluate model on a dataset of math problems."""
        model.eval()
        correct = 0
        total = 0

        for item in tqdm(dataset, desc=f"Eval {benchmark_name}", leave=False):
            question = item[question_key]
            gold_answer_text = item[answer_key]
            gold = extract_answer(gold_answer_text)

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
                )

            # Decode and extract answer
            pred_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            pred = extract_answer(pred_text)

            if numerical_compare(pred, gold):
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        return accuracy
