#!/usr/bin/env python3
"""Quick test: run the base Gemma 2 2B IT model on a math problem and stream output."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_NAME = "MCES10/maths-problems-gemma-2-2b-it"

PROBLEM = (
    r"Given $p$: $|4x-3|\leqslant 1$ and "
    r"$q$: $x^{2}-(2a+1)x+a^{2}+a\leqslant 0$, "
    r"find the range of values for $a$ if $p$ is a necessary "
    r"but not sufficient condition for $q$."
)

# ── Expected solution (for verification) ──────────────────────────────
# p: |4x−3| ≤ 1  ⟹  1/2 ≤ x ≤ 1        (set P = [1/2, 1])
# q: x²−(2a+1)x+a²+a ≤ 0
#    discriminant = 1, roots a and a+1     (set Q = [a, a+1])
#
# "p is necessary but not sufficient for q" means:
#   q ⟹ p  (Q ⊆ P)  but  p ⇏ q  (Q ≠ P, i.e. proper subset)
#
# Q ⊆ P requires  a ≥ 1/2  AND  a+1 ≤ 1  ⟹  1/2 ≤ a ≤ 0  (impossible)
#
# Note: Many textbook variants phrase this as "p is a sufficient but not
# necessary condition for q" (i.e. P ⊆ Q):
#   a ≤ 1/2  AND  a+1 ≥ 1  ⟹  0 ≤ a ≤ 1/2
# That gives the clean answer  a ∈ [0, 1/2].
#
# The model's reasoning will help us see which interpretation it uses.
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", action="store_true",
        help="Tokenize like training does: raw text with add_special_tokens=True, no chat template",
    )
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    # Load tokenizer + model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    if args.raw:
        # Match training: raw text, add_special_tokens=True, no chat template
        mode = "RAW (training-style)"
        inputs = tokenizer(
            PROBLEM,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        prompt_display = PROBLEM
    else:
        # Chat template (what the IT model expects)
        mode = "CHAT (instruction-tuned)"
        prompt_display = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROBLEM}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_display, return_tensors="pt").to(device)

    print("=" * 60)
    print(f"MODE: {mode}")
    print("=" * 60)
    print("PROMPT:")
    print("=" * 60)
    print(prompt_display)
    print("=" * 60)
    print("MODEL OUTPUT (streaming):")
    print("=" * 60)

    # Stream generation
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,          # greedy for reproducibility
            streamer=streamer,
        )

    # Print full decoded answer
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("FULL ANSWER:")
    print("=" * 60)
    print(answer_text)


if __name__ == "__main__":
    main()
