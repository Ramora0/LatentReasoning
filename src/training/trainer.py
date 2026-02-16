import os
import math
import time
import shutil
import functools
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from tqdm import tqdm

from ..model.latent_gemma import LatentReasoningModel
from .curriculum import CurriculumScheduler

try:
    import wandb
except ImportError:
    wandb = None


def _get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LatentReasoningTrainer:
    """Training loop with FSDP, mixed precision, and gradient accumulation.

    Supports both CUDA (GPU/NCCL) and XLA (TPU) backends via config.distributed.backend.
    """

    def __init__(self, config, model, train_dataset, collator, evaluator=None, tokenizer=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.collator = collator
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.K = config.latent.K
        stitching_depth = getattr(config.latent, "stitching_depth", None)
        self.stitching_depth = stitching_depth if stitching_depth is not None else self.K
        self._stitch_debug = {}

        self.backend = getattr(config.distributed, "backend", "cuda")
        self.use_xla = self.backend == "xla"

        # Import XLA modules lazily
        if self.use_xla:
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            self._xm = xm
            self._pl = pl
            self._torch_xla = torch_xla

        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main = self.rank == 0

        # Device selection
        if self.use_xla:
            self.device = self._xm.xla_device()
        else:
            self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

        # Autocast device type
        self._autocast_device = "xla" if self.use_xla else "cuda"

        # Wrap model with FSDP if distributed (CUDA only).
        # On XLA, skip FSDP — Gemma 2 2B fits on each TPU chip and
        # xm.optimizer_step handles gradient all-reduce automatically.
        # XLA FSDP's flatten-params wrapper breaks with retain_graph=True
        # (needed for gradient stitching) and with 1D norm parameters.
        if self.is_distributed and config.distributed.fsdp and not self.use_xla:
            self.model = self._wrap_fsdp(model)
        else:
            self.model = model.to(self.device)

        # Build optimizer with two param groups
        self.optimizer = self._build_optimizer()

        # DataLoader (must be built before scheduler so we can compute total steps)
        self.grad_accum_steps = config.training.gradient_accumulation_steps
        self.dataloader = self._build_dataloader()

        # Compute total optimizer steps from num_epochs
        self.num_epochs = config.training.num_epochs
        batches_per_epoch = len(self.dataloader)  # micro-batches per epoch per GPU
        optimizer_steps_per_epoch = batches_per_epoch // self.grad_accum_steps
        self.max_optimizer_steps = optimizer_steps_per_epoch * self.num_epochs

        if self.is_main:
            print(f"Training for {self.num_epochs} epochs")
            print(f"  {batches_per_epoch} micro-batches/epoch, "
                  f"{optimizer_steps_per_epoch} optimizer steps/epoch, "
                  f"{self.max_optimizer_steps} total optimizer steps")

        # Curriculum (needs max_optimizer_steps to compute p annealing)
        self.curriculum = CurriculumScheduler(config, self.max_optimizer_steps)

        # Build scheduler
        num_warmup = int(self.max_optimizer_steps * config.training.warmup_ratio)
        self.scheduler = _get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup, self.max_optimizer_steps
        )

        # Convert sample-based intervals to optimizer-step-based intervals
        samples_per_step = (
            config.training.batch_size_per_gpu
            * self.world_size
            * self.grad_accum_steps
        )
        self.log_every = max(1, round(config.logging.log_every_samples / samples_per_step))
        self.eval_every = max(1, round(config.eval.eval_every_samples / samples_per_step))
        self.save_every = max(1, round(config.checkpointing.save_every_samples / samples_per_step))

        if self.is_main:
            print(f"  Effective batch size: {samples_per_step} samples/step")
            print(f"  log every {self.log_every} steps, eval every {self.eval_every} steps, "
                  f"save every {self.save_every} steps")

        # State
        self.optimizer_step = 0
        self._prev_compile_count = 0
        self._prev_compile_time_ns = 0
        # Per-mark_step compilation tracking
        self._snap_compile_count = 0
        self._snap_compile_time_ns = 0
        self._prev_xla_compile_count = 0

        # Checkpointing
        self.save_dir = Path(config.checkpointing.save_dir)
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        if self.is_main and wandb is not None:
            wandb.init(
                project=config.logging.wandb_project,
                name=getattr(config.logging, "wandb_run_name", None),
                config=dict(config),
            )

    # ---- FSDP wrapping (backend-aware) ----

    def _wrap_fsdp(self, model):
        """Wrap model with FSDP (CUDA or XLA variant)."""
        if self.use_xla:
            return self._wrap_xla_fsdp(model)
        return self._wrap_cuda_fsdp(model)

    def _wrap_cuda_fsdp(self, model):
        """Wrap model with torch.distributed FSDP for CUDA/NCCL."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Gemma2DecoderLayer},
        )

        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = strategy_map.get(
            self.config.distributed.fsdp_sharding_strategy,
            ShardingStrategy.FULL_SHARD,
        )

        mp_policy = None
        if self.config.distributed.fsdp_mixed_precision == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )

        cpu_offload = None
        if self.config.distributed.fsdp_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            device_id=self.device,
            use_orig_params=True,
        )

        return model

    def _wrap_xla_fsdp(self, model):
        """Wrap model with XLA FSDP for TPU.

        Uses outer-only FSDP (no per-layer wrapping) to stay compatible with
        retain_graph=True needed by gradient stitching.  Nested XLA FSDP tracks
        per-module backward state that breaks when the graph is traversed twice.
        Gemma 2 2B is small enough for outer-only sharding on v4-8.
        """
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XlaFSDP

        # Move model to XLA device first
        model = model.to(self.device)

        # Outer-only wrap — no auto_wrap_policy so inner decoder layers are
        # NOT individually wrapped.  This avoids the BACKWARD_PRE/POST state
        # conflict when retain_graph=True is used for gradient stitching.
        model = XlaFSDP(
            model,
            compute_dtype=torch.bfloat16,
            shard_param_on_dim_0=True,
            pin_layout_in_collective_ops=True,
        )

        return model

    # ---- Optimizer ----

    def _build_optimizer(self):
        """Build AdamW for all trainable parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    # ---- DataLoader ----

    def _build_dataloader(self):
        """Build DataLoader with appropriate sampler for backend."""
        sampler = None
        shuffle = True

        if self.use_xla and self.is_distributed:
            import torch_xla.distributed.parallel_loader as pl
            # For XLA we build a standard DataLoader; wrapping with MpDeviceLoader
            # is done in the train loop
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            shuffle = False
        elif self.is_distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            shuffle = False

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size_per_gpu,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=self.collator,
            num_workers=4,
            pin_memory=not self.use_xla,  # pin_memory is CUDA-only
            drop_last=True,
        )

        return loader

    # ---- Gradient clipping (backend-aware) ----

    def _clip_grad_norm(self):
        """Clip gradient norms, returning the total norm.

        On XLA, returns the tensor without .item() to avoid forcing graph
        materialization (which creates a memory spike). The value is
        materialized later at logging time after mark_step.
        """
        max_norm = self.config.training.max_grad_norm

        if self.use_xla:
            # No XLA FSDP — model is plain, use standard clip
            # Return tensor — defer .item() to logging time
            return torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm
            )
        else:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            if self.is_distributed and isinstance(self.model, FSDP):
                grad_norm = self.model.clip_grad_norm_(max_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm
                )
            return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    # ---- XLA diagnostics ----

    def _xla_snap(self):
        """Snapshot current XLA compilation counters."""
        if not self.use_xla:
            return
        import torch_xla.debug.metrics as met
        compile_data = met.metric_data('CompileTime')
        if compile_data:
            # compile_data = (count, total_time_ns)
            self._snap_compile_count = compile_data[0]
            self._snap_compile_time_ns = compile_data[1]

    def _xla_check(self, label):
        """Print XLA compilations since last _xla_snap call."""
        if not self.use_xla:
            return
        import torch_xla.debug.metrics as met
        compile_data = met.metric_data('CompileTime')
        if compile_data:
            # compile_data = (count, total_time_ns)
            delta_count = compile_data[0] - self._snap_compile_count
            delta_time = (compile_data[1] - self._snap_compile_time_ns) / 1e9
            if delta_count > 0:
                print(f"  [XLA:{label}] +{delta_count} compilations ({delta_time:.1f}s)",
                      flush=True)
            else:
                print(f"  [XLA:{label}] no new compilations", flush=True)

    # ---- XLA step helper ----

    def _xla_step(self):
        """Execute optimizer step with XLA mark_step for graph execution."""
        self._xm.optimizer_step(self.optimizer)
        self.scheduler.step()

    # ---- Batch generation helpers ----

    def _collect_micro_batches(self, epoch_loader_iter, count):
        """Collect `count` micro-batches from the dataloader iterator.

        Skips None batches. Returns list of batch dicts (already on device).
        May return fewer than `count` if the iterator is exhausted.
        """
        batches = []
        while len(batches) < count:
            try:
                batch = next(epoch_loader_iter)
            except StopIteration:
                break
            if batch is None:
                continue
            if not self.use_xla:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            batches.append(batch)
        return batches

    def _merge_batches(self, batches):
        """Merge micro-batches into one big batch, re-padding to common lengths.

        Questions are left-padded, so we add more left-padding to shorter ones.
        Answers are right-padded, so we add more right-padding to shorter ones.

        Returns:
            (merged_batch, split_sizes): merged dict of tensors and list of
            per-micro-batch batch sizes for splitting later.
        """
        split_sizes = [b["question_ids"].shape[0] for b in batches]

        # Find max lengths across all micro-batches
        max_q_len = max(b["question_ids"].shape[1] for b in batches)
        max_a_len = max(b["answer_ids"].shape[1] for b in batches)

        padded_q_ids = []
        padded_q_mask = []
        padded_a_ids = []
        padded_a_mask = []

        for b in batches:
            q_ids = b["question_ids"]
            q_mask = b["question_mask"]
            a_ids = b["answer_ids"]
            a_mask = b["answer_mask"]

            # Left-pad questions
            q_pad = max_q_len - q_ids.shape[1]
            if q_pad > 0:
                q_ids = torch.cat([torch.zeros(q_ids.shape[0], q_pad, dtype=q_ids.dtype, device=q_ids.device), q_ids], dim=1)
                q_mask = torch.cat([torch.zeros(q_mask.shape[0], q_pad, dtype=q_mask.dtype, device=q_mask.device), q_mask], dim=1)

            # Right-pad answers
            a_pad = max_a_len - a_ids.shape[1]
            if a_pad > 0:
                a_ids = torch.cat([a_ids, torch.zeros(a_ids.shape[0], a_pad, dtype=a_ids.dtype, device=a_ids.device)], dim=1)
                a_mask = torch.cat([a_mask, torch.zeros(a_mask.shape[0], a_pad, dtype=a_mask.dtype, device=a_mask.device)], dim=1)

            padded_q_ids.append(q_ids)
            padded_q_mask.append(q_mask)
            padded_a_ids.append(a_ids)
            padded_a_mask.append(a_mask)

        merged = {
            "question_ids": torch.cat(padded_q_ids, dim=0),
            "question_mask": torch.cat(padded_q_mask, dim=0),
            "answer_ids": torch.cat(padded_a_ids, dim=0),
            "answer_mask": torch.cat(padded_a_mask, dim=0),
        }
        return merged, split_sizes

    def _split_generate_outputs(self, gen_outputs, split_sizes):
        """Split generate phase outputs back into micro-batch-sized chunks."""
        latent_chunks = torch.split(gen_outputs["latent_states"], split_sizes, dim=0)
        mask_chunks = torch.split(gen_outputs["latent_mask"], split_sizes, dim=0)

        # Split thought_inputs: list of K tensors, each [big_B, 1, d_model]
        thought_inputs = gen_outputs["thought_inputs"]
        thought_input_chunks = []
        if thought_inputs is not None:
            for i, size in enumerate(split_sizes):
                start = sum(split_sizes[:i])
                chunk = [t[start:start + size] for t in thought_inputs]
                thought_input_chunks.append(chunk)
        else:
            thought_input_chunks = [None] * len(split_sizes)

        return latent_chunks, mask_chunks, thought_input_chunks

    # ---- Training loop ----

    def _backward_and_stitch(self, outputs, p, accum_steps):
        """Run backward pass with gradient stitching for one micro-batch.

        Returns (stitch_debug dict, loss_value).
        """
        loss = outputs["loss"] / accum_steps
        thought_inputs = outputs.get("thought_inputs")
        thought_outputs = outputs.get("thought_outputs")
        D = self.stitching_depth
        stitch_debug = {}

        if thought_inputs and thought_outputs and D > 0:
            # Compute thought-token similarity: how close are latent thoughts
            # to their nearest decoded token embedding?
            thought_token_ids = outputs.get("thought_token_ids")
            if thought_token_ids is not None:
                with torch.no_grad():
                    raw_model = getattr(self.model, "module", self.model)
                    thought_cat = torch.cat(thought_inputs, dim=1)  # [B, K, d_model]
                    tok_emb = raw_model.embed_tokens(thought_token_ids) * raw_model.normalizer  # [B, K, d_model]
                    cos_sim = F.cosine_similarity(thought_cat, tok_emb, dim=-1)  # [B, K]
                    stitch_debug["_thought_token_cos_sim"] = cos_sim.mean()

            t_back = time.perf_counter()
            loss.backward(retain_graph=True)
            print(f"[backward] {time.perf_counter()-t_back:.2f}s", flush=True)

            decode_input = outputs.get("decode_input")
            has_decode_grad = decode_input is not None and decode_input.grad is not None
            has_thought_grads = all(t.grad is not None for t in thought_inputs)
            print(f"[stitch_branch] decode_grad={has_decode_grad} "
                  f"thought_grads={has_thought_grads}", flush=True)
            if has_decode_grad:
                t_gd = time.perf_counter()
                grad = decode_input.grad
                t_start, t_end = outputs["thought_positions"]
                a_start = outputs["answer_positions"][0]
                thought_grad = grad[:, t_start:t_end, :]
                answer_grad = grad[:, a_start:, :]
                stitch_debug["_thought_mean_t"] = thought_grad.norm(dim=-1).mean()
                stitch_debug["_answer_mean_t"] = answer_grad.norm(dim=-1).mean()
                stitch_debug["_thought_total_t"] = thought_grad.norm()
                stitch_debug["_answer_total_t"] = answer_grad.norm()
                print(f"[grad_decomp] {time.perf_counter()-t_gd:.2f}s", flush=True)

            t_stitch = time.perf_counter()
            for d in range(D):
                g = [t.grad.detach().clone() if t.grad is not None
                     else torch.zeros_like(t) for t in thought_inputs]
                for t in thought_inputs:
                    t.grad = None

                if d == 0:
                    g_sq_sum = sum(gk.norm().pow(2) for gk in g)
                    stitch_debug["_thought_sensitivity_t"] = g_sq_sum.sqrt()

                L_stitch = (1 - p) * sum(
                    (g[k] * thought_outputs[k]).sum()
                    for k in range(len(g))
                )

                t_sb = time.perf_counter()
                retain = (d < D - 1)
                L_stitch.backward(retain_graph=retain)
                print(f"[stitch d={d}] backward {time.perf_counter()-t_sb:.2f}s", flush=True)
            print(f"[stitch] D={D} total {time.perf_counter()-t_stitch:.2f}s", flush=True)
        else:
            t_back = time.perf_counter()
            loss.backward()
            print(f"[backward] {time.perf_counter()-t_back:.2f}s", flush=True)

        return stitch_debug, loss

    def train(self):
        """Main training loop.

        Iterates for exactly num_epochs over the dataset. All schedules
        (LR, p-annealing) and logging intervals are based on optimizer
        steps, which are deterministic given (dataset_size, batch_size,
        grad_accum, world_size, num_epochs).

        When gradient_accumulation_steps > 1, Phase 1+2 (generation) runs on
        the full accumulated batch for better GPU utilization, then Phase 3
        (decode + backward + stitching) runs on each micro-batch sequentially.
        """
        self.model.train()
        max_steps = self.max_optimizer_steps
        log_every = self.log_every
        eval_every = self.eval_every
        save_every = self.save_every

        accum_loss = 0.0       # loss within current grad-accum window
        log_loss = 0.0         # loss accumulated over logging window
        log_grad_norm = 0.0    # grad_norm accumulated over logging window
        log_steps = 0          # optimizer steps since last log
        # Gradient decomposition accumulators (averaged over logging window)
        log_thought_mean = 0.0
        log_answer_mean = 0.0
        log_thought_total = 0.0
        log_answer_total = 0.0
        log_sensitivity = 0.0
        log_thought_token_sim = 0.0
        log_grad_decomp_steps = 0  # may differ from log_steps if some steps lack grad info
        step_start_time = time.monotonic()

        pbar = tqdm(
            total=max_steps,
            desc="Training",
            disable=not self.is_main,
        )
        pbar.update(self.optimizer_step)

        for epoch in range(self.num_epochs):
            # Set epoch for distributed sampler (shuffles differently each epoch)
            if self.is_distributed and hasattr(self.dataloader, "sampler"):
                sampler = self.dataloader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            # For XLA, wrap the dataloader with MpDeviceLoader each epoch
            if self.use_xla:
                epoch_loader = self._pl.MpDeviceLoader(self.dataloader, self.device)
            else:
                epoch_loader = self.dataloader

            if self.K == 0:
                # Baseline mode: no latent steps, no stitching, simple forward+backward
                epoch_loader_iter = iter(epoch_loader)
                while True:
                    micro_batches = self._collect_micro_batches(
                        epoch_loader_iter, self.grad_accum_steps
                    )
                    if not micro_batches:
                        break

                    actual_accum = len(micro_batches)
                    for batch in micro_batches:
                        with torch.autocast(
                            device_type=self._autocast_device,
                            dtype=torch.bfloat16,
                            enabled=self.config.training.bf16,
                        ):
                            outputs = self.model(
                                question_ids=batch["question_ids"],
                                question_mask=batch["question_mask"],
                                answer_ids=batch["answer_ids"],
                                answer_mask=batch["answer_mask"],
                                K=0,
                                p=0.0,
                            )

                        loss = outputs["loss"] / actual_accum
                        loss.backward()

                        if self.use_xla:
                            accum_loss += loss.detach()
                        else:
                            accum_loss += loss.item()

                    self._stitch_debug = {}
                    self._last_thought_token_ids = None

                    accum_loss, log_loss, log_grad_norm, log_steps, \
                        log_thought_mean, log_answer_mean, log_thought_total, \
                        log_answer_total, log_sensitivity, log_thought_token_sim, \
                        log_grad_decomp_steps, \
                        step_start_time = self._do_optimizer_step(
                            accum_loss, log_loss, log_grad_norm, log_steps,
                            log_thought_mean, log_answer_mean, log_thought_total,
                            log_answer_total, log_sensitivity, log_thought_token_sim,
                            log_grad_decomp_steps,
                            step_start_time, 0, 0.0, epoch, pbar,
                            max_steps, log_every, eval_every, save_every,
                        )

            elif self.grad_accum_steps > 1:
                # Batched generation path: collect micro-batches, run Phase 1+2
                # on the full batch, then split for Phase 3+backward+stitching
                epoch_loader_iter = iter(epoch_loader)
                while True:
                    micro_batches = self._collect_micro_batches(
                        epoch_loader_iter, self.grad_accum_steps
                    )
                    if not micro_batches:
                        break

                    K = self.K
                    p = self.curriculum.get_p(self.optimizer_step)
                    actual_accum = len(micro_batches)

                    # Merge micro-batches and run Phase 1+2 on big batch
                    merged_batch, split_sizes = self._merge_batches(micro_batches)

                    with torch.no_grad():
                        with torch.autocast(
                            device_type=self._autocast_device,
                            dtype=torch.bfloat16,
                            enabled=self.config.training.bf16,
                        ):
                            gen_outputs = self.model(
                                question_ids=merged_batch["question_ids"],
                                question_mask=merged_batch["question_mask"],
                                K=K,
                                p=p,
                                phase="generate",
                            )

                    # Split outputs for micro-batch decode
                    latent_chunks, mask_chunks, thought_input_chunks = \
                        self._split_generate_outputs(gen_outputs, split_sizes)

                    # Split answer tensors back to micro-batch sizes
                    answer_id_chunks = torch.split(merged_batch["answer_ids"], split_sizes, dim=0)
                    answer_mask_chunks = torch.split(merged_batch["answer_mask"], split_sizes, dim=0)

                    thought_output_start = gen_outputs["thought_output_start"]

                    # Phase 3 + backward + stitching for each micro-batch
                    for i in range(actual_accum):
                        # Re-create thought_inputs with requires_grad for this micro-batch
                        micro_thought_inputs = None
                        if thought_input_chunks[i] is not None:
                            micro_thought_inputs = [
                                t.detach().clone().requires_grad_(True)
                                for t in thought_input_chunks[i]
                            ]

                        with torch.autocast(
                            device_type=self._autocast_device,
                            dtype=torch.bfloat16,
                            enabled=self.config.training.bf16,
                        ):
                            outputs = self.model(
                                answer_ids=answer_id_chunks[i],
                                answer_mask=answer_mask_chunks[i],
                                latent_states=latent_chunks[i],
                                latent_mask=mask_chunks[i],
                                thought_inputs=micro_thought_inputs,
                                thought_output_start=thought_output_start,
                                K_for_stitching=K,
                                phase="decode",
                            )

                        stitch_debug, loss = self._backward_and_stitch(
                            outputs, p, actual_accum
                        )

                        if self.use_xla:
                            accum_loss += loss.detach()
                            self._xla_snap()
                            t_ms = time.perf_counter()
                            self._xm.mark_step()
                            print(f"[mark_step] {time.perf_counter()-t_ms:.2f}s", flush=True)
                            self._xla_check(f"decode_micro{i}")
                        else:
                            accum_loss += loss.item()

                    self._stitch_debug = stitch_debug
                    self._last_thought_token_ids = gen_outputs.get("thought_token_ids")

                    # Optimizer step
                    accum_loss, log_loss, log_grad_norm, log_steps, \
                        log_thought_mean, log_answer_mean, log_thought_total, \
                        log_answer_total, log_sensitivity, log_thought_token_sim, \
                        log_grad_decomp_steps, \
                        step_start_time = self._do_optimizer_step(
                            accum_loss, log_loss, log_grad_norm, log_steps,
                            log_thought_mean, log_answer_mean, log_thought_total,
                            log_answer_total, log_sensitivity, log_thought_token_sim,
                            log_grad_decomp_steps,
                            step_start_time, K, p, epoch, pbar,
                            max_steps, log_every, eval_every, save_every,
                        )

            else:
                # Single micro-batch path (grad_accum_steps == 1): use phase="all"
                for batch in epoch_loader:
                    if batch is None:
                        continue

                    if not self.use_xla:
                        batch = {k: v.to(self.device) for k, v in batch.items()}

                    K = self.K
                    p = self.curriculum.get_p(self.optimizer_step)

                    with torch.autocast(
                        device_type=self._autocast_device,
                        dtype=torch.bfloat16,
                        enabled=self.config.training.bf16,
                    ):
                        outputs = self.model(
                            question_ids=batch["question_ids"],
                            question_mask=batch["question_mask"],
                            answer_ids=batch["answer_ids"],
                            answer_mask=batch["answer_mask"],
                            K=K,
                            p=p,
                        )

                    stitch_debug, loss = self._backward_and_stitch(outputs, p, 1)

                    self._stitch_debug = stitch_debug
                    self._last_thought_token_ids = outputs.get("thought_token_ids")
                    if self.use_xla:
                        accum_loss += loss.detach()
                        # No mark_step here — let xm.optimizer_step() compile
                        # the full step (fwd+bwd+stitch+clip+optim) as one graph.
                        # Separate mark_steps split the graph into pieces that may
                        # not cache well with retain_graph=True.
                    else:
                        accum_loss += loss.item()

                    # Optimizer step (always, since grad_accum_steps == 1)
                    accum_loss, log_loss, log_grad_norm, log_steps, \
                        log_thought_mean, log_answer_mean, log_thought_total, \
                        log_answer_total, log_sensitivity, log_thought_token_sim, \
                        log_grad_decomp_steps, \
                        step_start_time = self._do_optimizer_step(
                            accum_loss, log_loss, log_grad_norm, log_steps,
                            log_thought_mean, log_answer_mean, log_thought_total,
                            log_answer_total, log_sensitivity, log_thought_token_sim,
                            log_grad_decomp_steps,
                            step_start_time, K, p, epoch, pbar,
                            max_steps, log_every, eval_every, save_every,
                        )

            if self.is_main:
                print(f"\nEpoch {epoch + 1}/{self.num_epochs} complete "
                      f"(optimizer step {self.optimizer_step}/{max_steps})")

        pbar.close()

        # Final save
        self._save_checkpoint()

        if self.is_main and wandb is not None and wandb.run is not None:
            wandb.finish()

    def _do_optimizer_step(self, accum_loss, log_loss, log_grad_norm, log_steps,
                           log_thought_mean, log_answer_mean, log_thought_total,
                           log_answer_total, log_sensitivity, log_thought_token_sim,
                           log_grad_decomp_steps,
                           step_start_time, K, p, epoch, pbar,
                           max_steps, log_every, eval_every, save_every):
        """Execute optimizer step, logging, eval, and checkpoint. Returns updated accumulators."""
        # Gradient clipping
        t_clip = time.perf_counter()
        grad_norm = self._clip_grad_norm()
        print(f"[clip_grad] {time.perf_counter()-t_clip:.2f}s", flush=True)

        # Optimizer step (XLA needs mark_step which triggers graph execution)
        t_opt = time.perf_counter()
        if self.use_xla:
            self._xla_snap()
            print("[optim] calling xm.optimizer_step...", flush=True)
            self._xla_step()
            print(f"[optim] xla_step done {time.perf_counter()-t_opt:.2f}s", flush=True)
            self._xla_check("optim_step")
        else:
            self.optimizer.step()
            self.scheduler.step()
        t_zero = time.perf_counter()
        self.optimizer.zero_grad()
        print(f"[zero_grad] {time.perf_counter()-t_zero:.2f}s", flush=True)
        self.optimizer_step += 1
        print(f"[optim] total {time.perf_counter()-t_opt:.2f}s", flush=True)

        # Per-step XLA recompilation tracking
        if self.use_xla:
            import torch_xla.debug.metrics as met
            compile_data = met.metric_data('CompileTime')
            cur_count = compile_data[0] if compile_data else 0
            delta = cur_count - self._prev_xla_compile_count
            if delta > 0:
                print(f"  [XLA] step {self.optimizer_step}: +{delta} compilations "
                      f"(total={cur_count})", flush=True)
            self._prev_xla_compile_count = cur_count

        # Accumulate stats for logging window
        # On XLA, avoid .item() which breaks the lazy graph — accumulate as tensors,
        # only materialize inside the logging block
        if self.use_xla:
            if self.is_main:
                loss_val = accum_loss.detach() if isinstance(accum_loss, torch.Tensor) else torch.tensor(accum_loss, device=self.device)
                grad_val = grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm, device=self.device)
                if not hasattr(self, '_xla_log_losses'):
                    self._xla_log_losses = []
                    self._xla_log_grads = []
                self._xla_log_losses.append(loss_val)
                self._xla_log_grads.append(grad_val)
        else:
            log_loss += accum_loss
            log_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        log_steps += 1
        accum_loss = 0.0

        # Accumulate gradient decomposition stats (skip .item() on XLA)
        if hasattr(self, '_stitch_debug') and self._stitch_debug:
            sd = self._stitch_debug
            if not self.use_xla:
                if "_thought_mean_t" in sd:
                    log_thought_mean += sd["_thought_mean_t"].item()
                    log_answer_mean += sd["_answer_mean_t"].item()
                    log_thought_total += sd["_thought_total_t"].item()
                    log_answer_total += sd["_answer_total_t"].item()
                if "_thought_sensitivity_t" in sd:
                    log_sensitivity += sd["_thought_sensitivity_t"].item()
                if "_thought_token_cos_sim" in sd:
                    log_thought_token_sim += sd["_thought_token_cos_sim"].item()
            log_grad_decomp_steps += 1

        # Logging (based on optimizer steps)
        if self.is_main and self.optimizer_step % log_every == 0:
            t_log = time.perf_counter()
            # On XLA, materialize accumulated tensor stats now (inside logging block).
            # Tensors are already materialized from xm.optimizer_step()'s mark_step().
            if self.use_xla and hasattr(self, '_xla_log_losses') and self._xla_log_losses:
                log_loss = sum(t.item() for t in self._xla_log_losses)
                log_grad_norm = sum(t.item() for t in self._xla_log_grads)
                self._xla_log_losses.clear()
                self._xla_log_grads.clear()
            avg_loss = log_loss / log_steps
            avg_grad_norm = log_grad_norm / log_steps
            step_time = time.monotonic() - step_start_time
            samples_per_sec = (
                log_steps * self.grad_accum_steps
                * self.config.training.batch_size_per_gpu
                * self.world_size / step_time
            )
            lrs = self.scheduler.get_last_lr()
            log_dict = {
                "train/loss": avg_loss,
                "train/perplexity": math.exp(min(avg_loss, 20)),
                "train/grad_norm": avg_grad_norm,
                "train/lr": lrs[0],
                "train/epoch": epoch,
                "train/samples_per_sec": samples_per_sec,
                "curriculum/K": K,
                "curriculum/p": p,
                "data/question_discard_rate": self.train_dataset.discard_rate,
            }
            log_dict["stitching/depth"] = self.stitching_depth
            # Log averaged gradient decomposition metrics
            if log_grad_decomp_steps > 0:
                thought_mean = log_thought_mean / log_grad_decomp_steps
                answer_mean = log_answer_mean / log_grad_decomp_steps
                thought_total = log_thought_total / log_grad_decomp_steps
                answer_total = log_answer_total / log_grad_decomp_steps
                log_dict["grad/thought_mean_norm"] = thought_mean
                log_dict["grad/answer_mean_norm"] = answer_mean
                denom_mean = thought_mean + answer_mean
                if denom_mean > 0:
                    log_dict["grad/thought_pct_per_pos"] = thought_mean / denom_mean
                denom_total = thought_total + answer_total
                if denom_total > 0:
                    log_dict["grad/thought_pct_total"] = thought_total / denom_total
                sensitivity = log_sensitivity / log_grad_decomp_steps
                log_dict["thought_sensitivity"] = sensitivity
                if answer_mean > 0 and K > 0:
                    log_dict["grad/thought_sensitivity_vs_answer"] = (
                        sensitivity / (answer_mean * K)
                    )
                log_dict["thought/token_cos_sim"] = log_thought_token_sim / log_grad_decomp_steps
            if self.use_xla:
                import torch_xla.debug.metrics as met
                compile_data = met.metric_data('CompileTime')
                if compile_data:
                    # compile_data = (count, total_time_ns)
                    total_count = compile_data[0]
                    total_time_ns = compile_data[1]
                    delta_count = total_count - self._prev_compile_count
                    delta_time_s = (total_time_ns - self._prev_compile_time_ns) / 1e9
                    self._prev_compile_count = total_count
                    self._prev_compile_time_ns = total_time_ns
                    log_dict["xla/compile_count"] = total_count
                    log_dict["xla/compile_time_s"] = total_time_ns / 1e9
                    log_dict["xla/compile_count_delta"] = delta_count
                    log_dict["xla/compile_time_delta_s"] = delta_time_s
                    print(f"  [XLA] +{delta_count} compilations ({delta_time_s:.1f}s), "
                          f"total={total_count} ({total_time_ns/1e9:.1f}s)")
            if torch.cuda.is_available():
                log_dict["system/gpu_memory_allocated_gb"] = (
                    torch.cuda.max_memory_allocated(self.device) / 1e9
                )
                log_dict["system/gpu_memory_reserved_gb"] = (
                    torch.cuda.max_memory_reserved(self.device) / 1e9
                )
                torch.cuda.reset_peak_memory_stats(self.device)
            if wandb is not None and wandb.run is not None:
                wandb.log(log_dict, step=self.optimizer_step)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", K=K, p=f"{p:.3f}", epoch=epoch)
            print(f"[logging] {time.perf_counter()-t_log:.2f}s", flush=True)

            # Print sample thought tokens from first example in last batch
            if self.tokenizer is not None and self._last_thought_token_ids is not None:
                try:
                    thought_toks = self.tokenizer.decode(
                        self._last_thought_token_ids[0], skip_special_tokens=False
                    )
                    print(f"\n  [sample] thoughts: {thought_toks[:200]}")
                except Exception as e:
                    print(f"\n  [sample] decode error: {e}")

            log_loss = 0.0
            log_grad_norm = 0.0
            log_steps = 0
            log_thought_mean = 0.0
            log_answer_mean = 0.0
            log_thought_total = 0.0
            log_answer_total = 0.0
            log_sensitivity = 0.0
            log_thought_token_sim = 0.0
            log_grad_decomp_steps = 0
            step_start_time = time.monotonic()

        # Evaluation
        if self.optimizer_step % eval_every == 0 and self.evaluator is not None:
            self._run_eval(K, p)
            self.model.train()

        # Checkpointing
        if self.optimizer_step % save_every == 0:
            self._save_checkpoint()

        pbar.update(1)

        return (accum_loss, log_loss, log_grad_norm, log_steps,
                log_thought_mean, log_answer_mean, log_thought_total,
                log_answer_total, log_sensitivity, log_thought_token_sim,
                log_grad_decomp_steps,
                step_start_time)

    def _run_eval(self, K, p):
        """Run evaluation on configured benchmarks."""
        if self.evaluator is None:
            return

        self.model.eval()
        results = self.evaluator.evaluate(self.model, K=K, p=p)

        if self.is_main:
            print(f"\n[Step {self.optimizer_step}] Eval results: {results}")
            if wandb is not None and wandb.run is not None:
                wandb.log(
                    {f"eval/{k}": v for k, v in results.items()},
                    step=self.optimizer_step,
                )

    # ---- Checkpointing (backend-aware) ----

    def _save_checkpoint(self):
        """Save checkpoint (full state dict on rank 0)."""
        ckpt_path = self.save_dir / f"step_{self.optimizer_step}"

        if self.use_xla:
            self._save_checkpoint_xla(ckpt_path)
        else:
            self._save_checkpoint_cuda(ckpt_path)

        if self.is_main:
            print(f"Saved checkpoint to {ckpt_path}")
            self._cleanup_checkpoints()

    def _save_checkpoint_cuda(self, ckpt_path: Path):
        """Save checkpoint for CUDA backend."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )

        if self.is_distributed and isinstance(self.model, FSDP):
            full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_cfg):
                model_state = self.model.state_dict()
                if self.is_main:
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model_state,
                            "optimizer_state_dict": None,
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "optimizer_step": self.optimizer_step,
                            "K": self.K,
                            "p": self.curriculum.get_p(self.optimizer_step),
                        },
                        ckpt_path / "checkpoint.pt",
                    )
        elif self.is_main:
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "optimizer_step": self.optimizer_step,
                    "K": self.K,
                    "p": self.curriculum.get_p(self.optimizer_step),
                },
                ckpt_path / "checkpoint.pt",
            )

    def _save_checkpoint_xla(self, ckpt_path: Path):
        """Save checkpoint for XLA/TPU backend (no FSDP, plain model)."""
        import torch_xla.core.xla_model as xm

        model_state = self.model.state_dict()

        if self.is_main:
            ckpt_path.mkdir(parents=True, exist_ok=True)

        # xm.save handles serialization across XLA devices
        ckpt_data = {
            "model_state_dict": model_state,
            "optimizer_state_dict": None,
            "scheduler_state_dict": self.scheduler.state_dict(),
            "optimizer_step": self.optimizer_step,
            "K": self.K,
            "p": self.curriculum.get_p(self.optimizer_step),
        }
        xm.save(ckpt_data, str(ckpt_path / "checkpoint.pt"), master_only=True)

    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints."""
        keep_n = self.config.checkpointing.keep_last_n
        ckpts = sorted(
            self.save_dir.glob("step_*"),
            key=lambda p: int(p.name.split("_")[1]),
        )
        for ckpt in ckpts[:-keep_n]:
            shutil.rmtree(ckpt)
