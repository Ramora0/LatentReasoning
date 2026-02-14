import os
import math
import time
import shutil
import functools
from pathlib import Path

import torch
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

    def __init__(self, config, model, train_dataset, collator, evaluator=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.collator = collator
        self.evaluator = evaluator
        self.K = config.latent.K

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

        # Wrap model with FSDP if distributed
        if self.is_distributed and config.distributed.fsdp:
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

        # Checkpointing
        self.save_dir = Path(config.checkpointing.save_dir)
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        if self.is_main and wandb is not None:
            wandb.init(
                project=config.logging.wandb_project,
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

        if self.config.distributed.fsdp_activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
                checkpoint_wrapper,
                CheckpointImpl,
            )
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda m: isinstance(m, Gemma2DecoderLayer),
            )

        return model

    def _wrap_xla_fsdp(self, model):
        """Wrap model with XLA FSDP for TPU."""
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XlaFSDP

        # Move model to XLA device first
        model = model.to(self.device)

        # XLA FSDP's recursive_wrap passes different kwargs than CUDA FSDP,
        # so we use a simple callable instead of transformer_auto_wrap_policy.
        def auto_wrap_policy(module, recurse, **kwargs):
            if recurse:
                return True  # always recurse into children
            return isinstance(module, Gemma2DecoderLayer)

        model = XlaFSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            compute_dtype=torch.bfloat16,
            shard_param_on_dim_0=True,
            pin_layout_in_collective_ops=True,
        )

        # NOTE: XLA per-layer activation checkpointing (checkpoint_module) is
        # intentionally skipped. It conflicts with KV cache use inside phase2
        # (SlidingWindowCache breaks on recompute for short sequences). The model
        # already wraps the entire phase2 with torch_xla checkpoint, which
        # provides equivalent memory savings without the cache conflict.

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
            num_workers=0 if self.use_xla else 4,
            pin_memory=not self.use_xla,  # pin_memory is CUDA-only
            drop_last=True,
        )

        return loader

    # ---- Gradient clipping (backend-aware) ----

    def _clip_grad_norm(self) -> float:
        """Clip gradient norms, returning the total norm."""
        max_norm = self.config.training.max_grad_norm

        if self.use_xla:
            # XLA: use xm.optimizer_step handles sync; clip manually
            from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XlaFSDP
            if isinstance(self.model, XlaFSDP):
                grad_norm = self.model.clip_grad_norm_(max_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm
                )
            return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        else:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            if self.is_distributed and isinstance(self.model, FSDP):
                grad_norm = self.model.clip_grad_norm_(max_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm
                )
            return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    # ---- XLA step helper ----

    def _xla_step(self):
        """Execute optimizer step with XLA mark_step for graph execution."""
        self._xm.optimizer_step(self.optimizer)
        self.scheduler.step()

    # ---- Training loop ----

    def train(self):
        """Main training loop.

        Iterates for exactly num_epochs over the dataset. All schedules
        (LR, p-annealing) and logging intervals are based on optimizer
        steps, which are deterministic given (dataset_size, batch_size,
        grad_accum, world_size, num_epochs).
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
        micro_step = 0  # counts forward passes within current accumulation window
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

            for batch in epoch_loader:
                # Skip empty batches (all samples discarded by length filter)
                if batch is None:
                    continue

                # Move batch to device (MpDeviceLoader already does this for XLA)
                if not self.use_xla:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get curriculum values (based on optimizer steps for consistency)
                K = self.K
                p = self.curriculum.get_p(self.optimizer_step)

                # Forward pass
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
                    loss = outputs["loss"] / self.grad_accum_steps

                # Backward
                loss.backward()
                accum_loss += loss.item()
                micro_step += 1

                # Optimizer step every grad_accum_steps micro-batches
                if micro_step % self.grad_accum_steps == 0:
                    # Gradient clipping
                    grad_norm = self._clip_grad_norm()

                    # Optimizer step (XLA needs mark_step)
                    if self.use_xla:
                        self._xla_step()
                    else:
                        self.optimizer.step()
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.optimizer_step += 1

                    # Accumulate stats for logging window
                    log_loss += accum_loss
                    log_grad_norm += grad_norm
                    log_steps += 1
                    accum_loss = 0.0

                    # Logging (based on optimizer steps)
                    if self.is_main and self.optimizer_step % log_every == 0:
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
                        # Per-step latent gradient norms
                        base = self.model
                        while hasattr(base, "module"):
                            base = base.module
                        for step_k, norm_val in base._latent_grad_norms.items():
                            log_dict[f"latent_grads/step_{step_k}"] = norm_val
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
                        log_loss = 0.0
                        log_grad_norm = 0.0
                        log_steps = 0
                        step_start_time = time.monotonic()

                    # Evaluation
                    if self.optimizer_step % eval_every == 0 and self.evaluator is not None:
                        self._run_eval(K, p)
                        self.model.train()

                    # Checkpointing
                    if self.optimizer_step % save_every == 0:
                        self._save_checkpoint()

                    pbar.update(1)

            if self.is_main:
                print(f"\nEpoch {epoch + 1}/{self.num_epochs} complete "
                      f"(optimizer step {self.optimizer_step}/{max_steps})")

        pbar.close()

        # Final save
        self._save_checkpoint()

        if self.is_main and wandb is not None and wandb.run is not None:
            wandb.finish()

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
        """Save checkpoint for XLA/TPU backend."""
        import torch_xla.core.xla_model as xm
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XlaFSDP

        # Consolidate sharded state on host (rank 0)
        if isinstance(self.model, XlaFSDP):
            model_state = self.model.state_dict()
        else:
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
