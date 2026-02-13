class CurriculumScheduler:
    """Manages p annealing (linear) over training."""

    def __init__(self, config, max_optimizer_steps: int):
        self.p_start = config.latent.p_start
        self.p_end = config.latent.p_end
        self.p_anneal_steps = int(config.latent.p_anneal_ratio * max_optimizer_steps)

    def get_p(self, step: int) -> float:
        """Get p value for current optimizer step (linear annealing)."""
        if step >= self.p_anneal_steps:
            return self.p_end
        t = step / self.p_anneal_steps
        return self.p_start + (self.p_end - self.p_start) * t
