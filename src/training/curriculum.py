class CurriculumScheduler:
    """Manages p annealing (linear) over training."""

    def __init__(self, config):
        self.p_start = config.latent.p_start
        self.p_end = config.latent.p_end
        self.p_anneal_steps = config.latent.p_anneal_steps

    def get_p(self, step: int) -> float:
        """Get p value for current step (linear annealing)."""
        if step >= self.p_anneal_steps:
            return self.p_end
        t = step / self.p_anneal_steps
        return self.p_start + (self.p_end - self.p_start) * t
