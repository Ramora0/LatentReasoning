class CurriculumScheduler:
    """Manages K scheduling (stepwise) and p annealing (linear)."""

    def __init__(self, config):
        self.k_schedule = sorted(config.latent.k_schedule, key=lambda x: x[0])
        self.p_start = config.latent.p_start
        self.p_end = config.latent.p_end
        self.p_anneal_steps = config.latent.p_anneal_steps

    def get_K(self, step: int) -> int:
        """Get K value for current step (stepwise schedule)."""
        K = self.k_schedule[0][1]
        for threshold, value in self.k_schedule:
            if step >= threshold:
                K = value
            else:
                break
        return K

    def get_p(self, step: int) -> float:
        """Get p value for current step (linear annealing)."""
        if step >= self.p_anneal_steps:
            return self.p_end
        t = step / self.p_anneal_steps
        return self.p_start + (self.p_end - self.p_start) * t
