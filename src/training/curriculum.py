class CurriculumScheduler:
    """Manages p annealing (linear) and question visibility annealing over training."""

    def __init__(self, config, max_optimizer_steps: int):
        self.p_start = config.latent.p_start
        self.p_end = config.latent.p_end
        self.p_anneal_steps = int(config.latent.p_anneal_ratio * max_optimizer_steps)
        self.max_optimizer_steps = max_optimizer_steps
        self.q_anneal_start_step = int(
            getattr(config.latent, "question_anneal_start", 1.0) * max_optimizer_steps
        )
        self.q_anneal_end_step = int(
            getattr(config.latent, "question_anneal_end", 1.0) * max_optimizer_steps
        )

    def get_p(self, step: int) -> float:
        """Get p value for current optimizer step (linear annealing)."""
        if step >= self.p_anneal_steps:
            return self.p_end
        t = step / self.p_anneal_steps
        return self.p_start + (self.p_end - self.p_start) * t

    def get_q_visibility(self, step: int) -> float:
        """Get question visibility probability for current optimizer step.

        Returns 0.0 before annealing starts, ramps linearly to 1.0 at anneal end,
        then stays at 1.0.
        """
        if step < self.q_anneal_start_step:
            return 0.0
        if step >= self.q_anneal_end_step:
            return 1.0
        return (step - self.q_anneal_start_step) / (self.q_anneal_end_step - self.q_anneal_start_step)
