import math

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        if step >= self.total_steps:
            return 0
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.max_val * 0.5 * (1 + math.cos(math.pi * progress))
