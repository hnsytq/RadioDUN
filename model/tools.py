import random
import numpy as np
import torch

class SeedContextManager:
    def __init__(self, seed=None):
        self.seed = seed + 40 if seed is not None else None
        self.random_state = None
        self.np_random_state = None
        self.torch_random_state = None
        self.torch_cuda_random_state = None

    def __enter__(self):
        if self.seed is not None:
            self.random_state = random.getstate()
            self.np_random_state = np.random.get_state()
            self.torch_random_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                self.torch_cuda_random_state = torch.cuda.random.get_rng_state_all()

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed is not None:
            random.setstate(self.random_state)
            np.random.set_state(self.np_random_state)
            torch.random.set_rng_state(self.torch_random_state)
            if torch.cuda.is_available():
                torch.cuda.random.set_rng_state_all(self.torch_cuda_random_state)
