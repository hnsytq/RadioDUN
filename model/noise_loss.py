import torch
from torch.nn.modules.loss import _Loss

class NoiseLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, pl_map, pred, noise):
        mean_loss = torch.mean(noise)
        noise_std = torch.mean(torch.square(noise - mean_loss))
        noise_std_ = torch.mean(torch.square((pl_map - pred + noise) - mean_loss))
        return noise_std + noise_std_
        # kx = pred - noise
        # noise_ = pl_map - kx = pl_map - pred + noise
    print(torch.cuda.is_available())