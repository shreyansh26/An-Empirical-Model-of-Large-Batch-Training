import torch
from utils import exp_mov_avg
import pandas as pd

class NoiseScaleCallback():
    def __init__(self, bs, N, max_steps, beta=0.99):
        self.b_small = bs
        self.b_big = bs * N

        self.N = N
        self.max_steps = max_steps
        self.beta = beta

    def initialize(self):
        self.moving_average_scale = None
        self.moving_average_noise = None

        self.batches_grad = []
        self.noise_scale_list = []

    def update(self, step, grad):
        self.batches_grad.append(grad)

        if step >= self.max_steps:
            return

        if step % self.N == self.N - 1:
            batches_grad = torch.cat(self.batches_grad, dim=1)
            self.batches_grad = []

            batches_grad_mean = batches_grad.mean(dim=1)
            g_big = (batches_grad_mean ** 2).mean()
            g_small = (grad ** 2).mean()

            noise = (self.b_big * g_big - self.b_small * g_small) / (self.b_big - self.b_small)
            scale = (g_small - g_big) / ((1 / self.b_small) - (1 / self.b_big))

            self.moving_average_scale, scale = exp_mov_avg(self.moving_average_scale, scale, self.beta, step)
            self.moving_average_noise, noise = exp_mov_avg(self.moving_average_noise, noise, self.beta, step)

            scale = scale.item()
            noise = noise.item()

            noise_scale = scale / noise

            self.noise_scale_list.append({"noise_scale": noise_scale})

    def plot(self):
        df = pd.DataFrame(self.noise_scale_list)
        ax = df.noise_scale.plot(title=f"Average Noise scale : {df.noise_scale.mean()}")
        ax.figure.savefig('plot.png')