import torch
import numpy as np

class SimpleDDPM:

    def __init__(self, gen: torch.Generator, steps=1000, start=0.00085, end=0.0120):
        self.beta = torch.linspace(start ** 0.5, end ** 0.5, steps, dtype=torch.float32) ** 2
        self.alpha = 1.0 - self.beta
        self.alpha_prod = torch.cumprod(self.alpha, dim=0)
        self.one = torch.tensor(1.0, dtype=torch.float32)

        self.gen = gen

        self.steps = steps
        self.t = torch.from_numpy(np.arange(0, steps)[::-1].copy())

    def set_steps(self, inference_steps=50):
        self.inf_steps = inference_steps
        r = self.steps // self.inf_steps
        self.t = torch.from_numpy((np.arange(0, inference_steps) * r).round()[::-1].astype(np.int64))

    def prev_t(self, t: int) -> int:
        return t - self.steps // self.inf_steps

    def variance(self, t: int) -> torch.Tensor:
        prev = self.prev_t(t)

        alpha_t = self.alpha_prod[t]
        alpha_prev = self.alpha_prod[prev] if prev >= 0 else self.one
        beta_t = 1 - alpha_t / alpha_prev

        v = (1 - alpha_prev) / (1 - alpha_t) * beta_t
        v = torch.clamp(v, min=1e-20)

        return v

    def set_strength(self, strength=1):
        start_step = self.inf_steps - int(self.inf_steps * strength)
        self.t = self.t[start_step:]

    def step(self, t: int, latents: torch.Tensor, out: torch.Tensor):
        prev = self.prev_t(t)

        alpha_t = self.alpha_prod[t]
        alpha_prev = self.alpha_prod[prev] if prev >= 0 else self.one
        beta_t = 1 - alpha_t
        beta_prev = 1 - alpha_prev
        alpha_ratio = alpha_t / alpha_prev
        beta_ratio = 1 - alpha_ratio

        pred_x0 = (latents - beta_t ** 0.5 * out) / alpha_t ** 0.5

        coeff_x0 = (alpha_prev ** 0.5 * beta_ratio) / beta_t
        coeff_xt = alpha_ratio ** 0.5 * beta_prev / beta_t

        pred_prev = coeff_x0 * pred_x0 + coeff_xt * latents

        v = 0
        if t > 0:
            n = torch.randn(out.shape, generator=self.gen, device=out.device, dtype=out.dtype)
            v = (self.variance(t) ** 0.5) * n

        return pred_prev + v

    def add_noise(self, x: torch.FloatTensor, t: torch.IntTensor) -> torch.FloatTensor:
        sqrt_alpha = self.alpha_prod[t].to(x.device, dtype=x.dtype) ** 0.5
        sqrt_one_minus_alpha = (1 - self.alpha_prod[t]).to(x.device, dtype=x.dtype) ** 0.5

        n = torch.randn_like(x, generator=self.gen)
        return sqrt_alpha * x + sqrt_one_minus_alpha * n
