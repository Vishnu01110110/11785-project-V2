import torch
import numpy as np


class SimpleDDPM:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085,
                 beta_end: float = 0.0120):
        self.beta_values = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alpha_values = 1.0 - self.beta_values
        self.alpha_prod = torch.cumprod(self.alpha_values, dim=0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(num_training_steps)[::-1].astype(int))

    def configure_inference_steps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        inference_indices = np.arange(num_inference_steps) * step_ratio
        self.timesteps = torch.from_numpy(inference_indices[::-1].astype(np.int64))

    def calculate_previous_step(self, timestep: int) -> int:
        previous_step = timestep - (self.num_train_timesteps // self.num_inference_steps)
        return previous_step

    def compute_step_variance(self, timestep: int) -> torch.Tensor:
        earlier_timestep = self.calculate_previous_step(timestep)
        alpha_at_t = self.alpha_prod[timestep]
        alpha_before_t = self.alpha_prod[earlier_timestep] if earlier_timestep >= 0 else self.one
        beta_ratio = 1 - alpha_at_t / alpha_before_t
        var_calc = (1 - alpha_before_t) / (1 - alpha_at_t) * beta_ratio
        variance_safe = torch.clamp(var_calc, min=1e-20)
        return variance_safe

    def adjust_inference_strength(self, strength=1):
        adjusted_step_start = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[adjusted_step_start:]
        self.start_step = adjusted_step_start

    def perform_sampling_step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        current_t = timestep
        earlier_t = self.calculate_previous_step(current_t)
        alpha_at_t = self.alpha_prod[current_t]
        alpha_before_t = self.alpha_prod[earlier_t] if earlier_t >= 0 else self.one
        beta_at_t = 1 - alpha_at_t
        beta_before_t = 1 - alpha_before_t
        alpha_ratio = alpha_at_t / alpha_before_t
        beta_difference = 1 - alpha_ratio

        original_sample = (latents - torch.sqrt(beta_at_t) * model_output) / torch.sqrt(alpha_at_t)
        previous_sample_multiplier = (torch.sqrt(alpha_before_t) * beta_difference) / beta_at_t
        current_sample_multiplier = torch.sqrt(alpha_ratio) * beta_before_t / beta_at_t
        previous_sample = previous_sample_multiplier * original_sample + current_sample_multiplier * latents

        # Adding random noise based on calculated variance
        noise_term = 0
        if current_t > 0:
            noise = torch.randn(model_output.shape, generator=self.generator, device=model_output.device,
                                dtype=model_output.dtype)
            variance = torch.sqrt(self.compute_step_variance(current_t)) * noise
            previous_sample += variance

        return previous_sample

    def inject_noise_into_samples(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alphas_cumprod_device = self.alpha_prod.to(device=original_samples.device, dtype=original_samples.dtype)
        adjusted_timesteps = timesteps.to(original_samples.device)

        alpha_root = torch.sqrt(alphas_cumprod_device[adjusted_timesteps])
        while len(alpha_root.shape) < len(original_samples.shape):
            alpha_root = alpha_root.unsqueeze(-1)

        root_one_minus_alpha = torch.sqrt(1 - alphas_cumprod_device[adjusted_timesteps])
        while len(root_one_minus_alpha.shape) < len(original_samples.shape):
            root_one_minus_alpha = root_one_minus_alpha.unsqueeze(-1)

        random_noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device,
                                   dtype=original_samples.dtype)
        noisy_output = alpha_root * original_samples + root_one_minus_alpha * random_noise
        return noisy_output
