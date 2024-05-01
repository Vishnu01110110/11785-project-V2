import torch
import numpy as np
from tqdm import tqdm
from ddpm import SimpleDDPM

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def random_seed_generator(
        prompt,
        uncond_prompt=None,
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("Strength parameter must be within the range 0 to 1")

        transfer_to_idle = lambda x: x.to(idle_device) if idle_device else x

        seed_generator = torch.Generator(device=device)
        if seed is None:
            seed_generator.seed()
        else:
            seed_generator.manual_seed(seed)

        clip_model = models["clip"]
        clip_model.to(device)

        if do_cfg:
            encoded_prompt = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            encoded_prompt = torch.tensor(encoded_prompt, dtype=torch.long, device=device)
            contextual_embeddings = clip_model(encoded_prompt)

            encoded_uncond_prompt = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            encoded_uncond_prompt = torch.tensor(encoded_uncond_prompt, dtype=torch.long, device=device)
            unconditioned_embeddings = clip_model(encoded_uncond_prompt)

            embedding_context = torch.cat([contextual_embeddings, unconditioned_embeddings])
        else:
            simple_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            simple_tokens = torch.tensor(simple_tokens, dtype=torch.long, device=device)
            embedding_context = clip_model(simple_tokens)
        transfer_to_idle(clip_model)

        if sampler_name == "ddpm":
            sampler = SimpleDDPM(seed_generator)
            sampler.configure_inference_steps(n_inference_steps)
        else:
            raise ValueError("Unsupported sampler: %s" % sampler_name)

        shape_of_latents = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            img_encoder = models["encoder"]
            img_encoder.to(device)

            resized_input = input_image.resize((WIDTH, HEIGHT))
            input_tensor = np.array(resized_input)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32, device=device)
            input_tensor = rescale(input_tensor, (0, 255), (-1, 1))
            input_tensor = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            latent_noise = torch.randn(shape_of_latents, generator=seed_generator, device=device)
            latent_embeddings = img_encoder(input_tensor, latent_noise)

            sampler.adjust_inference_strength(strength)
            latent_embeddings = sampler.inject_noise_into_samples(latent_embeddings, sampler.timesteps[0])

            transfer_to_idle(img_encoder)
        else:
            latent_embeddings = torch.randn(shape_of_latents, generator=seed_generator, device=device)

        diffusion_model = models["diffusion"]
        diffusion_model.to(device)

        process_steps = tqdm(sampler.timesteps)
        for step, current_step in enumerate(process_steps):
            time_vector = get_time_embedding(current_step).to(device)
            diffusion_input = latent_embeddings

            if do_cfg:
                diffusion_input = diffusion_input.repeat(2, 1, 1, 1)

            diffusion_output = diffusion_model(diffusion_input, embedding_context, time_vector)

            if do_cfg:
                split_cond, split_uncond = diffusion_output.chunk(2)
                diffusion_output = cfg_scale * (split_cond - split_uncond) + split_uncond

            latent_embeddings = sampler.perform_sampling_step(current_step, latent_embeddings, diffusion_output)

        transfer_to_idle(diffusion_model)

        output_decoder = models["decoder"]
        output_decoder.to(device)
        final_images = output_decoder(latent_embeddings)
        transfer_to_idle(output_decoder)

        final_images = rescale(final_images, (-1, 1), (0, 255), clamp=True)
        final_images = final_images.permute(0, 2, 3, 1)
        final_images = final_images.to("cpu", torch.uint8).numpy()
        return final_images[0]


def rescale(tensor, original_range, target_range, clamp=False):
    min_original, max_original = original_range
    min_target, max_target = target_range
    tensor = (tensor - min_original) * (max_target - min_target) / (max_original - min_original) + min_target
    if clamp:
        tensor = torch.clamp(tensor, min_target, max_target)
    return tensor


def get_time_embedding(time_step):
    powers_of_ten = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)
    time_tensor = torch.tensor([time_step], dtype=torch.float32)[:, None] * powers_of_ten[None]
    return torch.cat([torch.cos(time_tensor), torch.sin(time_tensor)], dim=-1)
