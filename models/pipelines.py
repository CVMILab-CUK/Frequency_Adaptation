import inspect
import importlib
import warnings
import PIL
import numpy as np
from typing import Callable, List, Optional, Union

import torch
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser


from diffusers import DiffusionPipeline, LMSDiscreteScheduler, StableDiffusionMixin

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps




def _adapter_kwargs(unet, adapter_condition, cond_encoder=None):
    """The adapter latent goes to the attention processors only if one is
    installed; a skip-only model keeps stock processors, which would warn about
    and drop the kwarg on every denoising step. A trained condition encoder with
    no attention processor to read it means the pipeline is miswired: refuse."""
    from .attention_processor import StandAloneAttnProcessor
    has_sa = any(isinstance(p, StandAloneAttnProcessor) for p in unet.attn_processors.values())
    if not has_sa and cond_encoder is not None:
        raise RuntimeError("condition encoder present but no StandAlone attention processor is installed")
    return {"ip_hidden_states": adapter_condition} if has_sa else None

class ModelWrapper:
    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod

    def apply_model(self, *args, **kwargs):
        if len(args) == 3:
            encoder_hidden_states = args[-1]
            args = args[:2]
        if kwargs.get("cond", None) is not None:
            encoder_hidden_states = kwargs.pop("cond")
        return self.model(*args, encoder_hidden_states=encoder_hidden_states, **kwargs).sample


class SAAdapterPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for Structure-Aware Image Generation using SA-Adapter.
    Uses a frequency-filtered image as a structural condition.
    """
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        feature_extractor=None,
        safety_checker=None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]
            
            if isinstance(image[0], PIL.Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=PIL.Image.BICUBIC))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)
        
        image = image.to(device=device, dtype=dtype)

        # Whichever encoder the model was trained with has to be the one used
        # here; a mismatch silently feeds the adapter a distribution it never saw.
        cond_encoder = getattr(self, "cond_encoder", None)
        if cond_encoder is not None:
            image_latents = cond_encoder(image)
        elif image.shape[1] == 4:   # already latents
            image_latents = image
        else:
            image_latents = self.vae.encode(image).latent_dist.sample()
            image_latents = image_latents * self.vae.config.scaling_factor

        # Duplicate for num_images_per_prompt
        image_latents = image_latents.repeat_interleave(num_images_per_prompt, dim=0)
        
        return image_latents
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (?) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to ? in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def set_sampler(self, scheduler_type: str):
        warnings.warn("The `set_sampler` method is deprecated, please use `set_scheduler` instead.")
        return self.set_scheduler(scheduler_type)

    def set_scheduler(self, scheduler_type: str):
        library = importlib.import_module("k_diffusion")
        sampling = getattr(library, "sampling")
        self.sampler = getattr(sampling, scheduler_type)
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        if latents is None:
            # A CPU generator is what makes sampling reproducible across GPUs,
            # so honour its device and move the noise afterwards.
            gen_device = device
            if generator is not None:
                g0 = generator[0] if isinstance(generator, list) else generator
                gen_device = g0.device
            latents = torch.randn(shape, generator=generator, device=gen_device, dtype=dtype)
            latents = latents.to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        return latents
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, PIL.Image.Image] = None, # Filtered Image (Condition)
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        
        # 0. Default height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        if prompt is None and "prompt_embeds" not in kwargs:
             raise ValueError("Prompt must be provided")
        if image is None:
             raise ValueError("Structure Image (filtered_image) must be provided for SA-Adapter")

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Encode Input Prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        # 3. Prepare Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare Latents (Noise)
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare Condition Image (Structure Map -> VAE -> Latent)
        # image는 여기서 Frequency Filtered Image여야 합니다.
        adapter_condition = self.prepare_image(
            image, width, height, batch_size, num_images_per_prompt, device, prompt_embeds.dtype
        )

        # 5b. ControlNet-style skip residuals, when the run uses that placement.
        # They come from the *raw* condition image, not its VAE latent.
        skip_extra = {}
        injector = getattr(self, "skip_injector", None)
        if injector is not None:
            raw = image.to(device=device, dtype=prompt_embeds.dtype)
            downs, mid = injector(raw)
            downs = [d.repeat_interleave(num_images_per_prompt, dim=0) for d in downs]
            mid = mid.repeat_interleave(num_images_per_prompt, dim=0)
            skip_extra = {"down_block_additional_residuals": downs,
                          "mid_block_additional_residual": mid}
        
        # 6. Prepare CFG (Uncond embeddings)
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device))[0]
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            
            # Text Embeddings concat
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
            # Adapter Condition concat (Apply condition to both uncond and cond paths usually, 
            # or you can zero out for uncond if you trained with adapter dropout)
            # 여기서는 Training 코드와 맞춰서 복제해서 넣습니다.
            adapter_condition = torch.cat([adapter_condition, adapter_condition]).to(prompt_embeds.dtype)
            if skip_extra:
                skip_extra = {
                    "down_block_additional_residuals":
                        [torch.cat([d, d]) for d in skip_extra["down_block_additional_residuals"]],
                    "mid_block_additional_residual":
                        torch.cat([skip_extra["mid_block_additional_residual"]] * 2),
                }


        # 7. Denoising Loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for CFG
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).to(prompt_embeds.dtype)

                # Predict noise
                # [CORE] Pass adapter_condition to ip_hidden_states
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=_adapter_kwargs(self.unet, adapter_condition, getattr(self, "cond_encoder", None)),
                    return_dict=False,
                    **skip_extra,
                )[0]

                # Perform CFG
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        if not output_type == "latent":
            # The scheduler can hand back fp32 latents; the VAE runs in bf16.
            latents = latents.to(dtype=self.vae.dtype)
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)
            image=  torch.concat([self.image_processor.postprocess(i, output_type='pt') for i in image], dim=0)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    


class SDXL_SAAdapterPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for Structure-Aware Image Generation using SA-Adapter on SDXL.
    Uses a frequency-filtered image as a structural condition.
    """
    def __init__(
        self,
        vae,
        text_encoder,
        text_encoder_2, # [핵심 1] 두 번째 텍스트 인코더 추가
        tokenizer,
        tokenizer_2,    # [핵심 1] 두 번째 토크나이저 추가
        unet,
        scheduler,
        feature_extractor=None,
        safety_checker=None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # (prepare_image, prepare_extra_step_kwargs, prepare_latents 함수는 기존과 동일하게 유지)
    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]
            if isinstance(image[0], PIL.Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=PIL.Image.BICUBIC))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = (image - 0.5) / 0.5
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)
        
        image = image.to(device=device, dtype=dtype)
        
        if image.shape[1] == 4:
             image_latents = image
        else:
            image_latents = self.vae.encode(image).latent_dist.sample()
            image_latents = image_latents * self.vae.config.scaling_factor

        image_latents = image_latents.repeat_interleave(num_images_per_prompt, dim=0)
        return image_latents

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        if latents is None:
            # A CPU generator is what makes sampling reproducible across GPUs,
            # so honour its device and move the noise afterwards.
            gen_device = device
            if generator is not None:
                g0 = generator[0] if isinstance(generator, list) else generator
                gen_device = g0.device
            latents = torch.randn(shape, generator=generator, device=gen_device, dtype=dtype)
            latents = latents.to(device)
        else:
            latents = latents.to(device)
        return latents

    # [핵심 2] SDXL용 텍스트 인코딩 헬퍼 함수
    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        # 1. 긍정 프롬프트 인코딩
        text_inputs_1 = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_inputs_2 = self.tokenizer_2(prompt, padding="max_length", max_length=self.tokenizer_2.model_max_length, truncation=True, return_tensors="pt")
        
        out_1 = self.text_encoder(text_inputs_1.input_ids.to(device), output_hidden_states=True)
        out_2 = self.text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=True)
        
        prompt_embeds = torch.concat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1)
        pooled_prompt_embeds = out_2.text_embeds
        
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        # 2. 부정 프롬프트 (CFG) 인코딩
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            uncond_inputs_1 = self.tokenizer(uncond_tokens, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            uncond_inputs_2 = self.tokenizer_2(uncond_tokens, padding="max_length", max_length=self.tokenizer_2.model_max_length, truncation=True, return_tensors="pt")
            
            uncond_out_1 = self.text_encoder(uncond_inputs_1.input_ids.to(device), output_hidden_states=True)
            uncond_out_2 = self.text_encoder_2(uncond_inputs_2.input_ids.to(device), output_hidden_states=True)
            
            negative_prompt_embeds = torch.concat([uncond_out_1.hidden_states[-2], uncond_out_2.hidden_states[-2]], dim=-1)
            negative_pooled_prompt_embeds = uncond_out_2.text_embeds
            
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            
            # 합치기
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, PIL.Image.Image] = None, 
        height: int = 1024, # [핵심 3] SDXL 기본 해상도 1024
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0, # SDXL은 보통 5.0~7.5 사용
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pt", # pt로 기본 설정
        return_dict: bool = True,
        **kwargs,
    ):
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # 1. Encode Prompt (듀얼 인코더)
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance
        )

        # 2. Prepare Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare Latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents,
        )

        # 4. Prepare Condition Image
        adapter_condition = self.prepare_image(
            image, width, height, batch_size, num_images_per_prompt, device, prompt_embeds.dtype
        )
        if do_classifier_free_guidance:
            adapter_condition = torch.cat([adapter_condition, adapter_condition]).to(prompt_embeds.dtype)
            if skip_extra:
                skip_extra = {
                    "down_block_additional_residuals":
                        [torch.cat([d, d]) for d in skip_extra["down_block_additional_residuals"]],
                    "mid_block_additional_residual":
                        torch.cat([skip_extra["mid_block_additional_residual"]] * 2),
                }

        # [핵심 4] Time IDs (Micro-conditioning) 준비
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype, device=device)
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        # 5. Denoising Loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # [핵심 5] added_cond_kwargs 추가
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=_adapter_kwargs(self.unet, adapter_condition, getattr(self, "cond_encoder", None)), 
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **kwargs, return_dict=False)[0]

        # 6. Post-processing
        if not output_type == "latent":
            vae_dtype = self.vae.dtype
            latents = latents.to(dtype=torch.float32)
            self.vae.to(dtype=torch.float32)
            # The scheduler can hand back fp32 latents; the VAE runs in bf16.
            latents = latents.to(dtype=self.vae.dtype)
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            self.vae.to(dtype=vae_dtype)
        else:
            image = latents

        has_nsfw_concept = None

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)