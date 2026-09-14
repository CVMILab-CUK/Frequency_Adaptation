"""Frequency-Adaptation trainer: single-GPU, bf16, adapter-only optimisation.

Differences from the original `model_trainer.py`, all of them consequences of
bugs found while auditing it:

  * the conditioning image is produced by the shared `datalibs.frequency`
    filter and arrives already in [-1, 1], so train and eval see the same
    signal (they previously saw `np.abs` vs `.real` of the same transform --
    correlation 0.03);
  * DREAM is off by default because `compute_dream_and_update_latents` cannot
    forward `cross_attention_kwargs`, so its correction step runs an
    *unconditioned* UNet and biases the target;
  * checkpoints hold the ~0.1 M adapter tensors, not a 3.4 GB frozen UNet;
  * EMA shadows the adapter only;
  * captions come from the dataset instead of one hard-coded prompt, which is
    what makes CLIP-score measurable.
"""
from typing import List, Optional
import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr

from datalibs.frequency import make_condition_torch, to_model_range
from models.models import Model
from models.pipelines import SAAdapterPipeline
from trainer.base_trainer import BaseTrainer

# Cutoffs the validation curve is always measured at. Fixed for the life of the
# project so curves from different runs are directly comparable.
VAL_CUTOFFS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


class Trainer(BaseTrainer):
    def __init__(self, config_file_path):
        self.config = OmegaConf.load(config_file_path)
        c = self.config
        super().__init__(c.trainer.ckpt_dir, c.trainer.log_dir,
                         c.trainer.batch_size, c.trainer.num_workers)

        self.startEpoch = 0
        self.global_step = 0
        self.run_name = c.model.name
        self.save_dir = os.path.join(self.ckpt_dir, self.run_name)
        self.__checkDirectory__()
        self.set_seed(int(c.trainer.seed))

        self.mse_loss = nn.MSELoss(reduction="none")
        self.ddp = False
        self.ddp_dataset = False
        self.device = 0

    # ------------------------------------------------------------------ setup
    def model_define(self, gpu, ddp=False):
        c = self.config
        ds = None
        if bool(getattr(c.trainer, "use_deepspeed", False)):
            ds = OmegaConf.to_container(c.trainer.deep_speed, resolve=True)

        self.model = Model(
            model_id=c.model.model_id,
            report_to=c.model.report_to,
            output_dir=c.trainer.ckpt_dir,
            logging_dir=c.trainer.log_dir,
            mixed_precision=c.model.mixed_precision,
            cfg_scale=c.model.cfg_scale,
            gradient_accumulation_steps=c.model.gradient_accumulation_steps,
            sa_plugin=c.model.sa_adapter,
            learning_rate=c.optimizer.learning_rate,
            beta1=c.optimizer.beta1,
            beta2=c.optimizer.beta2,
            revision=c.model.revision,
            variant=c.model.variant,
            non_ema_revision=c.model.non_ema_revision,
            use_ema=c.model.use_ema,
            foreach_ema=c.model.foreach_ema,
            ds_plugin=ds,
        )

        # The frozen towers run in bf16: they are inference-only here, and bf16
        # halves their footprint without fp16's overflow behaviour.
        self.weight_dtype = torch.bfloat16 if c.model.mixed_precision == "bf16" else torch.float16
        self.model.vae.to(gpu, dtype=self.weight_dtype)
        self.model.text_encoder.to(gpu, dtype=self.weight_dtype)
        self.model.unet.to(gpu)

        if not bool(getattr(c.trainer, "gradient_checkpointing", True)):
            self.model.unet.disable_gradient_checkpointing()

        if self.model.cond_encoder is not None:
            self.model.cond_encoder.to(gpu)

        if getattr(self.model, "skip_injector", None) is not None:
            self.model.skip_injector.to(gpu)

        if c.model.use_ema and self.model.ema_unet is not None:
            self.model.ema_unet.to(gpu)

    # ------------------------------------------------------------- conditioning
    def encode_condition(self, filtered_img):
        """[-1, 1] structure image -> the spatial features the adapter reads.

        With `cond_encoder: conv` this is a trainable stem and therefore must
        stay inside the autograd graph; the VAE path is frozen and does not.
        """
        if not getattr(self.model, "attn_inject", True):
            return None          # skip-only: nothing reads an adapter latent
        x = filtered_img.to(dtype=self.weight_dtype)
        if self.model.cond_encoder is not None:
            return self.model.cond_encoder(x)
        z = self.model.vae.encode(x).latent_dist.sample()
        return z * self.model.vae.config.scaling_factor

    def skip_residuals(self, cond_image):
        """ControlNet-style residuals from the raw condition image, or None."""
        inj = getattr(self.model, "skip_injector", None)
        if inj is None:
            return None
        downs, mid = inj(cond_image)
        return {"down_block_additional_residuals": downs,
                "mid_block_additional_residual": mid}

    def encode_text(self, captions):
        tok = self.model.clip_tokenizer(
            list(captions),
            max_length=self.model.clip_tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        with torch.no_grad():
            return self.model.text_encoder(tok)[0].to(dtype=self.weight_dtype)

    # ------------------------------------------------------------------ train
    def train(self):
        self._train(gpu=0, size=1)

    def _train(self, gpu, size):
        c = self.config
        self.device = gpu
        self.initialize(gpu, size)
        self.model_define(gpu, ddp=self.ddp)

        rate_range = tuple(getattr(c.datasets, "rate_range", (0.0, 1.0)))
        self.makeDatasets(
            c.datasets.data_path, c.datasets.img_path,
            frequency_rate=c.datasets.frequency_rate,
            mode=c.datasets.frequency_mode,
            img_size=int(c.datasets.img_size),
            source_size=getattr(c.datasets, "source_size", None),
            rate_range=rate_range,
            rate_sampler=str(getattr(c.datasets, "rate_sampler", "uniform")),
            frequency_img=c.datasets.frequency_img,
            mean=0.5, std=0.5,
            caption_path=getattr(c.datasets, "caption_path", None),
            ddp=self.ddp_dataset,
        )

        accel = self.model.accelerator
        gas = int(c.model.gradient_accumulation_steps)
        steps_per_epoch = int(np.ceil(len(self.loader_train) / gas))
        max_train_steps = steps_per_epoch * int(c.trainer.total_epoch)

        self.model.lr_scheduler = get_scheduler(
            c.trainer.lr_scheduler,
            optimizer=self.model.optimizer,
            num_warmup_steps=int(c.trainer.lr_warmup_steps),
            num_training_steps=max_train_steps,
        )
        self.model.unet, self.model.optimizer, self.loader_train, self.model.lr_scheduler = accel.prepare(
            self.model.unet, self.model.optimizer, self.loader_train, self.model.lr_scheduler)

        self._maybe_resume()

        accel.init_trackers(f"{self.run_name}")
        self._dump_run_manifest(max_train_steps, rate_range)

        # mininterval keeps tqdm from writing a line per step into the log file;
        # the run is ~24k steps and the redirected bar dominated the log otherwise.
        progress_bar = tqdm(range(0, max_train_steps), initial=self.global_step,
                            desc=self.run_name, mininterval=30.0,
                            disable=not accel.is_local_main_process)

        train_loss, t_last, imgs_seen = 0.0, time.time(), 0
        for epoch in range(self.startEpoch, int(c.trainer.total_epoch)):
            self.epoch = epoch
            if self.global_step >= max_train_steps:
                break
            for data in self.loader_train:
                self.model.unet.train()
                with accel.accumulate(self.model.unet):
                    img = data["gt"].to(gpu, dtype=self.weight_dtype)
                    filtered_img = data["filtered_image"].to(gpu, dtype=self.weight_dtype)

                    with torch.no_grad():
                        latents = self.model.vae.encode(img).latent_dist.sample()
                        latents = latents * self.model.vae.config.scaling_factor
                        encoder_hidden_states = self.encode_text(data["caption"])
                    # outside no_grad: a conv condition encoder is trainable
                    cond = self.encode_condition(filtered_img)

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, self.model.noise_scheduler.config.num_train_timesteps,
                        (bsz,), device=latents.device).long()
                    noisy_latents = self.model.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Classifier-free guidance dropout on the *text* stream.
                    if self.model.cfg_scale > 0.0:
                        drop = (torch.rand(bsz, device=encoder_hidden_states.device)
                                < self.model.cfg_scale).view(bsz, 1, 1)
                        encoder_hidden_states = torch.where(
                            drop, torch.zeros_like(encoder_hidden_states), encoder_hidden_states)

                    # Independent dropout on the *structure* stream, so the model
                    # can be guided on the condition at inference time too.
                    cond_drop_p = float(getattr(c.model, "cond_dropout", 0.0))
                    if cond_drop_p > 0.0 and cond is not None:
                        cdrop = (torch.rand(bsz, device=cond.device) < cond_drop_p).view(bsz, 1, 1, 1)
                        cond = torch.where(cdrop, torch.zeros_like(cond), cond)

                    pred_type = self.model.noise_scheduler.config.prediction_type
                    if pred_type == "epsilon":
                        target = noise
                    elif pred_type == "v_prediction":
                        target = self.model.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {pred_type}")

                    extra = self.skip_residuals(filtered_img) or {}
                    model_pred = self.model.unet(
                        noisy_latents, timesteps, encoder_hidden_states,
                        return_dict=False,
                        cross_attention_kwargs=({"ip_hidden_states": cond} if cond is not None else None),
                        **extra,
                    )[0]

                    snr_gamma = getattr(c.trainer, "snr_gamma", None)
                    if snr_gamma is None:
                        loss = self.mse_loss(model_pred.float(), target.float()).mean()
                    else:
                        snr = compute_snr(self.model.noise_scheduler, timesteps)
                        w = torch.stack(
                            [snr, float(snr_gamma) * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                        w = w / snr if pred_type == "epsilon" else w / (snr + 1)
                        loss = self.mse_loss(model_pred.float(), target.float())
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * w
                        loss = loss.mean()

                    train_loss += accel.gather(loss.repeat(bsz)).mean().item() / gas
                    accel.backward(loss)
                    if accel.sync_gradients:
                        accel.clip_grad_norm_(self.model.params, c.trainer.max_grad_norm)
                    self.model.optimizer.step()
                    self.model.lr_scheduler.step()
                    self.model.optimizer.zero_grad(set_to_none=True)

                imgs_seen += bsz
                if not accel.sync_gradients:
                    continue

                self.global_step += 1
                progress_bar.update(1)

                if self.global_step >= max_train_steps:
                    break

                if c.model.use_ema and self.global_step % int(c.trainer.ema_step) == 0:
                    self.model.ema_unet.step(self.model.params)

                if self.global_step % int(c.trainer.log_interval) == 0:
                    dt = time.time() - t_last
                    ips = imgs_seen / dt if dt > 0 else 0.0
                    accel.log({"train/loss": train_loss,
                               "train/lr": self.model.lr_scheduler.get_last_lr()[0],
                               "perf/images_per_sec": ips,
                               "perf/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9},
                              step=self.global_step)
                    progress_bar.set_postfix(loss=f"{train_loss:.4f}", ips=f"{ips:.1f}")
                    train_loss, t_last, imgs_seen = 0.0, time.time(), 0

                if self.global_step % int(c.trainer.val_interval) == 0 and accel.is_main_process:
                    self.save_checkpoint()
                    self.validate()
                    gc.collect(); torch.cuda.empty_cache()

        if accel.is_main_process:
            self.save_checkpoint(final=True)
            self.validate()
        accel.wait_for_everyone()

    # ------------------------------------------------------------- checkpoints
    def save_checkpoint(self, final=False):
        tag = "final" if final else f"step-{self.global_step}"
        path = os.path.join(self.save_dir, tag, "adapter.safetensors")
        self.model.save_adapter(path)
        meta = {"global_step": self.global_step, "epoch": getattr(self, "epoch", 0)}
        with open(os.path.join(self.save_dir, tag, "meta.json"), "w") as f:
            json.dump(meta, f)
        # Optimizer/scheduler/EMA state so a multi-hour run survives a crash.
        # Small enough (adapter-sized) that writing it every validation is free.
        torch.save({
            "optimizer": self.model.optimizer.state_dict(),
            "lr_scheduler": self.model.lr_scheduler.state_dict(),
            "ema": self.model.ema_unet.state_dict() if self.model.ema_unet is not None else None,
            "global_step": self.global_step,
            "epoch": getattr(self, "epoch", 0),
        }, os.path.join(self.save_dir, tag, "train_state.pt"))
        with open(os.path.join(self.save_dir, "LATEST"), "w") as f:
            f.write(tag)
        if self.config.model.use_ema and self.model.ema_unet is not None:
            self.model.ema_unet.store(self.model.params)
            self.model.ema_unet.copy_to(self.model.params)
            self.model.save_adapter(os.path.join(self.save_dir, tag, "adapter_ema.safetensors"))
            self.model.ema_unet.restore(self.model.params)
        return path

    def build_pipeline(self):
        pipe = SAAdapterPipeline(
            vae=self.model.vae,
            text_encoder=self.model.text_encoder,
            tokenizer=self.model.clip_tokenizer,
            unet=self.model.accelerator.unwrap_model(self.model.unet),
            scheduler=self.model.noise_scheduler,
        )
        pipe.to(self.model.accelerator.device)
        pipe.set_progress_bar_config(disable=True)
        # carry the trained condition encoder onto the pipeline
        pipe.cond_encoder = self.model.cond_encoder
        # the skip injector is not a diffusers module either; None is a no-op
        pipe.skip_injector = getattr(self.model, "skip_injector", None)
        return pipe

    # ------------------------------------------------------------- validation
    @torch.no_grad()
    def validate(self, cutoffs: Optional[List[float]] = None, n_images: int = 4,
                 num_inference_steps: int = 30, guidance_scale: float = 7.5):
        """Sample the operating curve: same images, same seed, sweeping cutoff.

        Logs a strip per cutoff (GT | condition | generation) plus the
        reconstruction distance vs cutoff. These are heuristic in-loop numbers
        for monitoring; the reported metrics come from `eval/run_eval.py`.
        """
        c = self.config
        cutoffs = list(cutoffs or VAL_CUTOFFS)
        self.model.unet.eval()
        pipe = self.build_pipeline()

        batch = next(iter(self.loader_valid))
        gt = batch["gt"][:n_images].to(self.device)                 # [-1, 1]
        captions = list(batch["caption"][:n_images])
        gt_unit = torch.clamp((gt.float() + 1.0) / 2.0, 0.0, 1.0)   # [0, 1]

        try:
            import lpips as _lpips
            if not hasattr(self, "_lpips_fn"):
                self._lpips_fn = _lpips.LPIPS(net="alex").to(self.device).eval()
        except Exception:
            self._lpips_fn = None

        rows, curve = [], {}
        for cut in cutoffs:
            cond_unit = make_condition_torch(
                gt_unit, cut, frequency_img=c.datasets.frequency_img,
                mode=c.datasets.frequency_mode)          # [0, 1], same chain as training
            cond = to_model_range(cond_unit).to(dtype=self.weight_dtype)

            g = torch.Generator(device="cpu").manual_seed(int(c.trainer.seed))
            out = pipe(captions, cond, height=int(c.datasets.img_size),
                       width=int(c.datasets.img_size),
                       num_inference_steps=num_inference_steps,
                       num_images_per_prompt=1, guidance_scale=guidance_scale,
                       generator=g, output_type="pt").images.float().clamp(0, 1)

            curve[f"valid/l1_at_r{cut}"] = (out - gt_unit).abs().mean().item()
            if self._lpips_fn is not None:
                curve[f"valid/lpips_at_r{cut}"] = self._lpips_fn(
                    out * 2 - 1, gt_unit * 2 - 1).mean().item()

            # Structure consistency is the metric that answers the question the
            # run is actually asking: does the output obey the condition it was
            # given? LPIPS-to-ground-truth cannot tell "ignored the condition"
            # apart from "obeyed a condition that carries little information",
            # which is exactly the confusion a flat LPIPS curve creates.
            from eval.metrics import structure_consistency
            curve[f"valid/sc_at_r{cut}"] = float(structure_consistency(
                out, cond_unit, cut,
                frequency_img=c.datasets.frequency_img).mean())

            rows.append(torch.cat([gt_unit.cpu(), cond_unit.cpu(), out.cpu()], dim=0))
            torch.cuda.empty_cache()

        accel = self.model.accelerator
        accel.log(curve, step=self.global_step)
        if accel.trackers and accel.trackers[0].name == "tensorboard":
            grid = make_grid(torch.cat(rows, dim=0), nrow=n_images, padding=2, pad_value=1.0)
            accel.trackers[0].writer.add_image("valid/cutoff_sweep", grid, self.global_step)

        with open(os.path.join(self.log_dir, f"{self.run_name}_valcurve.jsonl"), "a") as f:
            f.write(json.dumps({"step": self.global_step, **curve}) + "\n")

        del pipe
        gc.collect(); torch.cuda.empty_cache()
        self.model.unet.train()
        return curve

    def _maybe_resume(self):
        """Pick up from the newest checkpoint if one exists for this run name.

        Resuming is opt-out rather than opt-in: relaunching after a crash should
        continue, not silently restart and quietly discard hours of compute.
        Set `trainer.resume: false` to force a fresh run.
        """
        if not bool(getattr(self.config.trainer, "resume", True)):
            return
        latest = os.path.join(self.save_dir, "LATEST")
        if not os.path.exists(latest):
            return
        tag = open(latest).read().strip()
        d = os.path.join(self.save_dir, tag)
        ck, st = os.path.join(d, "adapter.safetensors"), os.path.join(d, "train_state.pt")
        if not (os.path.exists(ck) and os.path.exists(st)):
            print(f"[resume] {d} is incomplete; starting fresh")
            return
        self.model.load_adapter(ck)
        state = torch.load(st, map_location="cpu", weights_only=False)
        self.model.optimizer.load_state_dict(state["optimizer"])
        self.model.lr_scheduler.load_state_dict(state["lr_scheduler"])
        if state.get("ema") is not None and self.model.ema_unet is not None:
            self.model.ema_unet.load_state_dict(state["ema"])
            # load_state_dict replaces the shadow tensors with the CPU copies
            # that were saved; they must go back to the training device or the
            # next EMA step mixes cuda and cpu tensors.
            self.model.ema_unet.to(self.device)
        self.global_step = int(state["global_step"])
        # Resume into the epoch the checkpoint stopped in, not epoch 0: E1
        # restored global_step but restarted the epoch loop, so it ran 26,420
        # steps instead of the planned 24,420.
        self.startEpoch = int(state.get("epoch", 0))
        self._resumed_at_step = self.global_step
        print(f"[resume] continuing {self.run_name} from {tag} "
              f"(step {self.global_step}, epoch {self.startEpoch})")

    # ------------------------------------------------------------------- misc
    def _dump_run_manifest(self, max_train_steps, rate_range):
        """Freeze exactly what produced this run, so every number is traceable."""
        n_train = len(self.train_dataset)
        manifest = {
            "run_name": self.run_name,
            "config": OmegaConf.to_container(self.config, resolve=True),
            "n_train": n_train,
            "n_valid": len(self.valid_dataset),
            "n_test": len(self.test_dataset),
            "max_train_steps": max_train_steps,
            "rate_range": list(rate_range),
            "trainable_params": int(sum(p.numel() for p in self.model.params)),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "run_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(json.dumps({k: manifest[k] for k in
                          ("run_name", "n_train", "max_train_steps",
                           "trainable_params", "gpu")}, indent=2))
