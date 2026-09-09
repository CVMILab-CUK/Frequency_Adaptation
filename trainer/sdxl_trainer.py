from typing import Optional, Union, List
import gc, os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL       import Image
from tqdm.auto import tqdm

from omegaconf import OmegaConf
from safetensors.torch import save_file, load_file

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)

import torch.backends.cudnn as cudnn
# cudnn.benchmark = False
# cudnn.deterministic = True

from einops import rearrange

import torch
import torch.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch             import nn


from torchvision.utils import make_grid


from models               import  SDXL_SAAdapterPipeline, Frozen_CLIPImageNTextEmbedder as ImageClip
from models.sdxl_models   import Model
from trainer.base_trainer import BaseTrainer
from utils import GpuFixedFFT
# from libs.metric          import get_similarity_metric

from diffusers.training_utils import compute_dream_and_update_latents, compute_snr
from diffusers.optimization   import get_scheduler


from torchvision.models      import ViT_H_14_Weights, vit_h_14
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Trainer(BaseTrainer):
    def __init__(self, config_file_path, training_mode=None):
        self.config = OmegaConf.load(config_file_path)
        self.startEpoch     = 0
        self.eps            = 1e-6
        self.training_mode  = training_mode 
        super().__init__(self.config.trainer.ckpt_dir, 
                        self.config.trainer.log_dir, 
                        self.config.trainer.batch_size,  
                        self.config.trainer.num_workers)
        # Set Prompt
        self.prompt_lst = [
            "a high-quality photo of a face",
            "a high-quality studio portrait photo of a face",
            "a high-quality photo of a face, natural lighting",
            "a high-quality photo of a face, dramatic lighting",
            "a high-quality photo of a face, smiling",
            "a high-quality photo of a face, side profile"
        ]
        # fsdp.state_dict_type = "full" 
        # fsdp.FSDP.allgather_chunk_size_mb = 128

        self.__checkDirectory__()
        self.losses = {"sdsc":[], "rec":[], "sim":[],"loss":[]}
        self.accuracy = {"sdsc":[], "mse":[]}
        self.val_losses = {"sdsc":[], "rec":[], "sim":[], "loss":[]}
        self.val_accuracy = {"sdsc":[], "mse":[]}


        # Set Loss
        self.mse_loss = nn.MSELoss(reduction='none')

        # Set Metrics
        weights = ViT_H_14_Weights.DEFAULT
        self.acc_model = vit_h_14(weights=weights)
        self.acc_model_preprocess = weights.transforms()

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

        # For Distributed Training
        self.ddp = False
        self.ddp_dataset =  False

        self.untrained_dtype = torch.bfloat16
        self.trained_dtype   = torch.bfloat16



    
    def model_define(self, gpu, ddp=True):

        self.model = Model(
            model_id                    = self.config.model.model_id,
            report_to                   = self.config.model.report_to,
            output_dir                  = self.config.trainer.ckpt_dir,
            logging_dir                 = self.config.trainer.log_dir,
            # Model optimization parameters
            mixed_precision             = self.config.model.mixed_precision,
            cfg_scale                   = self.config.model.cfg_scale,
            gradient_accumulation_steps = self.config.model.gradient_accumulation_steps,
            
            # Model Setting
            sa_plugin         = self.config.model.sa_adapter,

            # Train Pipeline Setting
            learning_rate     = self.config.optimizer.learning_rate,
            beta1             = self.config.optimizer.beta1,
            beta2             = self.config.optimizer.beta2,        
            # Pretraining model setting
            revision          = self.config.model.revision,
            variant           = self.config.model.variant,#"non_ema",
            non_ema_revision  = self.config.model.non_ema_revision,
            # Mymodel setting
            use_ema           = self.config.model.use_ema,
            foreach_ema       = self.config.model.foreach_ema,
            ds_plugin         = OmegaConf.to_container(self.config.trainer.deep_speed, resolve=True),
        )
        
        #Set LDM
        self.model.vae.to(gpu, dtype=self.untrained_dtype)
        self.model.unet.to(gpu, dtype=self.trained_dtype)
        self.model.text_encoder_1.to(gpu, dtype=self.untrained_dtype)
        self.model.text_encoder_2.to(gpu, dtype=self.untrained_dtype)

        if self.config.model.use_ema:
            if self.config.model.off_load_ema:
                self.model.ema_unet.pin_memory()
            else:
                self.model.ema_unet.to(gpu)


    @torch.no_grad()
    def top_k_accuracy(self, pred_imgs, gt_imgs):
        # For Metric
        self.acc_model.to(self.device, dtype = self.untrained_dtype)
        self.acc_model = self.acc_model.eval()

        pred = self.acc_model_preprocess(pred_imgs).to(dtype = self.untrained_dtype)
        gt = self.acc_model_preprocess(gt_imgs).to(dtype = self.untrained_dtype)

        gt_class_id = self.acc_model(gt)
        gt_class_id = gt_class_id.softmax(1).argmax()

        pred_class_id = self.acc_model(pred)
        pred_class_id = pred_class_id.softmax(1).argmax(1)

        acc_mean = (pred_class_id == gt_class_id).to(dtype=self.trained_dtype).mean().item()
        self.acc_model.to("cpu")
        return acc_mean
    
    @torch.no_grad()
    def psm_accuracy(self, pred_imgs, gt_imgs):
        self.lpips.to(self.device, dtype = self.untrained_dtype)
        acc_mean = self.lpips(pred_imgs.to(dtype = self.untrained_dtype), gt_imgs.to(dtype = self.untrained_dtype)).item()
        self.lpips.to("cpu")
        return acc_mean
    
    @torch.no_grad()
    def encode_prompt(self, prompt, device):
    # 1. 두 개의 토크나이저로 각각 토크나이징
        text_inputs_1 = self.model.tokenizer_1(
            prompt, padding="max_length", max_length=self.model.tokenizer_1.model_max_length, truncation=True, return_tensors="pt"
        )
        text_inputs_2 = self.model.tokenizer_2(
            prompt, padding="max_length", max_length=self.model.tokenizer_2.model_max_length, truncation=True, return_tensors="pt"
        )

        # 2. 첫 번째 인코더 (hidden_states 추출)
        output_1 = self.model.text_encoder_1(text_inputs_1.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_1 = output_1.hidden_states[-2]

        # 3. 두 번째 인코더 (hidden_states와 pooled 추출)
        output_2 = self.model.text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_2 = output_2.hidden_states[-2]
        pooled_prompt_embeds = output_2.text_embeds # SDXL 추가 요구사항

        # 4. 두 개의 hidden_states를 마지막 차원(dim=-1) 기준으로 병합
        prompt_embeds = torch.concat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        return prompt_embeds, pooled_prompt_embeds
    
    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, device, batch_size):
        # SDXL 논문에 명시된 6개의 차원 리스트 [H_orig, W_orig, crop_y, crop_x, H_target, W_target]
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        
        # 텐서 변환 및 배치 사이즈에 맞게 복제
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        
        return add_time_ids
    
    def _train(self, gpu, size):
        print(f"Now Initialize Rank: {gpu} | Number Of GPU : {size}")
        self.device = gpu
        self.initialize(gpu, size)
        self.model_define(gpu, ddp=self.ddp) 
        self.makeDatasets(self.config.datasets.data_path, 
                          self.config.datasets.img_path,
                          mean=0.5, 
                          std=0.5, 
                          img_size=self.config.datasets.img_size,
                        #   min_value=0, 
                          ddp=self.ddp_dataset)

        # Set Scheduler
        num_warmup_steps_for_scheduler = self.config.trainer.lr_warmup_steps * self.model.accelerator.num_processes
        len_train_dataloader_after_sharding = int(np.ceil(len(self.loader_train) / self.model.accelerator.num_processes))
        num_update_steps_per_epoch = int(np.ceil(len_train_dataloader_after_sharding / self.config.model.gradient_accumulation_steps))
        num_training_steps_for_scheduler = num_update_steps_per_epoch * self.model.accelerator.num_processes * self.config.trainer.total_epoch

        # Set Varialbes
        global_step = 0
        self.text_prompt = self.prompt_lst[self.config.trainer.prompt_idx] # For Fixed Text Prompt

        
        self.model.lr_scheduler = get_scheduler(
                self.config.trainer.lr_scheduler,
                optimizer=self.model.optimizer,
                num_warmup_steps=num_warmup_steps_for_scheduler,
                num_training_steps=num_training_steps_for_scheduler,
            )

        # Model define
        self.model.unet, self.model.optimizer, self.loader_train, self.model.lr_scheduler = self.model.accelerator.prepare(
            self.model.unet, self.model.optimizer, self.loader_train, self.model.lr_scheduler
        )

        # Define Log Dir
        self.model.accelerator.init_trackers(f"{self.config.model.name}_project")

        if not os.path.exists(os.path.join(".", "log", f"{self.config.model.name}_project")):
            os.makedirs(os.path.join(".", "log", f"{self.config.model.name}_project"), exist_ok=True)
        
        # self.scaler = torch.cuda.amp.GradScaler()

        print(f"DDP RANK {gpu} RUN...")
        if gpu == 0:
            initial_global_step = 0
            max_train_steps = self.config.trainer.total_epoch * num_update_steps_per_epoch
            progress_bar = tqdm(
                range(0, max_train_steps),
                initial=initial_global_step,
                desc="Train Steps",
                # Only show the progress bar once on each machine.
                disable=not self.model.accelerator.is_local_main_process,
            )

        for epoch in range(self.startEpoch, self.config.trainer.total_epoch):
            self.epoch = epoch
            # self.train_dataset_sampler.set_epoch(epoch),
            # self.test_dataset_sampler.set_epoch(epoch)
            torch.cuda.empty_cache()
            gc.collect()

            train_loss = 0.0

            for step, data in enumerate(self.loader_train):
                self.model.unet.train()                
                self.model.text_encoder_1.train()
                self.model.text_encoder_2.train()
                # self.model.unet.to(self.device, dtype=self.trained_dtype)

                
                with self.model.accelerator.accumulate(self.model.unet): # for cumulative execution.
                    with self.model.accelerator.autocast():  # For amp
                        # Convert images to latent space
                        # Data는 이미지, 주파수 필터링된 이미지, prompt를 제공할예정
                        img = data["gt"].to(gpu, self.untrained_dtype)
                        filtered_img = data["filtered_image"].to(gpu, self.untrained_dtype)

                        latents = self.model.vae.encode(img).latent_dist.sample()  # H/16, W/16
                        control_condition_vector =self.model.vae.encode(filtered_img).latent_dist.sample() # H/16, W/16
                        
                        # Scaling the latents
                        latents = latents * self.model.vae.config.scaling_factor
                        control_condition_vector = control_condition_vector * self.model.vae.config.scaling_factor

                        # Sample Noise
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]

                        
                        timesteps = torch.randint(0, self.model.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()
                        noisy_latents = self.model.noise_scheduler.add_noise(latents, noise, timesteps)

                        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                            [self.text_prompt] * bsz, device=gpu
                        )
                        prompt_embeds = prompt_embeds
                        pooled_prompt_embeds = pooled_prompt_embeds

                        # Classifier-Free Guidance (Conditional Dropout) Implementation
                        if self.model.cfg_scale > 0.0:
                            mask = (torch.rand(bsz, device=gpu, dtype=self.trained_dtype) < self.model.cfg_scale).view(bsz, 1, 1)
                            uncond_prompt_embeds = torch.zeros_like(prompt_embeds)
                            prompt_embeds = torch.where(mask, uncond_prompt_embeds, prompt_embeds)
                            
                            # Pooled 임베딩도 동일하게 마스킹 처리 (차원이 다르므로 view 조절)
                            mask_pooled = mask.view(bsz, 1)
                            uncond_pooled_embeds = torch.zeros_like(pooled_prompt_embeds)
                            pooled_prompt_embeds = torch.where(mask_pooled, uncond_pooled_embeds, pooled_prompt_embeds)
                        
                        add_time_ids = self._get_add_time_ids(
                            (self.config.datasets.img_size, self.config.datasets.img_size), (0, 0), (self.config.datasets.img_size, self.config.datasets.img_size), dtype=self.trained_dtype, device=gpu, batch_size=bsz
                        )

                        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

                        # Check mode
                        if self.model.noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif self.model.noise_scheduler.config.prediction_type == "v_prediction":
                            target = self.model.noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {self.model.noise_scheduler.config.prediction_type}")

                        # Predict the noise residual and compute loss
                      # 디버깅을 위한 타입 출력
                        model_pred = self.model.unet(noisy_latents, 
                                                    timesteps, 
                                                    prompt_embeds, 
                                                    return_dict=False, 
                                                    added_cond_kwargs=added_cond_kwargs,
                                                    cross_attention_kwargs={"ip_hidden_states": control_condition_vector})[0]

                        if self.config.trainer.snr_gamma is None:
                            loss  = self.mse_loss(model_pred.float(), target.float()).mean()

                        else:
                            snr = compute_snr(self.model.noise_scheduler, timesteps)
                            mse_loss_weights = torch.stack([snr, self.config.trainer.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                                dim=1
                            )[0]
                            if self.model.noise_scheduler.config.prediction_type == "epsilon":
                                mse_loss_weights = mse_loss_weights / snr
                            elif self.model.noise_scheduler.config.prediction_type == "v_prediction":
                                mse_loss_weights = mse_loss_weights / (snr + 1)

                            loss = self.mse_loss(model_pred.float(), target.float())
                            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                            loss = loss.mean()  

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.model.accelerator.gather(loss.repeat(self.batch_size)).mean()
                    train_loss += avg_loss.item() / self.config.model.gradient_accumulation_steps
                    
                    # Backpropagate
                    self.model.accelerator.backward(loss)

                    # Optimize Step when accumulate step
                    if self.model.accelerator.sync_gradients:
                        self.model.accelerator.clip_grad_norm_(self.model.params, self.config.trainer.max_grad_norm)
                    self.model.optimizer.step()                    
                    self.model.lr_scheduler.step()
                    self.model.optimizer.zero_grad()
                    

                if self.model.accelerator.sync_gradients:

                    is_ema_step = global_step % self.config.trainer.ema_step == 0 if self.config.model.use_ema else None
                    is_valid_step = global_step % self.config.trainer.val_interval == 0
                    is_log_step   = global_step % self.config.trainer.log_interval == 0
                    
                    if is_ema_step or is_valid_step:
                        current_unet_state_dict = self.model.accelerator.get_state_dict(self.model.unet)
                        unwrapped_unet = self.model.accelerator.unwrap_model(self.model.unet)
                    
                    # For EMA Using.
                    if is_ema_step:

                        if self.config.model.off_load_ema:
                            self.model.ema_unet.to(device="cuda", non_blocking=True)
                        # self.model.ema_unet.step(self.unwrap_unet)

                        if self.config.trainer.deep_speed.zero_optimization.stage ==3:
                            self.model.ema_unet.step(current_unet_state_dict.values())
                        else:
                            self.model.ema_unet.step(unwrapped_unet.parameters())

                        if self.config.model.off_load_ema:
                            self.model.ema_unet.to(device="cpu", non_blocking=True)

                    # If main process
                    if self.model.accelerator.is_main_process:
                        progress_bar.update(1)

                        # Check Log Inter and Logging
                        if is_log_step:
                            self.model.accelerator.log({"train/loss": train_loss}, step=global_step)
                            self.model.accelerator.log({"train/lr": self.model.lr_scheduler.get_last_lr()[0]}, step=global_step)
                            train_loss = 0.0
                        
                        # Validating when Valid Iteration and Save
                        if is_valid_step:
                            save_path = os.path.join(self.ckpt_dir, self.config.model.name, f"checkpoint-{global_step}")
                            unwrapped_unet.save_pretrained(os.path.join(save_path, "unet"),
                                                            state_dict=current_unet_state_dict)
                            
                            if is_ema_step:
                                self.model.ema_unet.copy_to(unwrapped_unet.parameters())

                            # Check Heuristic Validation
                            for idx, val_data in enumerate(self.loader_valid):
                                img = val_data["gt"].to(gpu)
                                self._valid(img, global_step)
                                break

                            if is_ema_step:
                                unwrapped_unet.load_state_dict(current_unet_state_dict)

                    if is_ema_step or is_valid_step:
                        del current_unet_state_dict
                    torch.cuda.empty_cache()
                    global_step += 1

            # Logging Every Steps
            if self.model.accelerator.is_main_process:
                logs = {"loss_mse": loss.detach().item(), "lr": self.model.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

        if gpu ==0:
            save_path = os.path.join(self.ckpt_dir, self.config.model.name, f"checkpoint-{global_step}")
            self.model.accelerator.save_state(save_path)
            if self.config.model.use_ema:
                self.model.ema_unet.save_pretrained(os.path.join(save_path, "ema_unet"))
            torch.cuda.empty_cache()

            # Check Heuristic Validation
            for idx, val_data in enumerate(self.loader_valid):
                img = val_data["gt"].to(gpu)
                self._valid(img, global_step)
                break

        self.model.accelerator.wait_for_everyone()

    @torch.no_grad()
    def _valid(self, img, global_step):
        """
        Validation with fixed filter scales.
        10 samples total: 0.01~0.1 (5 steps) + 0.1~0.5 (5 steps)
        Each image in the batch is filtered at each scale and generates 1 sample.
        """
        self.model.unet.eval()
        self.model.text_encoder_1.eval()
        self.model.text_encoder_2.eval()

        # Make Pipeline
        pipeline = SDXL_SAAdapterPipeline(
                vae=self.model.vae,
                text_encoder=self.model.text_encoder_1,
                text_encoder_2=self.model.text_encoder_2,
                tokenizer=self.model.tokenizer_1,
                tokenizer_2=self.model.tokenizer_2,
                unet=self.model.accelerator.unwrap_model(self.model.unet),
                scheduler=self.model.noise_scheduler,
            )
        pipeline.to(self.model.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # Fixed filter scales: 0.01~0.1 (5 steps) + 0.1~0.5 (5 steps) = 10 total
        filter_scales = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

        # Use first image in batch for validation
        gt_single = img[0:1]  # (1, C, H, W), normalized [-1, 1]
        gt_display = torch.clamp((gt_single + 1.0) / 2.0, min=0.0, max=1.0)  # [0, 1]

        device = gt_single.device
        img_size = self.config.datasets.img_size

        all_rows = []  # Each row: [gt, filtered, pred] for one scale
        top_k = []
        psm = []

        for fs in filter_scales:
            fft_filter = GpuFixedFFT(fft_scale=fs)

            # Filter on [0,1] range image
            filtered = fft_filter(gt_display.to(dtype=self.trained_dtype))  # (1, C, H, W) float32 [0,1]

            # Normalize filtered to [-1,1] for VAE encoding
            filtered_for_vae = (filtered * 2.0 - 1.0).to(dtype=self.untrained_dtype, device=device)

            pred_images = pipeline(
                prompt=self.text_prompt,
                image=filtered_for_vae,
                height=img_size,
                width=img_size,
                num_images_per_prompt=1,
                generator=None,
                output_type="pt",
            ).images  # (1, C, H, W) [0,1]

            # Metrics (both in [0,1] range)
            top_k.append(self.top_k_accuracy(pred_images.to(device), gt_display.to(device)))
            psm.append(self.psm_accuracy(pred_images.to(device), gt_display.to(device)))

            # Build row: [gt, filtered_vis, pred] all in [0,1]
            all_rows.append(torch.cat([
                gt_display.detach().cpu(),
                filtered.detach().cpu(),
                pred_images.detach().cpu(),
            ], dim=0))  # (3, C, H, W)

            torch.cuda.empty_cache()

        # Logging for Tensorboard
        if self.model.accelerator.trackers[0].name == "tensorboard":
            # Stack all rows: (10, 3, C, H, W) -> (30, C, H, W)
            grid = torch.cat(all_rows, dim=0)
            grid_image = make_grid(grid, nrow=3, padding=2, pad_value=1.0)  # 10 rows x 3 cols
            grid_image = 255. * rearrange(grid_image, 'c h w -> h w c').cpu().numpy()

            self.model.accelerator.log({"valid/top_k": np.mean(top_k).item()}, step=global_step)
            self.model.accelerator.log({"valid/psm": np.mean(psm).item()}, step=global_step)
            self.model.accelerator.trackers[0].writer.add_images(
                "valid/fixed_filter_grid",
                torch.from_numpy(grid_image.astype(np.uint8)).unsqueeze(0),
                global_step,
                dataformats="NHWC")

        del pipeline
        torch.cuda.empty_cache()

    def train(self):
        gpus = torch.cuda.device_count()
        self._train(gpu=0, size=gpus)
    
    @torch.no_grad()
    def inference(self, 
                  unet_path ="/workspace/logs/Frequency_Adaptation/ckpt_dir/SAModel/checkpoint-37000/unet/diffusion_pytorch_model.safetensors77",
                  filter_scale:Optional[Union[int, List[int]]] = 0.0,
                  num_samples=5, 
                  cfg_scale=7.5,
                  indexes:Optional[Union[int, List[int]]]  = [],
                  output_path = "./output"):
   
        if type(indexes) == int:
            indexes = [indexes]
        
        if type(filter_scale) == float:
            filter_scale = [filter_scale]
               
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.device = 0
        self.text_prompt = self.prompt_lst[self.config.trainer.prompt_idx] # For Fixed Text Prompt
        self.model_define(self.device, ddp=False) 
        
        self.model.from_pretrained(unet_path)

        self.model.unet.eval()
        self.model.text_encoder.eval()
        self.model.vae.eval()

        self.pipeline = SDXL_SAAdapterPipeline(vae=self.model.vae,
                                        text_encoder=self.model.text_encoder,
                                        tokenizer=self.model.clip_tokenizer,
                                        unet=self.model.accelerator.unwrap_model(self.model.unet),
                                        scheduler=self.model.noise_scheduler,
                                        )

        samples = []
        top_k   = []
        psm     = []

        progress_bar = tqdm(
                range(0, len(indexes)),
                initial=0,
                desc="Test Steps",
                # Only show the progress bar once on each machine.
            )

        test_path = os.path.join(output_path, "test")
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        # Make Validation Images and Calculate Metrics
        for fs in filter_scale:
            self.makeDatasets(self.config.datasets.data_path, 
                          self.config.datasets.img_path,
                          frequency_rate = fs,
                          mean=0.5, 
                          std=0.5, 
                          ddp=self.ddp_dataset)
            
            for idx, idx_val in enumerate(indexes):
                data = self.loader_test.dataset[idx_val]
                img = data["gt"].unsqueeze(0).to(self.device)
                filtered_img = data["filtered_image"].unsqueeze(0).to(self.device)
            
                pred_images = self.pipeline(self.text_prompt, 
                                            filtered_img,
                                            height = 512,
                                            width  = 512, 
                                            guidance_scale=cfg_scale,
                                            num_images_per_prompt=num_samples, 
                                            generator=None,
                                            output_type="pt").images

                gt_image    = torch.clamp((img+1.0)/2.0, min=0.0, max=1.0).repeat(num_samples, 1, 1, 1)
                samples.append(torch.cat([gt_image.detach().cpu(), pred_images.detach().cpu()], dim=0))

                # Check Metric For Stable diffusion
                top_k_now = self.top_k_accuracy(pred_images, gt_image)
                psm_now  = self.psm_accuracy(pred_images, gt_image)
                top_k.append(top_k_now)
                psm.append(psm_now)
                
                
                grid_elements = torch.cat([gt_image, pred_images, filtered_img], dim=0)
                grid_img = make_grid(grid_elements, nrow=2+num_samples, padding=2, pad_value=1.0)
                grid_np = rearrange(grid_img, 'c h w -> h w c').detach().cpu().numpy()

                pred_images = rearrange(pred_images, 'n c h w -> n h w c').detach().cpu()
                gt_image    = rearrange(gt_image, 'n c h w -> n h w c').detach().cpu()
                filtered_img = rearrange(filtered_img, 'n c h w -> n h w c').detach().cpu()

                plt.imsave(os.path.join(test_path, f"test{idx}-{fs}-0.png"), gt_image[0].numpy())
                for sample_idx, pred in enumerate(pred_images.numpy()):
                    plt.imsave(os.path.join(test_path, f"test{idx}-{fs}-{sample_idx+1}.png"), pred)
                plt.imsave(os.path.join(test_path, f"test{idx}-{fs}-filtered_images.png"), filtered_img[0].numpy())
                plt.imsave(os.path.join(test_path, f"test{idx}_fs{fs}_grid.png"), grid_np)

                progress_bar.update(1)
                logs = {"top_k": top_k_now, "psm": psm_now, "TOP_K":np.mean(top_k).item(),"PSM":np.mean(psm).item()}
                progress_bar.set_postfix(**logs)
        print(f"Test PSM {np.mean(psm).item()} TOP K ACC {np.mean(top_k).item()}")

    @torch.no_grad()
    def test(self, unet_path ="./ckpt_dir/EEG_LDM_SD2_TOKEN_77", cond_path= None, ema_path=None, ip_adaption_path =None, num_samples=5, cfg_scale=7.5,output_path = "./output"):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.device = 0
        self.makeDatasets(self.eeg_train_path, 
                            self.eeg_test_path, 
                            self.eeg_val_path, 
                            self.img_path, 
                            self.img_size,
                            mean=0.5, 
                            std=0.5, 
                            ddp=False)
        self.model_define(self.device, ddp=False) 
        self.model.from_pretrained(unet_path, cond_path, ema_path, ip_adaption_path)

        if self.ip_adapter_enabled:
            self.model.ip_adaption_modules.eval()


        if self.config.model.use_ema:
            self.model.ema_unet.store(self.model.unet.parameters())
            self.model.ema_unet.copy_to(self.model.unet.parameters())
        
        self.model.unet.eval()

        # Make Pipelines for diffusion models
        self.pipeline = SDXL_SAAdapterPipeline(
                vae=self.model.vae,
                
                text_encoder=self.model.text_encoder_1,
                text_encoder_2=self.model.text_encoder_2, # [핵심 1] 두 번째 텍스트 인코더 추가
                tokenizer=self.model.tokenizer_1,
                tokenizer_2=self.model.tokenizer_2, # [핵심 2] 두 번째 토크나이저 추가
                unet=self.model.accelerator.unwrap_model(self.model.unet),
                # scheduler=self.model.accelerator.unwrap_model(self.model.noise_scheduler),
                scheduler=self.model.noise_scheduler,
            )

        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

        #################################
        # Validation Data Start
        ################################
        # top_k   = []
        # psm     = []


        # progress_bar = tqdm(
        #         range(0, len(self.loader_valid)),
        #         initial=0,
        #         desc="Valid Steps",
        #         # Only show the progress bar once on each machine.
        #     )

        # val_path = os.path.join(output_path, "val")
        # if not os.path.exists(val_path):
        #     os.mkdir(val_path)

        # # Make Validation Images and Calculate Metrics
        # for idx, val_data in enumerate(self.loader_valid):
        #     img = val_data["gt"].to(self.device)
        #     eeg = val_data["eeg"].to(self.device)
           
        #     pred_images = self.pipeline(eeg, height = 512, width = 512, num_images_per_prompt=num_samples, generator=None).images
        #     gt_image    = torch.clamp((img+1.0)/2.0, min=0.0, max=1.0)

        #     # Check Metric For Stable diffusion
        #     top_k_now = self.top_k_accuracy(pred_images, gt_image)
        #     psm_now  = np.mean(self.psm_accuracy(pred_images, gt_image)).item()
        #     top_k.append(top_k_now)
        #     psm.append(psm_now)
            

        #     pred_images = rearrange(pred_images, 'n c h w -> n h w c').detach().cpu().numpy()
        #     gt_image    = rearrange(gt_image, 'n c h w -> n h w c').detach().cpu().numpy()

        #     plt.imsave(os.path.join(val_path, f"val{idx}-0-0.png"), gt_image[0])
        #     for sample_idx, pred in enumerate(pred_images):
        #         plt.imsave(os.path.join(val_path, f"val{idx}-0-{sample_idx+1}.png"), pred)
            
        #     progress_bar.update(1)
        #     logs = {"top_k": top_k_now, "psm": psm_now, "TOP_K":np.mean(top_k).item(),"PSM":np.mean(psm).item()}
        #     progress_bar.set_postfix(**logs)
        # print(f"Validation PSM {np.mean(psm).item()} TOP K ACC {np.mean(top_k).item()}")


        #################################
        # Test Data Start
        #################################

        samples = []
        top_k   = []
        psm     = []

        progress_bar = tqdm(
                range(0, len(self.loader_test)),
                initial=0,
                desc="Test Steps",
                # Only show the progress bar once on each machine.
            )

        test_path = os.path.join(output_path, "test")
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        # Make Validation Images and Calculate Metrics
        # Make Validation Images and Calculate Metrics
        for idx, val_data in enumerate(self.loader_test):
            img = val_data["gt"].to(self.device)
            eeg = val_data["fi"].to(self.device) # 테스트셋의 Structure Image라고 가정
            
            # [수정 2] 인자 명시, 해상도 1024, output_type 추가
            pred_images = self.pipeline(
                prompt=self.text_prompt, # [필수] 프롬프트 추가
                image=eeg,               # [필수] eeg를 image 인자로 지정
                height=self.config.datasets.img_size, 
                width=self.config.datasets.img_size, 
                num_images_per_prompt=num_samples, 
                guidance_scale=cfg_scale,
                generator=None,
                output_type="pt"         # [필수]
            ).images
            
            gt_image = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
            if gt_image.shape[-1] != self.config.datasets.img_size:
                gt_image = F.interpolate(gt_image, size=(self.config.datasets.img_size, self.config.datasets.img_size), mode='bilinear', align_corners=False)

            samples.append(torch.cat([gt_image.detach().cpu(), pred_images.detach().cpu()], dim=0))

            # Check Metric For Stable diffusion
            top_k_now = self.top_k_accuracy(pred_images, gt_image)
            psm_now  = self.psm_accuracy(pred_images, gt_image)
            top_k.append(top_k_now)
            psm.append(psm_now)
            
            # 저장 시 (C, H, W) -> (H, W, C) 로 변환
            pred_images_np = rearrange(pred_images, 'n c h w -> n h w c').detach().cpu().numpy()
            gt_image_np    = rearrange(gt_image, 'n c h w -> n h w c').detach().cpu().numpy()

            plt.imsave(os.path.join(test_path, f"test{idx}-0-0.png"), gt_image_np[0])
            for sample_idx, pred in enumerate(pred_images_np):
                plt.imsave(os.path.join(test_path, f"test{idx}-0-{sample_idx+1}.png"), pred)
            
            progress_bar.update(1)
            logs = {"top_k": top_k_now, "psm": psm_now, "TOP_K":np.mean(top_k).item(),"PSM":np.mean(psm).item()}
            progress_bar.set_postfix(**logs)
        print(f"Test PSM {np.mean(psm).item()} TOP K ACC {np.mean(top_k).item()}")
        



