from typing import Optional, Union, List
import gc, os
import cv2
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

from models               import Model, SAAdapterPipeline, Frozen_CLIPImageNTextEmbedder as ImageClip
from trainer.base_trainer import BaseTrainer
from utils import GpuFixedFFT
# from libs.metric          import get_similarity_metric

from diffusers.training_utils import compute_dream_and_update_latents, compute_snr
from diffusers.optimization   import get_scheduler


from torchvision.models      import ViT_H_14_Weights, vit_h_14
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Trainer(BaseTrainer):
    def __init__(self, config_file_path):
        self.config = OmegaConf.load(config_file_path)
        self.startEpoch     = 0
        self.eps            = 1e-6
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
        self.model.vae.to(gpu)
        self.model.unet.to(gpu)
        self.model.text_encoder.to(gpu)

        if self.config.model.use_ema:
            if self.config.model.off_load_ema:
                self.model.ema_unet.pin_memory()
            else:
                self.model.ema_unet.to(gpu)


    @torch.no_grad()
    def top_k_accuracy(self, pred_imgs, gt_imgs):
        # For Metric
        self.acc_model.to(self.device, dtype = torch.float16)
        self.acc_model = self.acc_model.eval()

        pred = self.acc_model_preprocess(pred_imgs).to(dtype = torch.float16)
        gt = self.acc_model_preprocess(gt_imgs).to(dtype = torch.float16)

        gt_class_id = self.acc_model(gt)
        gt_class_id = gt_class_id.softmax(1).argmax()

        pred_class_id = self.acc_model(pred)
        pred_class_id = pred_class_id.softmax(1).argmax(1)

        acc_mean = (pred_class_id == gt_class_id).to(dtype=torch.float32).mean().item()
        self.acc_model.to("cpu")
        return acc_mean
    
    @torch.no_grad()
    def psm_accuracy(self, pred_imgs, gt_imgs):
        self.lpips.to(self.device, dtype = torch.float16)
        acc_mean = self.lpips(pred_imgs.to(dtype = torch.float16), gt_imgs.to(dtype = torch.float16)).item()
        self.lpips.to("cpu")
        return acc_mean
       

    def _train(self, gpu, size):
        print(f"Now Initialize Rank: {gpu} | Number Of GPU : {size}")
        self.device = gpu
        self.initialize(gpu, size)
        self.model_define(gpu, ddp=self.ddp) 
        self.makeDatasets(self.config.datasets.data_path, 
                          self.config.datasets.img_path,
                          mean=0.5, 
                          std=0.5, 
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
                
                with self.model.accelerator.accumulate(self.model.unet): # for cumulative execution.
                    with self.model.accelerator.autocast():  # For amp
                        # Convert images to latent space
                        # Data는 이미지, 주파수 필터링된 이미지, prompt를 제공할예정
                        img = data["gt"].to(gpu)
                        filtered_img = data["filtered_image"].to(gpu)

                        latents = self.model.vae.encode(img).latent_dist.sample()  # H/16, W/16
                        control_condition_vector =self.model.vae.encode(filtered_img).latent_dist.sample().to(dtype=torch.float16) # H/16, W/16
                        
                        # Scaling the latents
                        latents = latents * self.model.vae.config.scaling_factor
                        control_condition_vector = control_condition_vector * self.model.vae.config.scaling_factor

                        # Sample Noise
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]

                        
                        timesteps = torch.randint(0, self.model.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()
                        noisy_latents = self.model.noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get Condition Vector
                        text_inputs = self.model.clip_tokenizer(
                            [self.text_prompt], # 학습 프롬프트 (데이터셋에 따라 다름)
                            max_length=self.model.clip_tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        ).input_ids.to(gpu)

                        with torch.no_grad():
                            encoder_hidden_states = self.model.text_encoder(text_inputs)[0]

                        # Classifier-Free Guidance (Conditional Dropout) Implementation
                        if self.model.cfg_scale > 0.0:
                            #Make CFG Mask
                            mask = (torch.rand(bsz, device=encoder_hidden_states.device) < self.model.cfg_scale).view(bsz, 1, 1)                            

                            unconditional_tokens = torch.zeros_like(encoder_hidden_states)
                            # Change Condition vector to 0
                            encoder_hidden_states = torch.where(mask, unconditional_tokens, encoder_hidden_states)

                        # Check mode
                        if self.model.noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif self.model.noise_scheduler.config.prediction_type == "v_prediction":
                            target = self.model.noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {self.model.noise_scheduler.config.prediction_type}")

                        #  Efficient Calculatiion but not use in v_prediction
                        if self.model.noise_scheduler.config.prediction_type == "epsilon":
                            noisy_latents, target = compute_dream_and_update_latents(
                                self.model.unet,
                                self.model.noise_scheduler,
                                timesteps,
                                noise,
                                noisy_latents,
                                target,
                                encoder_hidden_states,
                                1.0,
                            )
                        # Predict the noise residual and compute loss
                        model_pred = self.model.unet(noisy_latents, 
                                                    timesteps, 
                                                    encoder_hidden_states, 
                                                    return_dict=False, 
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

                                # ema_state_dict = self.model.ema_unet.state_dict()                    
                                # # 2. 저장용 EMA 체크포인트 생성 (메인 프로세스에서만)
                                # unwrapped_unet.save_pretrained(
                                #     os.path.join(save_path, "ema_unet"),
                                #     state_dict=ema_state_dict
                                # )
                                
                                # # 3. [스왑] 검증 이미지를 뽑기 위해 실제 모델에 EMA 주입
                                # unwrapped_unet.load_state_dict(ema_state_dict)
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
        Each scale produces 1 generated image from the first image in batch.
        Tensorboard grid: 10 rows x 3 cols (GT | Filtered | Generated)
        """
        self.model.unet.eval()
        self.model.text_encoder.eval()

        pipeline = SAAdapterPipeline(
                vae=self.model.vae,
                text_encoder=self.model.text_encoder,
                tokenizer=self.model.clip_tokenizer,
                unet=self.model.accelerator.unwrap_model(self.model.unet),
                scheduler=self.model.noise_scheduler,
            )
        pipeline.to(self.model.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        filter_scales = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

        gt_single = img[0:1]  # (1, C, H, W), normalized [-1, 1]
        gt_display = torch.clamp((gt_single + 1.0) / 2.0, min=0.0, max=1.0)  # [0, 1]
        device = gt_single.device

        all_rows = []
        top_k = []
        psm = []

        for fs in filter_scales:
            fft_filter = GpuFixedFFT(fft_scale=fs)
            filtered = fft_filter(gt_display.float())  # (1, C, H, W) float32 [0,1]

            filtered_for_vae = (filtered * 2.0 - 1.0).to(dtype=torch.float16, device=device)

            pred_images = pipeline(
                self.text_prompt,
                filtered_for_vae,
                height=512,
                width=512,
                num_images_per_prompt=1,
                guidance_scale=7.5,
                generator=None,
                output_type="pt",
            ).images  # (1, C, H, W) [0,1]

            # Metrics (both in [0,1] range)
            top_k.append(self.top_k_accuracy(pred_images.to(device), gt_display.to(device)))
            psm.append(self.psm_accuracy(pred_images.to(device), gt_display.to(device)))

            all_rows.append(torch.cat([
                gt_display.detach().cpu(),
                filtered.detach().cpu(),
                pred_images.detach().cpu(),
            ], dim=0))  # (3, C, H, W)

            torch.cuda.empty_cache()

        # Logging for Tensorboard
        if self.model.accelerator.trackers[0].name == "tensorboard":
            grid = torch.cat(all_rows, dim=0)  # (30, C, H, W)
            grid_image = make_grid(grid, nrow=3, padding=2, pad_value=1.0)
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

        self.pipeline = SAAdapterPipeline(vae=self.model.vae,
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
    def target_infer(self, 
                  unet_path ="/workspace/logs/Frequency_Adaptation/ckpt_dir/SAModel/checkpoint-37000/unet/diffusion_pytorch_model.safetensors77",
                  filter_scale:Optional[Union[int, List[int]]] = 0.0,
                  num_samples=5, 
                  cfg_scale=7.5,
                  indexes:Optional[Union[int, List[int]]]  = [],
                  img_path = "./images/test.png",
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

        self.pipeline = SAAdapterPipeline(vae=self.model.vae,
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
        filtered_img = torch.from_numpy(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2GRAY).astype(np.float32)).to(self.device).repeat(3, 1, 1).unsqueeze(0)  / 255.0
        # filtered_img = data["filtered_image"].unsqueeze(0).to(self.device)
    
        pred_images = self.pipeline(self.text_prompt, 
                                    filtered_img,
                                    height = 512,
                                    width  = 512, 
                                    guidance_scale=cfg_scale,
                                    num_images_per_prompt=num_samples, 
                                    generator=None,
                                    output_type="pt").images

        # gt_image    = torch.clamp((img+1.0)/2.0, min=0.0, max=1.0).repeat(num_samples, 1, 1, 1)
        # samples.append(torch.cat([gt_image.detach().cpu(), pred_images.detach().cpu()], dim=0))

        # Check Metric For Stable diffusion
        # top_k_now = self.top_k_accuracy(pred_images, gt_image)
        # psm_now  = self.psm_accuracy(pred_images, gt_image)
        # top_k.append(top_k_now)
        # psm.append(psm_now)
        

        # gt_image    = rearrange(gt_image, 'n c h w -> n h w c').detach().cpu()
        
        grid_elements = torch.cat([pred_images, filtered_img], dim=0)
        grid_img = make_grid(grid_elements, nrow=2+num_samples, padding=2, pad_value=1.0)
        grid_np = rearrange(grid_img, 'c h w -> h w c').detach().cpu().numpy()

        filtered_img = rearrange(filtered_img, 'n c h w -> n h w c').detach().cpu()
        pred_images = rearrange(pred_images, 'n c h w -> n h w c').detach().cpu()

        for sample_idx, pred in enumerate(pred_images.numpy()):
            plt.imsave(os.path.join(test_path, f"test-{sample_idx}.png"), pred)
        plt.imsave(os.path.join(test_path, f"skecth_images.png"), filtered_img[0].numpy())
        print(pred_images.shape, filtered_img.shape, grid_np.shape)
        plt.imsave(os.path.join(test_path, f"test-grid.png"), grid_np)

        progress_bar.update(1)
        # logs = {"top_k": top_k_now, "psm": psm_now, "TOP_K":np.mean(top_k).item(),"PSM":np.mean(psm).item()}
        logs = {"target_name": img_path}
        progress_bar.set_postfix(**logs)
        print(f"Test PSM {np.mean(psm).item()} TOP K ACC {np.mean(top_k).item()}")

    
    
    
    @torch.no_grad()
    def test(self, unet_path ="./ckpt_dir/EEG_LDM_SD2_TOKEN_77", cond_path= None, ema_path=None, ip_adaption_path =None, num_samples=5, cfg_scale=7.5,output_path = "./output"):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.device = 0
        self.model_define(self.device, ddp=False) 
        self.makeDatasets(self.config.datasets.data_path, 
                          self.config.datasets.img_path,
                          mean=0.5, 
                          std=0.5, 
                        #   min_value=0, 
                          ddp=self.ddp_dataset)
        self.model.from_pretrained(unet_path, cond_path, ema_path, ip_adaption_path)
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

        if self.ip_adapter_enabled:
            self.model.ip_adaption_modules.eval()


        if self.config.model.use_ema:
            self.model.ema_unet.store(self.model.unet.parameters())
            self.model.ema_unet.copy_to(self.model.unet.parameters())
        
        self.model.unet.eval()

        # Make Pipelines for diffusion models
        self.pipeline = SAAdapterPipeline(vae=self.model.vae,
                                            text_encoder=self.model.text_encoder,
                                            tokenizer=self.model.clip_tokenizer,
                                            unet=self.model.accelerator.unwrap_model(self.model.unet),
                                            scheduler=self.model.noise_scheduler)

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
        for idx, val_data in enumerate(self.loader_test):
            img = val_data["gt"].to(self.device)
            eeg = val_data["eeg"].to(self.device)
           
            pred_images = self.pipeline(eeg, height = 512, width = 512, num_images_per_prompt=num_samples, guidance_scale=cfg_scale,generator=None).images
            gt_image    = torch.clamp((img+1.0)/2.0, min=0.0, max=1.0)
            samples.append(torch.cat([gt_image.detach().cpu(), pred_images.detach().cpu()], dim=0))

            # Check Metric For Stable diffusion
            top_k_now = self.top_k_accuracy(pred_images, gt_image)
            psm_now  = self.psm_accuracy(pred_images, gt_image)
            top_k.append(top_k_now)
            psm.append(psm_now)
            

            pred_images = rearrange(pred_images, 'n c h w -> n h w c').detach().cpu().numpy()
            gt_image    = rearrange(gt_image, 'n c h w -> n h w c').detach().cpu().numpy()

            plt.imsave(os.path.join(test_path, f"test{idx}-0-0.png"), gt_image[0])
            for sample_idx, pred in enumerate(pred_images):
                plt.imsave(os.path.join(test_path, f"test{idx}-0-{sample_idx+1}.png"), pred)
            
            progress_bar.update(1)
            logs = {"top_k": top_k_now, "psm": psm_now, "TOP_K":np.mean(top_k).item(),"PSM":np.mean(psm).item()}
            progress_bar.set_postfix(**logs)
        print(f"Test PSM {np.mean(psm).item()} TOP K ACC {np.mean(top_k).item()}")
        



