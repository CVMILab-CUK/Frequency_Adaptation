import gc, os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL       import Image
from tqdm.auto import tqdm


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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.fsdp as fsdp
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,  CPUOffload


from torchvision.utils import make_grid

from models               import Model, EEGDiffusionPipeline, Frozen_CLIPImageNTextEmbedder as ImageClip
from trainer.base_trainer import BaseTrainer
# from libs.metric          import get_similarity_metric

from diffusers.training_utils import compute_dream_and_update_latents, compute_snr
from diffusers.optimization   import get_scheduler


from torchvision.models      import ViT_H_14_Weights, vit_h_14
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class LDM2Trainer(BaseTrainer):
    def __init__(self, json_file, sharedFilePath, training_mode="ddp"):
        self.json_dict = self.json_load(json_file)
        self.sharedFilePath = sharedFilePath
        self.startEpoch     = 0
        self.globalStep     = 0
        self.eps            = 1e-6
        self.training_mode  = training_mode 
        super().__init__(self.json_dict["ckpt_dir"], self.json_dict["log_dir"], self.json_dict["batch_size"], 
                         sharedFilePath, self.json_dict["num_workers"])

        fsdp.state_dict_type = "full" 
        # fsdp.FSDP.allgather_chunk_size_mb = 128

        self.json_parser()
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
        self.ddp = True
        self.ddp_dataset = True



        
    def json_parser(self):
        pass
        # ########################################
        # #            Train Setting
        # ########################################
        # self.name               = self.json_dict["name"]
        # self.data_name          = self.json_dict["data_name"]
        # self.log_dir            = self.json_dict["log_dir"]
        # self.ckpt_dir           = self.json_dict["ckpt_dir"]
        # self.eeg_train_path     = self.json_dict["eeg_train_path"]
        # self.eeg_test_path      = self.json_dict["eeg_test_path"]
        # self.eeg_val_path       = self.json_dict["eeg_val_path"]
        # self.eeg_pretrian_path  = self.json_dict["eeg_pretrian_path"]
        # self.pretrain_path      = self.json_dict["pretrain_path"]
        # self.img_path           = self.json_dict["img_path"]
        # self.split_path         = self.json_dict["split_path"]

        # ########################################
        # #            Model Setting
        # ########################################
        # self.in_seq         = self.json_dict["in_seq"]
        # self.in_channels    = self.json_dict["in_channels"]
        # self.z_channels     = self.json_dict["z_channels"]
        # self.out_seq        = self.json_dict["out_seq"]
        # self.dims           = self.json_dict["dims"]
        # self.shortcut       = bool(self.json_dict["shortcut"])
        # self.dropout        = self.json_dict["dropout"]
        # self.groups         = self.json_dict["groups"]
        # self.layer_mode     = self.json_dict["layer_mode"]
        # self.block_mode     = self.json_dict["block_mode"]
        # self.down_mode      = self.json_dict["down_mode"]
        # self.up_mode        = self.json_dict["up_mode"]
        # self.pos_mode       = self.json_dict["pos_mode"]
        # self.skip_mode      = self.json_dict["skip_mode"]
        # self.sim_mode       = self.json_dict["sim_mode"]
        # self.n_layer        = self.json_dict["n_layer"]
        # self.n_head         = self.json_dict["n_head"]
        # self.dff_factor     = self.json_dict["dff_factor"]
        # self.stride         = self.json_dict["stride"]
        # self.sdsc_lambda    = self.json_dict["sdsc_lambda"]
        # self.sim_lambda     = self.json_dict["sim_lambda"]
        # self.img_size       = self.json_dict["img_size"]


        # ########################################
        # #         Training Parameters
        # ########################################
        # self.lr        = self.json_dict["learningRate"]
        # self.epochs    = self.json_dict["trainEpochs"]
        # self.saveIter  = self.json_dict["saveIter"]
        # self.validIter = self.json_dict["validIter"]
        # self.logIter   = self.json_dict["logIter"]
        # self.restart   = self.json_dict["restart"]
        # self.cfg_scale = self.json_dict["cfg_scale"]

        # ########################################
        # #         LDM Parameters
        # ########################################
        # self.clip_tune   = bool(self.json_dict["clip_tune"])
        # self.cls_tune    = bool(self.json_dict["cls_tune"])
        # self.eval_avg    = bool(self.json_dict["eval_avg"])
        # self.num_samples = self.json_dict["num_samples"]
        # self.ddim_steps  = self.json_dict["ddim_steps"]
        # self.gradient_accumulation_step = self.json_dict["gradient_accumulation_step"]
        # self.lr_warmup_steps            = self.json_dict["lr_warmup_steps"]
        # self.lr_scheduler               = self.json_dict["lr_scheduler"]
        # self.max_grad_norm              = self.json_dict["max_grad_norm"]
        # self.snr_gamma                  = None if self.json_dict["snr_gamma"] == 0 else self.json_dict["snr_gamma"]
        # self.use_ema                    = bool(self.json_dict["use_ema"])
        # self.offload_ema                = bool(self.json_dict["offload_ema"])
        # self.ip_adapter_enabled         = bool(self.json_dict["ip_adapter_enabled"])
        # self.ip_adapter_token_num       = int(self.json_dict["ip_adapter_token_num"])

    
    def model_define(self, gpu, ddp=True):
        self.model = Model()
        # self.model = eLDM2(self.eeg_pretrian_path, torch.device("cuda"), self.pretrain_path, 
        #                   in_seq = self.in_seq, in_channels = self.in_channels, out_channels=self.z_channels, out_seq = self.out_seq,
        #                   dims = self.dims, shortcut = self.shortcut, dropout = self.dropout,
        #                   groups = self.groups, layer_mode = self.layer_mode, block_mode = self.block_mode,
        #                   down_mode = self.down_mode, pos_mode = self.pos_mode, skip_mode = self.skip_mode, learning_rate=self.lr,
        #                   n_layer = self.n_layer, n_head = self.n_head, dff_factor = self.dff_factor, use_ema=self.use_ema,
        #                   stride =  self.stride, epochs=self.epochs, gradient_accumulation_steps=self.gradient_accumulation_step, 
        #                   training_mode=self.training_mode, ip_adapter_enabled=self.ip_adapter_enabled, ip_adapter_token_num=self.ip_adapter_token_num, cfg_scale = self.cfg_scale)

        
        
        #Set LDM
        self.model.vae.to(gpu)
        self.model.unet.to(gpu)
        self.model.cond_models.to(gpu)
        # For clip similarity

        if self.clip_tune:            
            self.clip = ImageClip()
            self.clip.model.to(gpu)

        if self.use_ema:
            if self.offload_ema:
                self.model.ema_unet.pin_memory()
            else:
                self.model.ema_unet.to(gpu)

        if self.ip_adapter_enabled:
            self.model.ip_adaption_modules.to(gpu)

        if ddp:
            
            if self.ip_adapter_enabled:
                self.model.ip_adaption_modules = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(self.model.ip_adaption_modules))
                self.model.unet.to(gpu)
            else:
                self.model.unet= DDP(nn.SyncBatchNorm.convert_sync_batchnorm(self.model.unet), find_unused_parameters=False,  gradient_as_bucket_view=True)
            
            # self.model.cond_models= DDP(nn.SyncBatchNorm.convert_sync_batchnorm(self.model.cond_models), find_unused_parameters=True)

        # # For Metric

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
        self.makeDatasets(self.eeg_train_path, 
                          self.eeg_test_path, 
                          self.eeg_val_path, 
                          self.img_path, 
                          self.img_size,
                          mean=0.5, 
                          std=0.5, 
                          min_value=0, 
                          ddp=self.ddp_dataset)

        # Set Scheduler
        num_warmup_steps_for_scheduler = self.lr_warmup_steps * self.model.accelerator.num_processes
        len_train_dataloader_after_sharding = int(np.ceil(len(self.loader_train) / self.model.accelerator.num_processes))
        num_update_steps_per_epoch = int(np.ceil(len_train_dataloader_after_sharding / self.gradient_accumulation_step))
        num_training_steps_for_scheduler = num_update_steps_per_epoch * self.model.accelerator.num_processes * self.epochs

        # Set Varialbes
        global_step = 0

        
        self.model.lr_scheduler = get_scheduler(
                self.lr_scheduler,
                optimizer=self.model.optimizer,
                num_warmup_steps=num_warmup_steps_for_scheduler,
                num_training_steps=num_training_steps_for_scheduler,
            )

        # Model define
        if self.ip_adapter_enabled:
            self.model.unet, self.model.ip_adaption_modules, self.model.optimizer, self.loader_train, self.model.lr_scheduler = self.model.accelerator.prepare(
                self.model.unet, self.model.ip_adaption_modules, self.model.optimizer, self.loader_train, self.model.lr_scheduler
            )
        else:
            self.model.unet, self.model.optimizer, self.loader_train, self.model.lr_scheduler = self.model.accelerator.prepare(
                self.model.unet, self.model.optimizer, self.loader_train, self.model.lr_scheduler
            )

        # Define Log Dir
        self.model.accelerator.init_trackers(f"{self.name}_project")

        if not os.path.exists(os.path.join(".", "log", f"{self.name}_project")):
            os.makedirs(os.path.join(".", "log", f"{self.name}_project"), exists_ok=True)
        
        # self.scaler = torch.cuda.amp.GradScaler()

        print(f"DDP RANK {gpu} RUN...")
        if gpu == 0:
            initial_global_step = 0
            max_train_steps = self.epochs * num_update_steps_per_epoch
            progress_bar = tqdm(
                range(0, max_train_steps),
                initial=initial_global_step,
                desc="Train Steps",
                # Only show the progress bar once on each machine.
                disable=not self.model.accelerator.is_local_main_process,
            )

        for epoch in range(self.startEpoch, self.epochs):
            self.epoch = epoch
            self.train_dataset_sampler.set_epoch(epoch),
            self.test_dataset_sampler.set_epoch(epoch)
            torch.cuda.empty_cache()
            gc.collect()

            train_loss = 0.0

            for step, data in enumerate(self.loader_train):
                self.model.unet.train()
                if self.ip_adapter_enabled:
                    self.model.ip_adaption_modules.train()
                    # self.model.unet.eval()

                # self.model.cond_models.train()
                # self.model.optimizer.zero_grads()
                # torch.cuda.empty_cache()
                # torch.cuda.reset_peak_memory_stats()
                
                with self.model.accelerator.accumulate(self.model.unet): # for cumulative execution.
                    with torch.cuda.amp.autocast():  # For amp
                        # Convert images to latent space
                        img = data["image"].to(gpu, dtype=torch.float16)
                        eeg = data["eeg"].to(gpu, dtype=torch.float16)

                        latents = self.model.vae.encode(img).latent_dist.sample()
                        latents = latents * self.model.vae.config.scaling_factor

                        # Sample Noise
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]

                        
                        timesteps = torch.randint(0, self.model.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()
                        noisy_latents = self.model.noise_scheduler.add_noise(latents, noise, timesteps)

                        eeg_condition_vector = self.model.cond_models(eeg)

                        # Classifier-Free Guidance (Conditional Dropout) Implementation
                        if self.cfg_scale > 0.0:
                            #Make CFG Mask
                            mask = (torch.rand(bsz, device=eeg_condition_vector.device) < self.cfg_scale).view(bsz, 1, 1)                            

                            unconditional_tokens = torch.zeros_like(eeg_condition_vector)
                            # Change Condition vector to 0
                            eeg_condition_vector = torch.where(mask, unconditional_tokens, eeg_condition_vector)


                        # Check mode
                        if self.model.noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif self.model.noise_scheduler.config.prediction_type == "v_prediction":
                            target = self.model.noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {self.model.noise_scheduler.config.prediction_type}")

                        if self.ip_adapter_enabled:
                            eeg_adaptor_vector   = eeg_condition_vector.mean(dim=1)
                            eeg_adaptor_vector   = self.model.ip_adaption_modules(eeg_adaptor_vector)
                            eeg_condition_vector = torch.cat([eeg_condition_vector, eeg_adaptor_vector], dim=1)
                            # print(eeg_condition_vector.shape)
                            # eeg_condition_vector = self.model.ip_adaption_modules(eeg_condition_vector)

                        #  Efficient Calculatiion but not use in v_prediction
                        if self.model.noise_scheduler.config.prediction_type == "epsilon":
                            noisy_latents, target = compute_dream_and_update_latents(
                                self.model.unet,
                                self.model.noise_scheduler,
                                timesteps,
                                noise,
                                noisy_latents,
                                target,
                                eeg_condition_vector,
                                1.0,
                            )
                        # Predict the noise residual and compute loss
                        model_pred = self.model.unet(noisy_latents, timesteps, eeg_condition_vector, return_dict=False)[0]

                        if self.snr_gamma is None:
                            loss  = self.mse_loss(model_pred.float(), target.float()).mean()

                        else:
                            snr = compute_snr(self.model.noise_scheduler, timesteps)
                            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                                dim=1
                            )[0]
                            if self.model.noise_scheduler.config.prediction_type == "epsilon":
                                mse_loss_weights = mse_loss_weights / snr
                            elif self.model.noise_scheduler.config.prediction_type == "v_prediction":
                                mse_loss_weights = mse_loss_weights / (snr + 1)

                            loss = self.mse_loss(model_pred.float(), target.float())
                            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                            loss = loss.mean()  

                        # Clip Sim Loss
                        if self.clip_tune:
                            with torch.no_grad():
                                clip_embed =  self.clip(img)
                            clip_sim_loss = 1 - torch.cosine_similarity(eeg_condition_vector.squeeze().mean(1), clip_embed, dim=-1).mean()                            
                            loss += clip_sim_loss

                        # Gather the losses across all processes for logging (if we use distributed training).
                        avg_loss = self.model.accelerator.gather(loss.repeat(self.batch_size)).mean()
                        train_loss += avg_loss.item() / self.gradient_accumulation_step
                        
                        # Backpropagate
                        self.model.accelerator.backward(loss)

                        # Optimize Step when accumulate step
                        if self.model.accelerator.sync_gradients:
                            self.model.accelerator.clip_grad_norm_(self.model.params, self.max_grad_norm)
                        self.model.optimizer.step()                    
                        self.model.lr_scheduler.step()
                        self.model.optimizer.zero_grad()
                    

                if self.model.accelerator.sync_gradients:
                    # For EMA Using.
                    if self.use_ema:
                        if self.offload_ema:
                            self.model.ema_unet.to(device="cuda", non_blocking=True)
                        # self.model.ema_unet.step(self.unwrap_unet)
                        self.model.ema_unet.step(self.model.unet.parameters())
                        if self.offload_ema:
                            self.model.ema_unet.to(device="cpu", non_blocking=True)

                    # If main process
                    if self.model.accelerator.is_main_process:
                        progress_bar.update(1)

                        # Check Log Inter and Logging
                        if global_step % self.logIter == 0:
                            self.model.accelerator.log({"train/loss": train_loss}, step=global_step)
                            self.model.accelerator.log({"train/lr": self.model.lr_scheduler.get_last_lr()[0]}, step=global_step)
                            train_loss = 0.0
                        
                        # Validating when Valid Iteration and Save
                        if global_step % self.validIter == 0:
                            save_path = os.path.join(self.ckpt_dir, self.name, f"checkpoint-{global_step}")
                            unwrap_unet = self.model.accelerator.unwrap_model(self.model.unet)
                            unwrap_unet.save_pretrained(os.path.join(save_path, "unet"),
                                                            state_dict=self.model.accelerator.get_state_dict(self.model.unet))
                            
                            if self.ip_adapter_enabled:
                                # For SAVE IP Adapter
                                unwrap_ip_adaption = self.model.accelerator.unwrap_model(self.model.ip_adaption_modules)
                                if self.model.accelerator.is_main_process:
                                    state_dict = unwrap_ip_adaption.state_dict()
                                    os.makedirs(os.path.join(save_path,"ip_adaption"), exist_ok=True)
                                    save_file(state_dict, 
                                            os.path.join(save_path,"ip_adaption","diffusion_pytorch_model.safetensors"))
                                ip_adaption_weights = load_file(os.path.join(save_path,"ip_adaption","diffusion_pytorch_model.safetensors"))
                                for name, param in self.model.ip_adaption_modules.named_parameters():
                                    if name in ip_adaption_weights:
                                        param.data.copy_(ip_adaption_weights[name])
                                    elif "module."+name in ip_adaption_weights:
                                        param.data.copy_(ip_adaption_weights["module."+name])

                            if self.use_ema:
                                self.model.ema_unet.store( self.model.unet.parameters())
                                self.model.ema_unet.copy_to( self.model.unet.parameters())
                                unwrap_unet = self.model.accelerator.unwrap_model(self.model.unet)
                                unwrap_unet.save_pretrained(os.path.join(save_path, "ema_unet"),
                                                            state_dict=self.model.accelerator.get_state_dict(self.model.unet))
                                self.model.ema_unet.restore(self.model.unet.parameters())

                            torch.cuda.empty_cache()

                            # Check Heuristic Validation 
                            for idx, val_data in enumerate(self.loader_valid):
                                img = val_data["image"].to(gpu)
                                eeg = val_data["eeg"].to(gpu)
                                self._valid(eeg, img, global_step)
                                break
                                # if idx == 4:
                                #     break

                    dist.barrier() 
                    global_step += 1

            # Logging Every Steps
            if self.model.accelerator.is_main_process:
                logs = {"loss_mse": loss.detach().item(), "lr": self.model.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

        if gpu ==0:
            save_path = os.path.join(self.ckpt_dir, self.name, f"checkpoint-{global_step}")
            self.model.accelerator.save_state(save_path)
            if self.use_ema:
                self.model.ema_unet.save_pretrained(os.path.join(save_path, "ema_unet"))
            torch.cuda.empty_cache()

            # Check Heuristic Validation 
            for idx, val_data in enumerate(self.loader_valid):
                img = val_data["image"].to(gpu)
                eeg = val_data["eeg"].to(gpu)
                self._valid(eeg, img, global_step)
                break

        self.model.accelerator.wait_for_everyone()

    @torch.no_grad()
    def _valid(self, eeg, image, global_step, num_samples:int=3):# For Validation
        batch_size, _, _ = eeg.shape
        
        self.model.unet.eval()
        if self.ip_adapter_enabled:
            self.model.ip_adaption_modules.eval()

        if self.use_ema:            
            self.model.ema_unet.store( self.model.unet.parameters())
            self.model.ema_unet.copy_to( self.model.unet.parameters())

        # Make Pipelines for diffusion models
        self.pipeline = EEGDiffusionPipeline(#self.model.accelerator.unwrap_model(self.model.vae),
        #                                      self.model.accelerator.unwrap_model(self.model.cond_models),
                                            self.model.vae,
                                            self.model.cond_models,
                                            self.model.accelerator.unwrap_model(self.model.unet), 
                                            self.model.accelerator.unwrap_model(self.model.noise_scheduler),
                                            self.model.accelerator.unwrap_model(self.model.ip_adaption_modules),
                                            )

        self.pipeline.to(self.model.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

        # print("unwrap ema")
        samples = []
        top_k   = []
        psm     = []

        # Make Validation Images and Calculate Metrics
        for eeg_data, gt_image in zip(eeg, image):
            pred_images = self.pipeline(eeg_data.unsqueeze(0), height = 512, width = 512, num_images_per_prompt=num_samples, generator=None).images
            gt_image    = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0).unsqueeze(0)
            samples.append(torch.cat([gt_image.detach().cpu(), pred_images.detach().cpu()], dim=0))
            # Check Metric For Stable diffusion
            top_k.append(self.top_k_accuracy(pred_images, gt_image))
            psm.append(self.psm_accuracy(pred_images, gt_image))
            torch.cuda.empty_cache()
        
        # Logging for Tensorboard
        if self.model.accelerator.trackers[0].name == "tensorboard":
            grid = torch.stack(samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid_image = make_grid(grid, nrow=num_samples+1)
            grid_image = 255. * rearrange(grid_image, 'c h w -> h w c').cpu().numpy()

            #Actual Log Part
            self.model.accelerator.log({"valid/top_k": np.mean(top_k).item()}, step=global_step)
            self.model.accelerator.log({"valid/psm":  np.mean(psm).item()}, step=global_step)
            self.model.accelerator.trackers[0].writer.add_images("valid/img", 
                                                                torch.from_numpy(grid_image.astype(np.uint8)).unsqueeze(0), 
                                                                global_step, 
                                                                dataformats="NHWC")
            
        # For Memory Efficiency.
        del self.pipeline
        torch.cuda.empty_cache()

        if self.use_ema:
            self.model.ema_unet.restore(self.model.unet.parameters())
            # self.model.ema_unet.restore( self.unwrap_unet)
            # self.model.unet = self.model.acce


    def train(self):
        gpus = torch.cuda.device_count()
        mp.spawn(self._train, args=(gpus,), nprocs=gpus)
    
    
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
                            min_value=0, 
                            ddp=False)
        self.model_define(self.device, ddp=False) 
        self.model.from_pretrained(unet_path, cond_path, ema_path, ip_adaption_path)

        self.model.unet.eval()
        self.model.cond_models.eval()
        if self.ip_adapter_enabled:
            self.model.ip_adaption_modules.eval()


        if self.use_ema:
            self.model.ema_unet.store(self.model.unet.parameters())
            self.model.ema_unet.copy_to(self.model.unet.parameters())

        # Make Pipelines for diffusion models
        self.pipeline = EEGDiffusionPipeline(self.model.vae,
                                             self.model.cond_models,
                                             self.model.unet, 
                                             self.model.noise_scheduler,
                                             self.model.ip_adaption_modules)

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
        #     img = val_data["image"].to(self.device)
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
            img = val_data["image"].to(self.device)
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
        



