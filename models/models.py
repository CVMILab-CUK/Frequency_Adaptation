# Default Library
import os
import numpy   as np
from copy   import deepcopy as copy
from typing import Any, Callable, Dict, List, Optional, Union
from einops.layers.torch import Rearrange


# Torch Library
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torchvision.utils import make_grid
from torch.utils.data  import DataLoader
from safetensors.torch import load_file

# Accelerator Libraries
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
# from torch.distributed.fsdp.sharding_strategy import ShardingStrategy


from transformers.utils import ContextManagers


# Diffusion Model

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module



from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor


class Model(nn.Module):
    def __init__(self,
                 # Baseline setting
                 model_id                        = "stabilityai/stable-diffusion-2-1",
                 report_to:str                   = "tensorboard",
                 output_dir                      = "./ckpt_dir",
                 logging_dir                     = "./log",

                 # Model optimization parameters
                 mixed_precision:str             = "fp16",
                 cfg_scale:float = 0.1,
                 gradient_accumulation_steps:int = 1,

                 # Model Setting
                 ip_adapter_token_num:int = 4,
                 out_seq:int              = 768,

                 # Train Pipeline Setting
                 learning_rate: int      = 1e-4,
                 beta1:int               = 0.9,
                 beta2:int               = 0.999,
                
                # Pretraining model setting
                 revision:str                    = None,
                 variant:str                     = None,#"non_ema",
                 non_ema_revision:str            = None,
                # Mymodel setting
                 use_ema:bool                    = False,
                 foreach_ema:bool                = True,
                 ):
        self.use_ema = use_ema
        self.cfg_scale = cfg_scale

        # Set Accelerator
        accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
        self.accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps,
                                  mixed_precision = mixed_precision,
                                  log_with = report_to,
                                  project_config=accelerator_project_config)

        self.accelerator.print = lambda *args, **kwargs: None

        

        # Set Diffusion
        self.noise_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        with ContextManagers(self.deepspeed_zero_init_disabled_context_manager()):
            self.vae = AutoencoderKL.from_pretrained(
                model_id, subfolder="vae", revision=revision, variant=variant,
            )

            self.unet = UNet2DConditionModel.from_pretrained(
                model_id, subfolder="unet", revision=non_ema_revision,
            )
        
        # Setting IP Adaptation Module
        self.ip_adaption_modules= nn.Sequential(
                nn.Linear(out_seq, ip_adapter_token_num * out_seq),
                Rearrange('b (n e) -> b n e', n=ip_adapter_token_num),
                nn.LayerNorm(out_seq))


        self.ip_adaption_modules.train()
        self.set_ip_adaption()
        
        # Model Freeze
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Gradient check pointing
        self.unet.enable_gradient_checkpointing()

        if use_ema:
            # self.ema_unet = UNet2DConditionModel.from_pretrained(
            #     model_id, subfolder="unet", revision=revision, variant=variant
            # )
            self.ema_unet = copy(self.unet)
            self.ema_unet = EMAModel(
                self.ema_unet.parameters(),
                model_cls=self.ema_unet,
                model_config=self.ema_unet.config,
                foreach=foreach_ema,
            )

        
        # Set Model parameters
        cond_params = []
        for proc in self.unet.attn_processors.values():
            proc.requires_grad = True
            cond_params += list(proc.parameters())

        cond_params += list(self.ip_adaption_modules.parameters())
        self.params = set(cond_params)

        total_params = sum(p.numel() for p in self.params)

        print(f"Number of Trainable Params: {total_params:,}")


        self.optimizer = torch.optim.AdamW(
                                            self.params,
                                            lr=learning_rate,
                                            betas=(beta1, beta2),
                                        )

    def set_ip_adaption(self):

        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():

            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)