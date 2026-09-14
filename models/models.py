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
from accelerate.utils import DeepSpeedPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
# from torch.distributed.fsdp.sharding_strategy import ShardingStrategy


from transformers.utils import ContextManagers


# Diffusion Model
import os
from huggingface_hub import login
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    login(token=_hf_token)
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


# CLIP Model
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


from .skip_injector import SkipInjector
from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, StandAloneAttnProcessor as SAProcessor
from .condition_encoder import ConditionEncoder        


class Model(nn.Module):
    def __init__(self,
                 # Baseline setting
                 model_id                        = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 report_to:str                   = "tensorboard",
                 output_dir                      = "./ckpt_dir",
                 logging_dir                     = "./log",

                 # Model optimization parameters
                 mixed_precision:str             = "fp16",
                 cfg_scale:float = 0.1,
                 gradient_accumulation_steps:int = 1,

                 # Model Setting
                 sa_plugin               = None,

                 # Train Pipeline Setting
                 learning_rate:int      = 1e-4,
                 beta1:int               = 0.9,
                 beta2:int               = 0.999,
                
                # Pretraining model setting
                 revision:str                    = None,
                 variant:str                     = None,#"non_ema",
                 non_ema_revision:str            = None,
                # Mymodel setting
                 use_ema:bool                    = False,
                 foreach_ema:bool                = True,

                 #Deep Speed Settings
                 ds_plugin:dict                  = None,
                 ):
        super().__init__()
        self.use_ema = use_ema
        self.cfg_scale = cfg_scale
        self.sa_plugin = sa_plugin

        # Set Accelerator
        accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
        self.accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps,
                                  deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_plugin) if ds_plugin else None,
                                  mixed_precision = mixed_precision,
                                  log_with = report_to,
                                  project_config=accelerator_project_config)

        self.accelerator.print = lambda *args, **kwargs: None

        

        # Set Diffusion
        self.noise_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")        
        self.clip_tokenizer  = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
     
        with ContextManagers(self.deepspeed_zero_init_disabled_context_manager()):
            self.vae = AutoencoderKL.from_pretrained(
                model_id, subfolder="vae", revision=revision, variant=variant,
            )           
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_id, subfolder="text_encoder"
            )

        self.unet = UNet2DConditionModel.from_pretrained(
                model_id, subfolder="unet", revision=non_ema_revision,
            )

        # Condition encoder: 'vae' reuses the frozen SD encoder (the original
        # design); 'conv' trains a small stem on the raw condition image, which
        # the VAE was measured to destroy at high cutoffs.
        self.cond_encoder_kind = str(getattr(sa_plugin, "cond_encoder", "vae"))
        # attn_inject: false trains the skip path alone (ralplan E18). The
        # cross-attention sites then keep the stock processor and nothing reads
        # an adapter latent, so no condition encoder is built either.
        self.attn_inject = bool(getattr(sa_plugin, "attn_inject", True))
        if not self.attn_inject:
            self.cond_encoder = None
            self.cond_channels = self.vae.config.latent_channels
        elif self.cond_encoder_kind == "conv":
            self.cond_encoder = ConditionEncoder(
                in_channels=3,
                out_channels=int(getattr(sa_plugin, "cond_channels", 4)),
                base=int(getattr(sa_plugin, "cond_base", 32)),
            )
            self.cond_channels = self.cond_encoder.out_channels
        else:
            self.cond_encoder = None
            self.cond_channels = self.vae.config.latent_channels

        # Setting IP Adaptation Module
        self.set_ip_adaption()
        
        # Model Freeze
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Gradient check pointing
        self.unet.enable_gradient_checkpointing()

        # Trainable parameters: the adapter projections only. A list (not a set)
        # so ordering is deterministic across runs -- optimizer state and EMA
        # both depend on it.
        cond_params = []
        for proc in self.unet.attn_processors.values():
            if isinstance(proc, SAProcessor):
                proc.requires_grad_(True)
                cond_params.extend(proc.parameters())
            else:
                proc.requires_grad_(False)

        if self.cond_encoder is not None:
            self.cond_encoder.requires_grad_(True)
            cond_params.extend(self.cond_encoder.parameters())

        # Optional ControlNet-style placement: residuals on every skip
        # connection instead of (or alongside) the cross-attention sites.
        self.skip_injector = None
        if bool(getattr(self.sa_plugin, "skip_inject", False)):
            spec, mid_spec = SkipInjector.spec_from_unet(self.unet)
            self.skip_injector = SkipInjector(
                spec, mid_spec,
                in_ch=3,
                base=int(getattr(self.sa_plugin, "skip_base", 32)),
                scale=float(getattr(self.sa_plugin, "skip_scale", 1.0)),
            )
            self.skip_injector.requires_grad_(True)
            cond_params.extend(self.skip_injector.parameters())
            print(f"SkipInjector: {len(spec)} down residuals + mid {mid_spec}, "
                  f"{sum(p.numel() for p in self.skip_injector.parameters()):,} params")

        self.params = cond_params
        total_params = sum(p.numel() for p in self.params)
        total_unet = sum(p.numel() for p in self.unet.parameters())
        print(f"Trainable (adapter) params: {total_params:,} "
              f"({100.0 * total_params / max(total_unet, 1):.4f}% of the UNet's {total_unet:,})")

        if use_ema:
            # EMA tracks the adapter only. Shadowing the full UNet would cost
            # ~3.4 GB to average parameters that never move.
            self.ema_unet = EMAModel(self.params, foreach=foreach_ema)
        else:
            self.ema_unet = None


        self.optimizer = torch.optim.AdamW(
                                            self.params,
                                            lr=learning_rate,
                                            betas=(beta1, beta2),
                                        )

    def set_ip_adaption(self):

        attn_procs = {}
        unet_sd = self.unet.state_dict()
        z_channels = self.cond_channels
        # print(z_channels)
        # raise ValueError
        for name in self.unet.attn_processors.keys():

            boc = self.unet.config.block_out_channels
            n_levels = len(boc)
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            # n_down: how many times the structure latent must be halved to
            # reach this block's spatial resolution.
            if name.startswith("mid_block"):
                hidden_size = boc[-1]
                n_down = n_levels - 1
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(boc))[block_id]
                n_down = n_levels - 1 - block_id
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = boc[block_id]
                n_down = block_id
            if cross_attention_dim is None or not self.attn_inject:
                attn_procs[name] = AttnProcessor()
            else:
                # layer_name = name.split(".processor")[0]
                # weights = {
                #     "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                #     "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                # }
                attn_procs[name] = SAProcessor(
                    z_channels = z_channels,
                    hidden_size=hidden_size, 
                    cross_attention_dim=self.cond_channels,
                    scale=self.sa_plugin.scale, 
                    use_vae=self.sa_plugin.use_vae, 
                    kernel_size=self.sa_plugin.kernel_size,
                    down_mode=self.sa_plugin.down_mode,
                    stem_channels=int(getattr(self.sa_plugin, "stem_channels", 0) or 0),
                    n_down=n_down,
                    zero_init_gate=bool(getattr(self.sa_plugin, "zero_init_gate", False)))
                # attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)
    
    ADAPTER_KEY = ".processor."

    def adapter_state_dict(self):
        """Just the trained tensors, not the frozen 3.4 GB UNet: the attention
        processors, plus the condition encoder when one is used."""
        sd = {k: v.detach().cpu().clone()
              for k, v in self.unet.state_dict().items() if self.ADAPTER_KEY in k}
        if self.cond_encoder is not None:
            for k, v in self.cond_encoder.state_dict().items():
                sd[f"cond_encoder.{k}"] = v.detach().cpu().clone()
        if self.skip_injector is not None:
            for k, v in self.skip_injector.state_dict().items():
                sd[f"skip_injector.{k}"] = v.detach().cpu().clone()
        return sd

    def save_adapter(self, path):
        from safetensors.torch import save_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(self.adapter_state_dict(), path)
        return path

    def load_adapter(self, path):
        raw = load_file(path)
        n_side = sum(1 for k in raw if k.startswith(("skip_injector.", "cond_encoder.")))
        sd = self._split_and_load_encoder(raw)
        missing, unexpected = self.unet.load_state_dict(sd, strict=False)
        loaded = [k for k in sd if self.ADAPTER_KEY in k]
        if self.attn_inject and not loaded:
            # A skip-only checkpoint loaded into a two-path model would leave
            # every attention site at its zero-initialised gate and still run.
            raise RuntimeError(f"{path} has no attention-processor tensors but the config enables the attention path")
        if not loaded and not n_side:
            raise RuntimeError(f"{path} contains no adapter tensors")
        if unexpected:
            raise RuntimeError(f"unexpected keys when loading adapter: {unexpected[:5]}")
        print(f"loaded {len(loaded)} attention-processor tensors and {n_side} "
              f"encoder/injector tensors from {path}")
        return self

    def _split_and_load_encoder(self, sd):
        inj = {k[len("skip_injector."):]: v for k, v in sd.items() if k.startswith("skip_injector.")}
        sd = {k: v for k, v in sd.items() if not k.startswith("skip_injector.")}
        if inj:
            if self.skip_injector is None:
                raise RuntimeError("checkpoint has a skip injector but the config does not enable one")
            self.skip_injector.load_state_dict(inj)
        elif self.skip_injector is not None:
            raise RuntimeError("config enables a skip injector but the checkpoint has none")

        enc = {k[len("cond_encoder."):]: v for k, v in sd.items() if k.startswith("cond_encoder.")}
        rest = {k: v for k, v in sd.items() if not k.startswith("cond_encoder.")}
        if enc:
            if self.cond_encoder is None:
                raise RuntimeError("checkpoint has a condition encoder but the config says cond_encoder: vae")
            self.cond_encoder.load_state_dict(enc)
        elif self.cond_encoder is not None:
            raise RuntimeError("config asks for a condition encoder but the checkpoint has none")
        return rest

    def from_pretrained(self, unet_path):
        """Accepts either an adapter-only checkpoint or a full UNet dump."""
        sd = self._split_and_load_encoder(load_file(unet_path))
        missing, unexpected = self.unet.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in {unet_path}: {unexpected[:5]}")
        n_adapter = sum(1 for k in sd if self.ADAPTER_KEY in k)
        print(f"loaded {len(sd)} tensors ({n_adapter} adapter) from {unet_path}")
        return self

    def deepspeed_zero_init_disabled_context_manager(self):
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return deepspeed_plugin.zero3_init_context_manager(enable=False)