# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
# Original Code : https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        self.attn_map = ip_attention_probs
        ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            #print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


## for controlnet
class CNAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, num_tokens=4):
        self.num_tokens = num_tokens

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs,):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CNAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, num_tokens=4):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.num_tokens = num_tokens

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class StandAloneAttnProcessor(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        use_vae (`bool`, defaults to True):
            Whether to use VAE features or not.
        kernel_size (`int`, defaults to 3):
            The kernel size for the sliding window attention.
        down_mode (`str`, defaults to "avg"):
            The downsampling mode for sketch features. Options are "avg", "max", "conv".)
    """

    def __init__(self, 
                 z_channels,
                 hidden_size, 
                 cross_attention_dim=None, 
                 scale=1.0, 
                 use_vae:bool=True, 
                 kernel_size: int = 3,
                 down_mode:str="avg",
                 stem_channels: int = 0,
                 n_down: int = 4,
                 zero_init_gate: bool = False) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale  # assuming fixed spatial size for sketch features
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 # 1 for kernel_size=3

        # [핵심 수정 1] 스케치용 Key/Value 프로젝션
        # 스케치 피처(ip_hidden_states)를 받아서 K, V로 변환합니다.
        # 입력이 이미 채널이 맞춰진 Feature Map이라고 가정합니다.
        in_ch = cross_attention_dim or hidden_size
        if use_vae:
            # `stem_channels == 0` keeps the minimal 1x1 projection (~0.1M params
            # over the whole UNet). A positive value inserts a small non-linear
            # stem so adapter capacity can be ablated independently of the rest
            # of the design.
            def proj():
                if stem_channels and stem_channels > 0:
                    return nn.Sequential(
                        nn.Conv2d(in_ch, stem_channels, kernel_size=3, padding=1, bias=True),
                        nn.SiLU(),
                        nn.Conv2d(stem_channels, hidden_size, kernel_size=1, bias=False),
                    )
                return nn.Conv2d(in_ch, hidden_size, kernel_size=1, bias=False)
            self.to_k_sketch = proj()
            self.to_v_sketch = proj()
        else:
            self.to_k_sketch = nn.Linear(in_ch, hidden_size, bias=False)
            self.to_v_sketch = nn.Linear(in_ch, hidden_size, bias=False)

        # Zero-initialised per-channel output gate.
        #
        # Without it the adapter injects a randomly-initialised signal into a
        # frozen UNet from step 0, and early training goes into undoing that
        # rather than into learning structure. ControlNet's zero convolutions,
        # LoRA's B=0 and IP-Adapter all start their injection at exactly zero
        # for this reason.
        #
        # A per-channel vector rather than a scalar, and applied *after* the
        # output projection: with a single scalar at zero, `to_k_sketch` and
        # `to_v_sketch` would receive zero gradient too and never start
        # learning. Here d(loss)/d(gate) = grad_out * out_spatial is non-zero
        # at init, so the gate lifts off first and the projections follow.
        # Cost is `hidden_size` parameters per site (12,480 in total).
        self.zero_init_gate = bool(zero_init_gate)
        if self.zero_init_gate:
            self.out_gate = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.out_gate = None

        # Exactly the number of halvings this attention site needs, no more.
        # Allocating a fixed 4 gave every processor dead convolutions that never
        # saw a gradient but still collected optimizer and EMA state, and
        # inflated the reported trainable-parameter count.
        self.n_down = int(n_down)
        if down_mode == "avg":
            self.downsample = nn.ModuleList(
                [nn.AvgPool2d(kernel_size=2, stride=2) for _ in range(self.n_down)])
        elif down_mode == "max":
            self.downsample = nn.ModuleList(
                [nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(self.n_down)])
        elif down_mode == "conv":
            self.downsample = nn.ModuleList(
                [nn.Conv2d(z_channels, z_channels, kernel_size=2, stride=2, padding=0)
                 for _ in range(self.n_down)])
        elif down_mode == "none":
            self.downsample = nn.ModuleList()
        else:
            raise ValueError(f"Unsupported down_mode: {down_mode}")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
        ip_hidden_states=None, # 여기가 [Sketch Spatial Feature]가 들어올 자리입니다.
    ):
        
        
        target_dtype = next(self.to_k_sketch.parameters()).dtype
        hidden_states = hidden_states.to(dtype=target_dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)

        if ip_hidden_states is not None:
            ip_hidden_states = ip_hidden_states.to(dtype=target_dtype)
            
        residual = hidden_states
        if ip_hidden_states is not None:
            # 1. 차원 준비
            _, _, ori_h,ori_w = ip_hidden_states.shape  # (B, L_ip, D)
            b, seq_len, c = hidden_states.shape
            h = w = int(seq_len ** 0.5)
            # ip_hidden_states = ip_hidden_states.to(dtype=target_dtype)
            
            # 2. Bring the structure latent to this block's spatial resolution.
            n_needed = max(0, int(round(float(np.log2(ori_h / h)))))
            if n_needed != len(self.downsample):
                raise RuntimeError(
                    f"SAProcessor needs {n_needed} halvings ({ori_h}->{h}) but was "
                    f"built with {len(self.downsample)}; n_down was set wrong.")
            for down in self.downsample:
                ip_hidden_states = down(ip_hidden_states)

            # 3. K, V 생성 (Conv2d라 Rearrange 없이 바로 통과)
            # (B, 4, H, W) -> (B, Dim, H, W)
            k_spatial = self.to_k_sketch(ip_hidden_states)
            v_spatial = self.to_v_sketch(ip_hidden_states)

            # 4. Q 생성 (기존 Linear 사용) 및 차원 정리
            # (B, L, D) -> (B, Heads, L, 1, Head_Dim) : Einsum 준비 완료
            query = attn.to_q(hidden_states)
            query = rearrange(query, 'b l (heads d) -> b heads l 1 d', heads=attn.heads)

            # 5. Unfold (Sliding Window)
            # 입력: (B, Dim, H, W) -> 출력: (B, Dim*9, L)
            k_patches = F.unfold(k_spatial, kernel_size=self.kernel_size, padding=self.padding)
            v_patches = F.unfold(v_spatial, kernel_size=self.kernel_size, padding=self.padding)

            # 6. Unfold 결과 정리 (Rearrange 한 번만 사용)
            # (B, Heads*D*9, L) -> (B, Heads, L, 9, D)
            k_local = rearrange(k_patches, 'b (heads d k_sq) l -> b heads l k_sq d', heads=attn.heads, k_sq=self.kernel_size**2)
            v_local = rearrange(v_patches, 'b (heads d k_sq) l -> b heads l k_sq d', heads=attn.heads, k_sq=self.kernel_size**2)

            # 7. Attention Calculation (Einsum)
            # Q(1, D) x K(9, D)^T -> Score(1, 9)
            sim = torch.einsum('b h l i d, b h l j d -> b h l i j', query, k_local)
            sim = sim * (query.shape[-1] ** -0.5)
            attn_probs = sim.softmax(dim=-1)

            # Score(1, 9) x V(9, D) -> Out(1, D)
            out_spatial = torch.einsum('b h l i j, b h l j d -> b h l i d', attn_probs, v_local)
            
            # 8. Output 정리 (차원 복구 + Projection)
            out_spatial = rearrange(out_spatial, 'b h l 1 d -> b l (h d)')
            out_spatial = attn.to_out[0](out_spatial)
            out_spatial = attn.to_out[1](out_spatial)

            if self.out_gate is not None:
                out_spatial = out_spatial * self.out_gate.to(out_spatial.dtype)

            # Structure informs the text query below; it is added to the layer's
            # output once, in the return statement, not folded in here as well.
            hidden_states = hidden_states + self.scale * out_spatial
       # ----------------------------------------------------------------
        # Phase 2: Standard Text Cross-Attention
        # "구조가 잡힌 Latent에 텍스트 의미를 입힌다"
        # ----------------------------------------------------------------
        # *기존 AttnProcessor 로직 그대로*
        
        # 업데이트된 hidden_states로 다시 Query를 만듭니다.
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Text cross-attention via SDPA (flash/mem-efficient kernels) instead of
        # materialising the full attention matrix with head_to_batch_dim + bmm.
        b_, q_len, _ = hidden_states.shape
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key.shape[1], b_)
            attention_mask = attention_mask.view(b_, attn.heads, -1, attention_mask.shape[-1])

        query = query.view(b_, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(b_, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(b_, -1, attn.heads, head_dim).transpose(1, 2)

        text_out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        text_out = text_out.transpose(1, 2).reshape(b_, -1, attn.heads * head_dim)
        text_out = text_out.to(query.dtype)

        # linear proj & dropout
        text_out = attn.to_out[0](text_out)
        text_out = attn.to_out[1](text_out)

        # Return only what this attention layer contributes.
        #
        # SD1.5 cross-attention has `residual_connection = False`, which means
        # BasicTransformerBlock adds the residual itself:
        #     hidden_states = attn_output + hidden_states
        # Returning `hidden_states + text_out` therefore injected norm2(x) into
        # the residual stream a second time, at all 16 cross-attention layers
        # (measured: ||norm2(x)|| = 7.19 against a residual of 17.88). The
        # adapter then spent its whole capacity learning to cancel that term
        # instead of carrying structure -- which is why disabling it collapsed
        # the model (FID 71 -> 333) while structure consistency stayed at ~0.
        #
        # The structure signal is still added residually, as intended, and it
        # still shapes the text query above; only the spurious copy of the
        # input is gone.
        out = text_out
        if ip_hidden_states is not None:
            out = out + self.scale * out_spatial
        if attn.residual_connection:
            out = out + residual

        return out / attn.rescale_output_factor
