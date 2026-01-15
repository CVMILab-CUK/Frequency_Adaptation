import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math

class GpuFixedFFT(nn.Module):
    def __init__(self, fft_scale: float = 0.5):
        super().__init__()
        assert fft_scale >=0, f"fft_scale must be positive."

        self.fft_scale = fft_scale


    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if self.fft_scale == 0:
            return x
        
        B, C, H, W = x.shape
        device = x.device

        D0 = (min(H, W) / 2.0) * self.fft_scale


        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device), 
                            torch.arange(W, device=device), indexing='ij')
        
        # D shape: (H, W)
        D = torch.sqrt((y - cy)**2 + (x - cx)**2)

        low_pass_mask = torch.exp(- (D**2) / (2 * (D0**2)))
        high_pass_mask = 1.0 - low_pass_mask
        

        high_pass_mask_b = high_pass_mask.view(1, 1, H, W).repeat(B, C, 1, 1)

        img_fft = torch.fft.fft2(x)
        img_fft_shifted = torch.fft.fftshift(img_fft, dim=(-2, -1))

        filtered_fft_shifted = img_fft_shifted * high_pass_mask_b

        filtered_fft = torch.fft.ifftshift(filtered_fft_shifted, dim=(-2, -1))
        filtered_img = torch.fft.ifft2(filtered_fft)

        filtered_img_real = filtered_img.real # (B, C, H, W)
        
        min_val = torch.amin(filtered_img_real, dim=(-3, -2, -1), keepdim=True)
        max_val = torch.amax(filtered_img_real, dim=(-3, -2, -1), keepdim=True)
        
        filtered_img_normalized = (filtered_img_real - min_val) / (max_val - min_val + 1e-6)

        return filtered_img_normalized
    


class GpuRandomFFT(nn.Module):
    r"""
    Get
    """
    def __init__(self, fft_scale_range=(0.1, 0.5)):
        super().__init__()
        assert len(fft_scale_range) == 2, "fft_scale_range must be a tuple of (min, max)"
        assert fft_scale_range[0] >= 0.0, "min scale (range[0]) must be 0.0 or positive."
        assert fft_scale_range[1] <= 1.0, "max scale (range[1]) must be 1.0 or less."
        assert fft_scale_range[0] <= fft_scale_range[1], "min scale must be less than or equal to max scale."
        
        self.fft_scale_range = fft_scale_range       


    @torch.no_grad() 
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        device = x.device

        rand_tensor = torch.rand(B, device=device)
        fft_scale = rand_tensor * (self.fft_scale_range[1] - self.fft_scale_range[0]) + self.fft_scale_range[0]
        skip_mask = (fft_scale == 0.0).view(B, 1, 1, 1)
        fft_scale_safe = torch.clamp(fft_scale, min=1e-9)
        fft_scale_b = fft_scale.view(B, 1, 1, 1)

        D0 = (min(H, W) / 2.0) * fft_scale_b

        cy, cx = H // 2, W // 2
        y, x_grid = torch.meshgrid(torch.arange(H, device=device),  
                                 torch.arange(W, device=device), indexing='ij')
        
        D = torch.sqrt((y - cy)**2 + (x_grid - cx)**2)
        D_b = D.unsqueeze(0).unsqueeze(0)

        low_pass_mask = torch.exp(- (D_b**2) / (2 * (D0**2)))
        high_pass_mask = 1.0 - low_pass_mask
        
        high_pass_mask_c = high_pass_mask.repeat(1, C, 1, 1)

        img_fft = torch.fft.fft2(x)
        img_fft_shifted = torch.fft.fftshift(img_fft, dim=(-2, -1))

        filtered_fft_shifted = img_fft_shifted * high_pass_mask_c

        filtered_fft = torch.fft.ifftshift(filtered_fft_shifted, dim=(-2, -1))
        filtered_img = torch.fft.ifft2(filtered_fft)

        filtered_img_real = filtered_img.real 
        
        min_val = torch.amin(filtered_img_real, dim=(-3, -2, -1), keepdim=True)
        max_val = torch.amax(filtered_img_real, dim=(-3, -2, -1), keepdim=True)
        

        filtered_img_normalized = (filtered_img_real - min_val) / (max_val - min_val + 1e-6)
        

        output = torch.where(skip_mask, x, filtered_img_normalized)

        return output