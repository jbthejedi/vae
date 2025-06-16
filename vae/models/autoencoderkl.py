import torch
import sys
import lpips
import wandb
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms as T

from typing import Callable
from omegaconf import OmegaConf
from PIL import Image


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.act1  = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.act2  = nn.SiLU()
        
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels!=out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.skip(x)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=4):
        super().__init__()
        self.res1 = ResnetBlock(in_ch, out_ch, num_groups)
        self.res2 = ResnetBlock(out_ch, out_ch, num_groups)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x_down = self.down(x)
        return x_down, x  # x_down is half spatial size; x is the skip


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=4):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")
        self.res1 = ResnetBlock(in_ch, out_ch, num_groups)
        self.res2 = ResnetBlock(out_ch, out_ch, num_groups)

    def forward(self, x, skip):
        x = self.up(x)                              # (B, in_ch/2, H×2, W×2)
        x = torch.cat([x, skip], dim=1)             # (B, in_ch/2 + skip_ch, H×2, W×2)
        x = self.res1(x)
        x = self.res2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(4, channels, eps=1e-6, affine=True)  # small num_groups for testing
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H*W).permute(0,2,1)  # B,HW,C
        k = self.k(h).reshape(B, C, H*W)                 # B,C,HW
        v = self.v(h).reshape(B, C, H*W).permute(0,2,1)  # B,HW,C

        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = torch.softmax(attn, dim=-1)

        h_attn = torch.bmm(attn, v).permute(0,2,1).reshape(B, C, H, W)
        return x + self.proj(h_attn)

class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res1 = ResnetBlock(channels, channels, num_groups=4)
        self.attn = AttentionBlock(channels)
        self.res2 = ResnetBlock(channels, channels, num_groups=4)

    def forward(self, x):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class LatentBottleneck(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super().__init__()
        self.conv_mu     = nn.Conv2d(in_channels, latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(in_channels, latent_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        mu     = self.conv_mu(x)          # → (B, latent_channels, H, W)
        logvar = self.conv_logvar(x)      # → (B, latent_channels, H, W)
        std    = torch.exp(0.5 * logvar)  # standard deviation
        eps    = torch.randn_like(std)    # random noise
        z      = mu + eps * std           # reparameterized latent
        return z, mu, logvar


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            DownEncoderBlock2D(3, 8),
            DownEncoderBlock2D(8, 16),
            DownEncoderBlock2D(16, 16),
            DownEncoderBlock2D(16, 16),
        ])
    def forward(self, x):
        hiddens = []
        for block in self.blocks:
            x, skip = block(x)
            hiddens.append(skip)
        return x, hiddens

import torch
import torch.nn as nn


class AutoencoderKLSmall(nn.Module):
    def __init__(self, in_channels=3, base_channels=(8,16,16,16), latent_channels=4, num_groups=2):
        super().__init__()
        # Encoder
        chans = [in_channels] + list(base_channels)
        self.encoder = nn.ModuleList([
            DownEncoderBlock2D(chans[i], chans[i+1], num_groups=num_groups)
            for i in range(len(base_channels))
        ])
        mid_ch = base_channels[-1]

        # Mid-block
        self.mid = nn.Sequential(
            ResnetBlock(mid_ch, mid_ch, num_groups=num_groups),
            AttentionBlock(mid_ch),
            ResnetBlock(mid_ch, mid_ch, num_groups=num_groups),
        )

        # Bottleneck
        self.bottleneck = LatentBottleneck(mid_ch, latent_channels)
        self.scaling = 0.18215
        self.conv_proj = nn.Conv2d(latent_channels, mid_ch, kernel_size=1)

        # Decoder (reverse of encoder)
        skip_chs = list(base_channels[::-1])
        decoder_ins  = [mid_ch] + skip_chs[:-1]
        decoder_outs = skip_chs
        self.decoder = nn.ModuleList([
            UpDecoderBlock2D(in_ch + skip_ch, out_ch, num_groups=num_groups)
            for in_ch, skip_ch, out_ch in zip(decoder_ins, skip_chs, decoder_outs)
        ])
        # Final RGB conv
        self.conv_out = nn.Conv2d(base_channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # ---- Encode ----
        skips = []
        for down in self.encoder:
            x, skip = down(x)
            skips.append(skip)

        # ---- Mid + Latent ----
        x = self.mid(x)
        z, mu, logvar = self.bottleneck(x)
        x = z * self.scaling
        x = self.conv_proj(x)

        # ---- Decode ----
        for up, skip in zip(self.decoder, reversed(skips)):
            x = up(x, skip)

        # ---- Final reconstruction ----
        recon = self.conv_out(x)
        return recon, mu, logvar