from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mvface.backbone import MultiViewBackbone
from mvface.decoder import MultiViewDecoder


class MultiViewLandmark3D(nn.Module):
    """rgbd (B,N,4,H,W) -> MultiViewBackbone -> feats (B,N,256,h,w) -> MultiViewDecoder -> per-layer 3D + 2D landmark predictions
    """
    
    def __init__(self, assets_dir, num_layers: int = 4,
                 use_depth: bool = True, img_size: int | None = None) -> None:
        super().__init__()
        self.backbone = MultiViewBackbone()
        self.decoder = MultiViewDecoder.from_assets(assets_dir, num_layers=num_layers)
        self.use_depth = use_depth # flag to turn the fourth channel of backbone to zero, RGBD vs RGB
        self.img_size = img_size

    def forward(self, rgbd: torch.Tensor, proj: torch.Tensor, image_hw):
        B, N, C, H, W = rgbd.shape
        if not self.use_depth:
            rgbd = rgbd.clone()
            rgbd[:, :, 3] = 0.0
        x = rgbd
        if self.img_size is not None and self.img_size != H:    # optinal resize the image according to input
            x = F.interpolate(rgbd.reshape(B * N, C, H, W), size=self.img_size,
                              mode="bilinear", align_corners=False)
            x = x.reshape(B, N, C, self.img_size, self.img_size)
        feats = self.backbone(x)
        
        return self.decoder(feats, proj, image_hw)
