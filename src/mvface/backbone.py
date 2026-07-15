import torch
from torch import nn
from torchvision.models import resnet50


class RGBDPoseResNet50(nn.Module):
    """ResNet-50 with a 4-channel (RGBD) with pooling cropped and added 3 layers of deconv head 
    """

    def __init__(
        self,
        num_deconv_layers: int = 3,
        deconv_channels: int = 256,
    ) -> None:
        super().__init__()
        rn = resnet50(weights=None) # random initialized weight for now, may change in the future
        rn.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) # swap for 4 channel

        # Include Stage 1-5
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        # Add 3 layers of deconv
        self.deconv_head = self._build_deconv_head(
            in_channels=2048,
            num_layers=num_deconv_layers,
            out_channels=deconv_channels,
        )
        self.out_channels = deconv_channels

    @staticmethod
    def _build_deconv_head(in_channels: int, num_layers: int, out_channels: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    ch_in, out_channels, kernel_size=4, stride=2, padding=1, bias=False
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            ch_in = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_head(x)
        return x


class MultiViewBackbone(nn.Module):
    """run the RGBD backbone over all N views of each sample.
    
    Args:
        rgbd (B, N, 4, H, W): B samples, N views each
        
    Returns:
        features (B, N, C, H/4, W/4)  C = backbone.out_channels (256)
        
    The same weights process every view. So instead of looping over views, fold the N-view axis into the batch dimension, do one backbone pass, then unfold.
    """

    def __init__(self, deconv_channels: int = 256) -> None:
        super().__init__()
        self.backbone = RGBDPoseResNet50(deconv_channels=deconv_channels)
        self.out_channels = self.backbone.out_channels

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = rgbd.shape
        # folds, B*N images feed in at once
        x = rgbd.reshape(B*N, C, H, W)
        features = self.backbone(x)
        # unfold
        features = features.reshape(B, N, self.out_channels, features.shape[-2], features.shape[-1])

        return features
