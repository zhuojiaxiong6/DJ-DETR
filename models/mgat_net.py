"""
MGAT-Net: Multi-Scale Gradient-Aware Transfer Network
Part of DJ-DETR for tomato leaf disease detection

This module encodes gradient cues using Sobel operators to enhance
localization stability for small or blurred lesions.

Components:
- GPG (Gradient Pyramid Generator): Generates multi-scale gradient features
- ESFM (Edge-Semantic Fusion Module): Fuses gradient and semantic features
"""

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Auto padding calculation"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SobelConv(nn.Module):
    """
    Sobel gradient operator for edge detection
    
    Applies horizontal and vertical Sobel kernels to extract gradient information.
    """
    def __init__(self, c):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        
        k_x = sobel_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        k_y = sobel_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        
        self.conv_x = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.conv_y = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        
        self.conv_x.weight = nn.Parameter(k_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(k_y, requires_grad=False)

    def forward(self, x):
        return self.conv_x(x) + self.conv_y(x)


class GPG(nn.Module):
    """
    GPG: Gradient Pyramid Generator
    
    Generates multi-scale gradient features using Sobel operators.
    Creates a pyramid of edge features by successive max pooling operations.
    """
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        self.extractor = SobelConv(in_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.projections = nn.ModuleList([
            Conv(in_channels, c, 1) for c in out_channels_list
        ])

    def forward(self, x):
        feat = self.extractor(x)
        features = []
        for i, layer in enumerate(self.projections):
            feat = self.pool(feat)
            features.append(layer(feat))
        return features


class ESFM(nn.Module):
    """
    ESFM: Edge-Semantic Fusion Module
    
    Fuses gradient features with semantic features from backbone stages.
    Integrates edge information with high-level semantic representations.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        total_in = sum(in_channels_list) if isinstance(in_channels_list, list) else in_channels_list
        
        self.reduce = Conv(total_in, mid_channels, 1)
        self.enhance = nn.Sequential(
            Conv(mid_channels, mid_channels, 3),
            Conv(mid_channels, out_channels, 1)
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.enhance(self.reduce(x))


class MGATNet(nn.Module):
    """
    MGAT-Net: Multi-Scale Gradient-Aware Transfer Network
    
    A backbone architecture that explicitly uses edge information as it 
    progresses up the feature hierarchy. Combines GPG (Gradient Pyramid 
    Generator) and ESFM (Edge-Semantic Fusion Module) for gradient-aware
    feature extraction.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels_list: List of output channels for each scale
    """
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        self.gpg = GPG(in_channels, out_channels_list)
        self.esfm_modules = nn.ModuleList([
            ESFM(c * 2, c) for c in out_channels_list
        ])

    def forward(self, x, semantic_features):
        """
        Args:
            x: Input image tensor
            semantic_features: List of semantic feature maps from backbone
        Returns:
            List of gradient-enhanced feature maps
        """
        gradient_features = self.gpg(x)
        outputs = []
        for i, (grad_feat, sem_feat) in enumerate(zip(gradient_features, semantic_features)):
            fused = self.esfm_modules[i]([grad_feat, sem_feat])
            outputs.append(fused)
        return outputs


if __name__ == "__main__":
    # Test the module
    model = MGATNet(in_channels=3, out_channels_list=[64, 128, 256])
    x = torch.randn(1, 3, 640, 640)
    semantic_feats = [
        torch.randn(1, 64, 80, 80),
        torch.randn(1, 128, 40, 40),
        torch.randn(1, 256, 20, 20)
    ]
    outputs = model(x, semantic_feats)
    for i, out in enumerate(outputs):
        print(f"Output {i}: {out.shape}")
