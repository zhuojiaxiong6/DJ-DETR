"""
BCPN: Bidirectional Context Pyramid Network
Part of DJ-DETR for tomato leaf disease detection

This module produces multi-scale lesion features through feature sequence 
interaction and multi-kernel depthwise convolution, effectively reducing 
cross-scale semantic gaps.

Components:
- ADown: Adaptive Downsampling module
- CoreFeatureBlock: Core computation unit for bidirectional feature interaction
- BCPN: Main Bidirectional Context Pyramid Network
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


class ADown(nn.Module):
    """
    Adaptive Downsampling Module
    
    Performs downsampling with channel adjustment through a combination
    of average pooling and convolution operations.
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class CoreFeatureBlock(nn.Module):
    """
    CoreFeatureBlock: Core computation unit for BCPN
    
    Performs multi-scale feature interaction through:
    1. Branch fusion from three different scales (upsample, identity, downsample)
    2. Multi-kernel depthwise convolution for scale-aware processing
    3. Residual connection for gradient flow
    
    Args:
        inc: List of input channels [c1, c2, c3] for three scales
        k: Tuple of kernel sizes for multi-scale convolution
        e: Expansion ratio for intermediate channels
    """
    def __init__(self, inc, k=(5, 7, 9, 11), e=0.5):
        super().__init__()
        c_ = int(inc[1] * e)
        
        # Three branches for different scales
        self.branch_1 = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            Conv(inc[0], c_, 1)
        )  # Upsample from smaller resolution
        self.branch_2 = Conv(inc[1], c_, 1) if e != 1 else nn.Identity()  # Same resolution
        self.branch_3 = ADown(inc[2], c_)  # Downsample from larger resolution
        
        # Multi-kernel depthwise convolutions for scale-aware processing
        self.scales = nn.ModuleList(
            nn.Conv2d(c_ * 3, c_ * 3, kernel_size=ki, padding=autopad(ki), groups=c_ * 3) 
            for ki in k
        )
        
        # Fusion layer
        self.fusion = Conv(c_ * 3, c_ * 3)
    
    def forward(self, x):
        """
        Args:
            x: List of three feature maps [x1, x2, x3] from different scales
        Returns:
            Fused multi-scale feature map
        """
        x1, x2, x3 = x
        
        # Concatenate features from three branches
        features = torch.cat([
            self.branch_1(x1), 
            self.branch_2(x2), 
            self.branch_3(x3)
        ], 1)
        
        # Multi-scale convolution with residual connections
        s = features
        for op in self.scales:
            s = s + op(features)
        
        # Final fusion with residual
        return features + self.fusion(s)


class BCPN(nn.Module):
    """
    BCPN: Bidirectional Context Pyramid Network
    
    A feature pyramid architecture that creates a recursive network of 
    information highway through bidirectional feature flow. Ensures that 
    small, early lesions survive the feature extraction process.
    
    Features:
    - Bidirectional information flow between scales
    - Multi-kernel aggregation for scale-invariant representation
    - Efficient cross-scale semantic gap reduction
    
    Args:
        channels_list: List of channel dimensions for each pyramid level
        k: Tuple of kernel sizes for CoreFeatureBlock
        e: Expansion ratio
    """
    def __init__(self, channels_list, k=(5, 7, 9, 11), e=0.5):
        super().__init__()
        self.num_levels = len(channels_list)
        
        # Top-down pathway
        self.top_down_blocks = nn.ModuleList()
        for i in range(self.num_levels - 2):
            inc = [channels_list[i], channels_list[i+1], channels_list[i+2]]
            self.top_down_blocks.append(CoreFeatureBlock(inc, k, e))
        
        # Bottom-up pathway
        self.bottom_up_blocks = nn.ModuleList()
        for i in range(self.num_levels - 2):
            c_ = int(channels_list[i+1] * e) * 3
            self.bottom_up_blocks.append(Conv(c_, channels_list[i+1], 1))
        
        # Output projections
        self.output_convs = nn.ModuleList(
            Conv(channels_list[i], channels_list[i], 3) 
            for i in range(self.num_levels)
        )

    def forward(self, features):
        """
        Args:
            features: List of feature maps from backbone [P3, P4, P5, ...]
        Returns:
            List of enhanced feature maps
        """
        # Top-down pathway
        td_features = list(features)
        for i, block in enumerate(self.top_down_blocks):
            if i + 2 < len(td_features):
                input_feats = [td_features[i], td_features[i+1], td_features[i+2]]
                td_features[i+1] = block(input_feats)
        
        # Bottom-up pathway
        bu_features = td_features
        for i in range(len(self.bottom_up_blocks) - 1, -1, -1):
            bu_features[i+1] = self.bottom_up_blocks[i](bu_features[i+1])
        
        # Output projections
        outputs = []
        for i, conv in enumerate(self.output_convs):
            outputs.append(conv(bu_features[i]))
        
        return outputs


if __name__ == "__main__":
    # Test CoreFeatureBlock
    print("Testing CoreFeatureBlock...")
    block = CoreFeatureBlock(inc=[128, 256, 512], k=(5, 7, 9, 11), e=0.5)
    x1 = torch.randn(1, 128, 20, 20)  # Small resolution, fewer channels
    x2 = torch.randn(1, 256, 40, 40)  # Medium resolution
    x3 = torch.randn(1, 512, 80, 80)  # Large resolution, more channels
    output = block([x1, x2, x3])
    print(f"CoreFeatureBlock output shape: {output.shape}")
    
    # Test BCPN
    print("\nTesting BCPN...")
    bcpn = BCPN(channels_list=[64, 128, 256, 512], k=(5, 7, 9, 11), e=0.5)
    features = [
        torch.randn(1, 64, 160, 160),
        torch.randn(1, 128, 80, 80),
        torch.randn(1, 256, 40, 40),
        torch.randn(1, 512, 20, 20)
    ]
    outputs = bcpn(features)
    print("BCPN output shapes:")
    for i, out in enumerate(outputs):
        print(f"  Level {i}: {out.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in bcpn.parameters()):,}")
