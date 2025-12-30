"""
RFAM: Retentive Feature Aggregation Module
Part of DJ-DETR for tomato leaf disease detection

This module builds global dependencies while filtering background noise
using the Retention mechanism from RetNet.

Components:
- RelPos2d: 2D Relative Position Encoding
- RetBlock: Retention Block with spatial decay
- RFAM: Main Retentive Feature Aggregation Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        return self.cv2(torch.cat(y, 1))


class RelPos2d(nn.Module):
    """
    2D Relative Position Encoding
    
    Generates relative position embeddings for spatial retention computation.
    """
    def __init__(self, embed_dim, num_heads, initial_value=2, heads_range=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.initial_value = initial_value
        self.heads_range = heads_range
        
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)
        
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads) / num_heads))
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H, W):
        """Generate 2D spatial decay matrix"""
        grid_h = torch.arange(H, device=self.decay.device)
        grid_w = torch.arange(W, device=self.decay.device)
        
        index_h = grid_h.view(1, -1, 1) - grid_h.view(1, 1, -1)
        index_w = grid_w.view(1, -1, 1) - grid_w.view(1, 1, -1)
        
        decay_h = self.decay.view(-1, 1, 1) * index_h.abs().float()
        decay_w = self.decay.view(-1, 1, 1) * index_w.abs().float()
        
        decay_h = decay_h.exp()
        decay_w = decay_w.exp()
        
        return decay_h, decay_w

    def generate_1d_decay(self, l):
        """Generate 1D decay for chunkwise processing"""
        index = torch.arange(l, device=self.decay.device).float()
        mask = index.view(1, -1, 1) - index.view(1, 1, -1)
        mask = mask.abs()
        decay = self.decay.view(-1, 1, 1) * mask
        return decay.exp()

    def forward(self, shape, chunkwise_recurrent=False):
        """
        Args:
            shape: (H, W) spatial dimensions
            chunkwise_recurrent: Whether to use chunkwise processing
        Returns:
            Position encoding tensors
        """
        H, W = shape
        if chunkwise_recurrent:
            decay_h = self.generate_1d_decay(H)
            decay_w = self.generate_1d_decay(W)
        else:
            decay_h, decay_w = self.generate_2d_decay(H, W)
        return decay_h, decay_w


class RetBlock(nn.Module):
    """
    Retention Block
    
    Implements the retention mechanism with spatial decay for 
    capturing long-range dependencies in feature maps.
    """
    def __init__(self, retention, dim, num_heads, value_dim, expansion_ratio=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.value_dim = value_dim
        self.expansion_ratio = expansion_ratio
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion_ratio),
            nn.GELU(),
            nn.Linear(dim * expansion_ratio, dim)
        )
        
        self.gamma = nn.Parameter(torch.ones(dim))

    def retention(self, q, k, v, decay):
        """
        Compute retention attention with spatial decay
        
        Args:
            q, k, v: Query, Key, Value tensors
            decay: Spatial decay matrix
        """
        decay_h, decay_w = decay
        B, H, W, C = q.shape
        
        q = q.view(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        v = v.view(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        
        # Retention along height
        qk_h = torch.einsum('bnhwc,bnHwc->bnhHw', q, k)
        qk_h = qk_h * decay_h.unsqueeze(0).unsqueeze(-1)
        out_h = torch.einsum('bnhHw,bnHwc->bnhwc', qk_h, v)
        
        # Retention along width
        qk_w = torch.einsum('bnhwc,bnhWc->bnhwW', q, k)
        qk_w = qk_w * decay_w.unsqueeze(0).unsqueeze(2)
        out_w = torch.einsum('bnhwW,bnhWc->bnhwc', qk_w, v)
        
        out = (out_h + out_w) / 2
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        
        return out

    def forward(self, x, incremental_state=None, chunkwise_recurrent=False, pe=None):
        """
        Args:
            x: Input tensor (B, H, W, C)
            incremental_state: For incremental decoding (not used)
            chunkwise_recurrent: Whether to use chunkwise processing
            pe: Position encoding from RelPos2d
        """
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        x = self.retention(q, k, v, pe)
        x = self.proj(x)
        x = residual + x * self.gamma
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class RFAM(C2f):
    """
    RFAM: Retentive Feature Aggregation Module
    
    Uses the Retention mechanism to build global dependencies while 
    filtering background noise without sacrificing semantic coherence.
    
    The retention mechanism provides:
    - Long-range spatial dependency modeling
    - Decay-modulated attention for noise filtering
    - Efficient computation compared to standard attention
    
    Args:
        c1: Input channels
        c2: Output channels
        n: Number of retention blocks
        retention: Retention mode ('chunk' or 'whole')
        num_heads: Number of attention heads
        shortcut: Whether to use shortcut connection
        g: Groups for convolution
        e: Expansion ratio
    """
    def __init__(self, c1, c2, n=1, retention='chunk', num_heads=8, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.mode = retention
        self.pe = RelPos2d(self.c, num_heads, 2, 4)
        self.fusion_layers = nn.ModuleList(
            RetBlock(retention, self.c, num_heads, self.c) for _ in range(n)
        )

    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            Aggregated feature map with global dependencies
        """
        b, c, h, w = x.size()
        pe = self.pe((h, w), chunkwise_recurrent=self.mode == 'chunk')
        
        y = list(self.cv1(x).chunk(2, 1))
        
        for layer in self.fusion_layers:
            # Permute to (B, H, W, C) for retention computation
            feat = y[-1].permute(0, 2, 3, 1)
            feat = layer(feat, None, self.mode == 'chunk', pe)
            # Permute back to (B, C, H, W)
            y.append(feat.permute(0, 3, 1, 2))
        
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    # Test the module
    model = RFAM(c1=256, c2=256, n=2, retention='chunk', num_heads=8)
    x = torch.randn(1, 256, 40, 40)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
