class SobelConv(nn.Module):
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

class MultiScaleGradientGenerator(nn.Module):
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

class GradientSemanticFusion(nn.Module):
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
