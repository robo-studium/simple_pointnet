import torch
import torch.nn as nn
import torch.nn.functional as F


# =================================================
# Basic Blocks
# =================================================


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# =================================================
# Bottleneck Axial Attention
# =================================================


class BottleneckAxialAttention(nn.Module):
    """
    Axial Self-Attention along width (yaw direction only)
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Collapse vertical dimension (pitch)
        x_mean = x.mean(dim=2)  # (B, C, W)
        x_seq = x_mean.permute(0, 2, 1)  # (B, W, C)

        # Self-attention along width
        x_norm = self.norm(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)

        # Back to (B, C, W)
        attn_out = attn_out.permute(0, 2, 1)

        # Expand back to H dimension
        attn_out = attn_out.unsqueeze(2).expand(-1, -1, H, -1)

        # Residual connection
        return x + attn_out


# =================================================
# U-Net with Bottleneck Attention
# =================================================


class UNetDenoiser(nn.Module):
    """
    U-Net + Bottleneck Axial Attention for LiDAR range image denoising
    """

    def __init__(
        self,
        in_channels=5,
        num_classes=2,
        base_channels=64,
        bilinear=False,
        attn_heads=4,
    ):
        super().__init__()
        self.base_channels=base_channels

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # Bottleneck Attention
        self.bottleneck_attn = BottleneckAxialAttention(
            base_channels * 16 // factor, num_heads=attn_heads
        )

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bottleneck Attention
        x5 = self.bottleneck_attn(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def forward_with_features(self, x):
        """
        Forward pass with intermediate features for knowledge distillation
        
        Returns:
            dict: {
                'logits': final output (B, 2, H, W)
                'x2': encoder stage 1 output (B, 128, H/2, W/2)
                'x3': encoder stage 2 output (B, 256, H/4, W/4)
                'x4': encoder stage 3 output (B, 512, H/8, W/8)
                'x5': bottleneck output with attention (B, 1024, H/16, W/16)
            }
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Bottleneck Attention (IMPORTANT: Apply attention before returning features)
        x5 = self.bottleneck_attn(x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        return {
            'logits': logits,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x5': x5,  # This includes the attention-enhanced features
        }


# =================================================
# Test
# =================================================

if __name__ == "__main__":
    model = UNetDenoiser(
        in_channels=5,
        num_classes=2,
        base_channels=64,
        bilinear=False,
        attn_heads=4,
    )

    x = torch.randn(2, 5, 64, 1024)
    
    # Test standard forward
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    
    # Test forward_with_features
    features = model.forward_with_features(x)
    print("\nForward with features:")
    print(f"Logits shape: {features['logits'].shape}")
    print(f"x2 shape: {features['x2'].shape}")
    print(f"x3 shape: {features['x3'].shape}")
    print(f"x4 shape: {features['x4'].shape}")
    print(f"x5 shape: {features['x5'].shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")