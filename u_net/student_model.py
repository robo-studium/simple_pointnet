import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (Depthwise + Pointwise)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channels, 
            bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightDoubleConv(nn.Module):
    """Lightweight (DepthwiseSeparableConv -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class LightDown(nn.Module):
    """Lightweight downscaling: MaxPool + LightDoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            LightDoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class LightUp(nn.Module):
    """Lightweight upscaling: Bilinear upsample + single DepthwiseSeparableConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Reduce channels after concatenation
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution for classification"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class StudentUNet(nn.Module):
    """
    Lightweight Student U-Net for Knowledge Distillation
    
    Key features:
    - Depthwise Separable Convolutions throughout
    - No Attention modules
    - Efficient decoder with bilinear upsampling
    - Feature extraction support for distillation
    
    Args:
        in_channels: Number of input channels (5: x, y, z, intensity, range)
        num_classes: Number of output classes (2: noise, clean)
        base_channels: Base number of channels (24 or 32 recommended)
    """
    
    def __init__(self, in_channels=5, num_classes=2, base_channels=24):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        # Encoder
        self.inc = LightDoubleConv(in_channels, base_channels)
        self.down1 = LightDown(base_channels, base_channels * 2)      # 24 -> 48
        self.down2 = LightDown(base_channels * 2, base_channels * 4)  # 48 -> 96
        self.down3 = LightDown(base_channels * 4, base_channels * 8)  # 96 -> 192
        
        # Bottleneck (no attention, just conv)
        self.down4 = LightDown(base_channels * 8, base_channels * 10) # 192 -> 240
        
        # Decoder (lightweight with bilinear upsample)
        self.up1 = LightUp(base_channels * 10 + base_channels * 8, base_channels * 8)  # 240+192 -> 192
        self.up2 = LightUp(base_channels * 8 + base_channels * 4, base_channels * 4)   # 192+96 -> 96
        self.up3 = LightUp(base_channels * 4 + base_channels * 2, base_channels * 2)   # 96+48 -> 48
        self.up4 = LightUp(base_channels * 2 + base_channels, base_channels)           # 48+24 -> 24
        
        # Output
        self.outc = OutConv(base_channels, num_classes)
    
    def forward(self, x):
        """Standard forward pass"""
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output logits
        logits = self.outc(x)
        return logits
    
    def forward_with_features(self, x):
        """
        Forward pass with intermediate features for knowledge distillation
        
        Returns:
            dict: {
                'logits': final output (B, 2, H, W)
                'x2': encoder stage 1 output (B, 48, H/2, W/2)
                'x3': encoder stage 2 output (B, 96, H/4, W/4)
                'x4': encoder stage 3 output (B, 192, H/8, W/8)
                'x5': bottleneck output (B, 240, H/16, W/16)
            }
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output logits
        logits = self.outc(x)
        
        return {
            'logits': logits,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x5': x5,
        }


# =================================================
# Test
# =================================================
if __name__ == "__main__":
    print("="*60)
    print("Student U-Net Architecture Test")
    print("="*60)
    
    # Create model
    model = StudentUNet(in_channels=5, num_classes=2, base_channels=24)
    
    # Test input (batch_size=2, channels=5, H=64, W=1024)
    x = torch.randn(2, 5, 64, 1024)
    
    print(f"\nInput shape: {x.shape}")
    
    # Test standard forward
    print("\n--- Standard Forward ---")
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # Test forward with features
    print("\n--- Forward with Features (for Distillation) ---")
    features = model.forward_with_features(x)
    print(f"Logits shape: {features['logits'].shape}")
    print(f"x2 shape: {features['x2'].shape}")
    print(f"x3 shape: {features['x3'].shape}")
    print(f"x4 shape: {features['x4'].shape}")
    print(f"x5 shape: {features['x5'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("Model Statistics")
    print("="*60)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Base channels: {model.base_channels}")
    
    # Compare with different base_channels
    print("\n" + "="*60)
    print("Parameter Comparison")
    print("="*60)
    for bc in [24, 32, 48, 64]:
        m = StudentUNet(in_channels=5, num_classes=2, base_channels=bc)
        params = sum(p.numel() for p in m.parameters())
        print(f"base_channels={bc:2d}: {params:,} params ({params/1e6:.2f}M)")
    
    print("\n" + "="*60)
    print("✓ Student model is ready for knowledge distillation!")
    print("="*60)