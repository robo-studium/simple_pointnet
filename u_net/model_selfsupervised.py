import torch
import torch.nn as nn
from model_unet_trans import UNetDenoiser


class SelfSupervisedModel(nn.Module):
    """
    Self-supervised learning model for range image inpainting.

    Uses the U-Net architecture from model_unet_trans.py but changes
    the output layer to reconstruct (range, intensity) instead of
    classification (noise/clean).

    Input: 4 channels (range, intensity, xyz, mask)
    Output: 2 channels (range_recon, intensity_recon)
    """

    def __init__(
        self,
        base_channels=64,
        bilinear=False,
        attn_heads=4,
    ):
        """
        Args:
            base_channels: Base number of channels in U-Net
            bilinear: Use bilinear upsampling instead of transposed conv
            attn_heads: Number of attention heads in bottleneck
        """
        super().__init__()

        # Use the existing U-Net architecture with 4 input channels
        # We'll modify the output layer after initialization
        self.backbone = UNetDenoiser(
            in_channels=4,  # range, intensity, xyz (collapsed to 1), mask
            num_classes=2,  # Placeholder, will be replaced
            base_channels=base_channels,
            bilinear=bilinear,
            attn_heads=attn_heads,
        )

        # Replace the output layer to produce 2 channels instead of num_classes
        # The original outc expects: OutConv(base_channels, num_classes)
        # We replace it with: OutConv(base_channels, 2)
        self.backbone.outc = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, 4, H, W)
               - Channel 0: range
               - Channel 1: intensity
               - Channel 2: xyz (mean of x, y, z)
               - Channel 3: mask

        Returns:
            output: Reconstructed tensor of shape (B, 2, H, W)
                    - Channel 0: reconstructed range
                    - Channel 1: reconstructed intensity
        """
        return self.backbone(x)


if __name__ == "__main__":
    # Test the model
    model = SelfSupervisedModel(
        base_channels=64,
        bilinear=False,
        attn_heads=4,
    )

    # Create dummy input: (batch, channels, height, width)
    # 4 channels: range, intensity, xyz, mask
    x = torch.randn(2, 4, 64, 1024)

    # Forward pass
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Expected output: (batch_size, 2, height, width)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
