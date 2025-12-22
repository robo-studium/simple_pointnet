import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from model_unet_trans import UNetDenoiser

if __name__ == "__main__":
    print("="*60)
    print("Student U-Net Architecture Test")
    print("="*60)

    # -------------------------------------------------
    # Create model
    # -------------------------------------------------
    model = UNetDenoiser(
        in_channels=5,
        num_classes=2,
        base_channels=64,
        bilinear=False,
        attn_heads=4,
    )
    model.eval()  # 推論モード（重要）

    # -------------------------------------------------
    # Dummy input (batch_size = 1 for FLOPs)
    # -------------------------------------------------
    x = torch.randn(1, 5, 64, 1024)

    print(f"\nInput shape: {x.shape}")

    # -------------------------------------------------
    # Forward test
    # -------------------------------------------------
    with torch.no_grad():
        logits = model(x)

    print(f"Output shape: {logits.shape}")

    # -------------------------------------------------
    # Parameter count
    # -------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # -------------------------------------------------
    # FLOPs calculation
    # -------------------------------------------------
    flops = FlopCountAnalysis(model, x)
    total_flops = flops.total()

    # -------------------------------------------------
    # Print statistics
    # -------------------------------------------------
    print("\n" + "="*60)
    print("Model Statistics")
    print("="*60)

    print(f"Total parameters      : {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Trainable parameters  : {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    print(f"Total FLOPs           : {total_flops:,} ({total_flops / 1e9:.2f} GFLOPs)")
    print(f"Base channels         : {model.base_channels}")

    # -------------------------------------------------
    # Optional: Layer-wise FLOPs (debug / analysis)
    # -------------------------------------------------
    # print(flops.by_module())

    print("\n" + "="*60)
    print("✓ FLOPs & parameter counting completed successfully!")
    print("="*60)
