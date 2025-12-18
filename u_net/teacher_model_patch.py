"""
Patch for Teacher Model (model_unet_trans.py)
Add this method to the UNetDenoiser class to support knowledge distillation
"""

def forward_with_features(self, x):
    """
    Forward pass with intermediate features for knowledge distillation
    
    Add this method to your UNetDenoiser class in model_unet_trans.py
    
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
    
    # Bottleneck Attention
    x5 = self.bottleneck_attn(x5)
    
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


# ==============================================================================
# FULL MODIFIED model_unet_trans.py (FOR REFERENCE)
# ==============================================================================
"""
Simply add the forward_with_features method to your existing UNetDenoiser class.
Here's how the modified class should look:

class UNetDenoiser(nn.Module):
    def __init__(self, in_channels=5, num_classes=2, base_channels=64, bilinear=False, attn_heads=4):
        super().__init__()
        # ... existing __init__ code ...
    
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
"""

print("""
=================================================================
Teacher Model Patch Instructions (for model_unet_trans.py)
=================================================================

To enable knowledge distillation with Axial Attention Teacher,
add the following method to your UNetDenoiser class:

def forward_with_features(self, x):
    '''Forward pass with intermediate features'''
    # Encoder
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    
    # Bottleneck Attention (CRITICAL: Apply before returning)
    x5 = self.bottleneck_attn(x5)
    
    # Decoder
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    
    logits = self.outc(x)
    
    return {
        'logits': logits,
        'x2': x2,  # 128 channels
        'x3': x3,  # 256 channels
        'x4': x4,  # 512 channels
        'x5': x5,  # 1024 channels (with attention)
    }

IMPORTANT NOTES:
1. The x5 feature returned includes the axial attention enhancement
2. This allows the student to learn from attention-enhanced features
3. Student will learn to mimic attention effects through feature distillation
4. No need to modify the existing forward() method

Feature Shapes (base_channels=64):
- x2: (B, 128, 32, 512)
- x3: (B, 256, 16, 256)
- x4: (B, 512, 8, 128)
- x5: (B, 1024, 4, 64) with Axial Attention applied

=================================================================
""")