import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# =================================================
# Utility: Window Partition / Reverse
# =================================================
def window_partition(x, window_size):
    # x: (B, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    # windows: (num_windows*B, window_size^2, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# =================================================
# Early Convolutional Stem (CvT-style)
# =================================================
class ConvStem(nn.Module):
    def __init__(self, in_channels=5, hidden_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


# =================================================
# Patch Embedding
# =================================================
class PatchEmbedding(nn.Module):
    def __init__(self, patch_h, patch_w, in_channels, embed_dim):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


# =================================================
# Window Attention
# =================================================
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


# =================================================
# Transformer Block (Window-based)
# =================================================
class WindowTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, H, W)

        x = x.view(B, N, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# =================================================
# Vision Transformer Denoiser (Final)
# =================================================
class VisionTransformerDenoiser(nn.Module):
    def __init__(
        self,
        img_h=64,
        img_w=1024,
        patch_h=2,
        patch_w=32,
        embed_dim=384,
        depth=6,
        num_heads=6,
        window_size=4,
        num_classes=2,
    ):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.Hp = img_h // patch_h
        self.Wp = img_w // patch_w

        # Early Conv
        self.conv_stem = ConvStem(5, 64)

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_h, patch_w, in_channels=64, embed_dim=embed_dim
        )

        # Angle-based PE
        self.angle_mlp = nn.Sequential(
            nn.Linear(4, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                WindowTransformerBlock(embed_dim, num_heads, window_size=window_size)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.upsample = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=self.Hp, w=self.Wp),
            nn.ConvTranspose2d(
                num_classes,
                num_classes,
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w),
            ),
        )

    def _angle_pe(self, B, device):
        yaw = torch.linspace(-torch.pi, torch.pi, self.img_w, device=device)
        pitch = torch.linspace(
            2.0 / 180 * torch.pi, -24.9 / 180 * torch.pi, self.img_h, device=device
        )
        yaw_map = yaw.unsqueeze(0).repeat(self.img_h, 1)
        pitch_map = pitch.unsqueeze(1).repeat(1, self.img_w)

        pe = (
            torch.stack(
                [
                    torch.sin(yaw_map),
                    torch.cos(yaw_map),
                    torch.sin(pitch_map),
                    torch.cos(pitch_map),
                ],
                dim=0,
            )
            .unsqueeze(0)
            .repeat(B, 1, 1, 1)
        )

        pe = F.avg_pool2d(
            pe,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        pe = rearrange(pe, "b c h w -> b (h w) c")
        return self.angle_mlp(pe)

    def forward(self, x):
        B = x.size(0)
        device = x.device

        x = self.conv_stem(x)
        x = self.patch_embed(x)
        x = x + self._angle_pe(B, device)

        for blk in self.blocks:
            x = blk(x, self.Hp, self.Wp)

        x = self.norm(x)
        x = self.head(x)
        x = self.upsample(x)
        return x


# =================================================
# Test
# =================================================
if __name__ == "__main__":
    model = VisionTransformerDenoiser(
        img_h=64,
        img_w=1024,
        patch_h=2,
        patch_w=32,
        embed_dim=384,
        depth=6,
        num_heads=6,
        window_size=4,
        num_classes=2,
    )

    x = torch.randn(2, 5, 64, 1024)
    y = model(x)
    print("Output shape:", y.shape)
