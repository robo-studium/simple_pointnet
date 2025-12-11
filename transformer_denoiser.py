import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PointEmbedding(nn.Module):
    """
    Point-wise embedding: (N, 4) -> (N, 128)
    """
    def __init__(self, in_dim=4, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, x):
        """
        x: (B, N, 4)
        return: (B, N, 128)
        """
        B, N, C = x.shape
        x = x.reshape(B * N, C)  # (B*N, 4)
        x = self.mlp(x)  # (B*N, 128)
        x = x.reshape(B, N, -1)  # (B, N, 128)
        return x


class PatchTokenizer(nn.Module):
    """
    10000点を100グループに分割し、各グループを平均poolingでトークン化
    """
    def __init__(self, num_points=10000, num_patches=100, embed_dim=128, pool_type='mean'):
        super().__init__()
        self.num_points = num_points
        self.num_patches = num_patches
        self.points_per_patch = num_points // num_patches
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        
        assert num_points % num_patches == 0, \
            f"num_points ({num_points}) must be divisible by num_patches ({num_patches})"
    
    def forward(self, x):
        """
        x: (B, N=10000, 128)
        return: (B, num_patches=100, 128)
        """
        B, N, D = x.shape
        
        # (B, N, D) -> (B, num_patches, points_per_patch, D)
        x = x.reshape(B, self.num_patches, self.points_per_patch, D)
        
        # Pooling: (B, num_patches, points_per_patch, D) -> (B, num_patches, D)
        if self.pool_type == 'mean':
            tokens = x.mean(dim=2)
        elif self.pool_type == 'max':
            tokens = x.max(dim=2)[0]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        return tokens


class PositionalEncoding(nn.Module):
    """
    各パッチの中心座標(mean xyz)から128次元のpositional encodingを生成
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    
    def forward(self, points):
        """
        points: (B, N=10000, 4) - original point cloud with xyz
        return: (B, num_patches=100, embed_dim)
        """
        B, N, _ = points.shape
        num_patches = 100
        points_per_patch = N // num_patches
        
        # xyz座標のみ取得
        xyz = points[:, :, :3]  # (B, N, 3)
        
        # パッチごとに分割して中心座標を計算
        xyz = xyz.reshape(B, num_patches, points_per_patch, 3)
        patch_centers = xyz.mean(dim=2)  # (B, num_patches, 3)
        
        # MLPで embed_dim 次元に投影
        B, P, _ = patch_centers.shape
        patch_centers = patch_centers.reshape(B * P, 3)
        pos_enc = self.mlp(patch_centers)  # (B*P, embed_dim)
        pos_enc = pos_enc.reshape(B, P, -1)  # (B, num_patches, embed_dim)
        
        return pos_enc


class TransformerEncoder(nn.Module):
    """
    Standard Transformer Encoder
    """
    def __init__(self, embed_dim=128, num_heads=8, num_layers=6, ffn_dim=512, dropout=0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, x):
        """
        x: (B, num_patches, embed_dim)
        return: (B, num_patches, embed_dim)
        """
        return self.transformer(x)


class PatchExpander(nn.Module):
    """
    パッチトークン(100 tokens)を元の10000点に展開
    各パッチの出力を対応する100点に複製
    """
    def __init__(self, num_points=10000, num_patches=100):
        super().__init__()
        self.num_points = num_points
        self.num_patches = num_patches
        self.points_per_patch = num_points // num_patches
    
    def forward(self, patch_features):
        """
        patch_features: (B, num_patches=100, embed_dim)
        return: (B, N=10000, embed_dim)
        """
        B, P, D = patch_features.shape
        
        # 各パッチの特徴をpoints_per_patch回複製
        # (B, num_patches, embed_dim) -> (B, num_patches, points_per_patch, embed_dim)
        expanded = patch_features.unsqueeze(2).expand(B, P, self.points_per_patch, D)
        
        # (B, num_patches, points_per_patch, embed_dim) -> (B, N, embed_dim)
        expanded = expanded.reshape(B, self.num_points, D)
        
        return expanded


class ClassificationHead(nn.Module):
    """
    最終分類層: 128 -> 64 -> 2
    """
    def __init__(self, embed_dim=128, num_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        x: (B, N, embed_dim)
        return: (B, N, num_classes)
        """
        B, N, D = x.shape
        x = x.reshape(B * N, D)  # (B*N, embed_dim)
        x = self.mlp(x)  # (B*N, num_classes)
        x = x.reshape(B, N, -1)  # (B, N, num_classes)
        return x


class TransformerPointCloudDenoiser(nn.Module):
    """
    Transformer-based Point Cloud Denoising Model
    
    Architecture:
        1. Point Embedding: (N, 4) -> (N, 128)
        2. Patch Tokenization: 10000 points -> 100 patches
        3. Positional Encoding: patch center coords -> 128 dim
        4. Transformer Encoder: 100 tokens -> 100 tokens
        5. Patch Expansion: 100 tokens -> 10000 points
        6. Classification Head: (N, 128) -> (N, 2)
    
    Args:
        num_points: Number of input points (default: 10000)
        num_patches: Number of patches (default: 100)
        in_dim: Input feature dimension (default: 4 for x,y,z,intensity)
        embed_dim: Embedding dimension (default: 128)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 6)
        ffn_dim: Feedforward network hidden dimension (default: 512)
        num_classes: Number of output classes (default: 2 for binary classification)
        pool_type: Pooling type for patch tokenization ('mean' or 'max')
    """
    def __init__(
        self,
        num_points=10000,
        num_patches=100,
        in_dim=4,
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        ffn_dim=512,
        num_classes=2,
        pool_type='mean',
        dropout=0.1
    ):
        super().__init__()
        
        self.num_points = num_points
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # 1. Point-wise embedding
        self.point_embedding = PointEmbedding(in_dim=in_dim, out_dim=embed_dim)
        
        # 2. Patch tokenization
        self.patch_tokenizer = PatchTokenizer(
            num_points=num_points,
            num_patches=num_patches,
            embed_dim=embed_dim,
            pool_type=pool_type
        )
        
        # 3. Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim)
        
        # 4. Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout
        )
        
        # 5. Patch expander
        self.patch_expander = PatchExpander(
            num_points=num_points,
            num_patches=num_patches
        )
        
        # 6. Classification head
        self.classifier = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes
        )
    
    def forward(self, x):
        """
        x: (B, N=10000, 4) - Point cloud with x,y,z,intensity
        return: (B, N=10000, 2) - Binary classification logits
        """
        # 1. Point embedding: (B, N, 4) -> (B, N, 128)
        point_features = self.point_embedding(x)
        
        # 2. Patch tokenization: (B, N, 128) -> (B, 100, 128)
        patch_tokens = self.patch_tokenizer(point_features)
        
        # 3. Positional encoding
        pos_enc = self.positional_encoding(x)
        
        # Add positional encoding to tokens
        patch_tokens = patch_tokens + pos_enc
        
        # 4. Transformer encoding: (B, 100, 128) -> (B, 100, 128)
        encoded_tokens = self.transformer(patch_tokens)
        
        # 5. Expand patches back to points: (B, 100, 128) -> (B, N, 128)
        expanded_features = self.patch_expander(encoded_tokens)
        
        # 6. Classification: (B, N, 128) -> (B, N, 2)
        logits = self.classifier(expanded_features)
        
        return logits


# ============================
#   モデル情報表示関数
# ============================
def print_model_info(model):
    """
    モデルのパラメータ数と構造を表示
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # テスト実行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # モデル作成
    model = TransformerPointCloudDenoiser(
        num_points=10000,
        num_patches=100,
        in_dim=4,
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        ffn_dim=512,
        num_classes=2,
        pool_type='mean'
    ).to(device)
    
    # モデル情報表示
    print_model_info(model)
    
    # テストデータ
    batch_size = 2
    test_input = torch.randn(batch_size, 10000, 4).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 10000, 2)")
    
    assert output.shape == (batch_size, 10000, 2), "Output shape mismatch!"
    print("\n✓ Model test passed!")