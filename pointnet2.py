import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling
    xyz: (B, N, 3)
    npoint: number of samples
    return: (B, npoint) indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points, idx):
    """
    Index points according to idx
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    return: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query
    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    return: (B, S, nsample) indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3)
        points: (B, N, C) or None
        return: new_xyz, new_points
        """
        if self.group_all:
            new_xyz = xyz[:, :1, :]  # (B, 1, 3)
            grouped_xyz = xyz.unsqueeze(2)  # (B, N, 1, 3)
            if points is not None:
                grouped_points = points.unsqueeze(2)  # (B, N, 1, C)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
            new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz_norm

        # (B, npoint, nsample, C) -> (B, C, nsample, npoint)
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, 2)[0]  # (B, C, npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, C)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: (B, N, 3) - target points
        xyz2: (B, S, 3) - source points
        points1: (B, N, C1) - target features (can be None)
        points2: (B, S, C2) - source features
        return: (B, N, mlp[-1])
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # points2は(B, 1, C2)の形状なので、そのまま(B, N, C2)に拡張
            interpolated_points = points2.expand(B, N, -1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # (B, N, 3)

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            indexed = index_points(points2, idx)  # (B, N, 3, C2)
            interpolated_points = torch.sum(
                indexed * weight.view(B, N, 3, 1), dim=2
            )  # (B, N, C2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # (B, C, N)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = new_points.permute(0, 2, 1)  # (B, N, C)
        return new_points


class PointNet2(nn.Module):
    def __init__(self, in_dim=5, num_classes=2, num_points=32768):
        """
        PointNet++ for semantic segmentation (3-layer version)
        in_dim: 5 (x, y, z, intensity, range)
        num_classes: 2 (normal=0, noise=1)
        num_points: 入力点数
        """
        super(PointNet2, self).__init__()

        # 入力点数に応じてダウンサンプリング数を設定
        if num_points >= 32768:
            npoint1, npoint2 = 4096, 512
        elif num_points >= 16384:
            npoint1, npoint2 = 2048, 256
        elif num_points >= 8192:
            npoint1, npoint2 = 1024, 128
        else:
            npoint1, npoint2 = 512, 64

        # 3層のSet Abstraction
        self.sa1 = PointNetSetAbstraction(
            npoint=npoint1,
            radius=0.5,
            nsample=16,
            in_channel=in_dim + 3,
            mlp=[32, 32, 64],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=npoint2,
            radius=1.0,
            nsample=16,
            in_channel=64 + 3,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=128 + 3,
            mlp=[128, 256, 512],
            group_all=True,
        )

        # 3層のFeature Propagation
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 128, mlp=[256, 128])
        self.fp2 = PointNetFeaturePropagation(in_channel=128 + 64, mlp=[128, 64])
        self.fp1 = PointNetFeaturePropagation(in_channel=64 + in_dim, mlp=[64, 64, 64])

        # Final classifier
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

    def forward(self, x):
        """
        x: (B, N, 5) - [x, y, z, intensity, range]
        return: (B, N, num_classes)
        """
        B, N, C = x.shape

        # xyzと特徴量を分離
        xyz = x[:, :, :3]  # (B, N, 3)
        features = x  # (B, N, 5)

        # Set Abstraction (エンコーダー)
        l1_xyz, l1_points = self.sa1(xyz, features)  # (B, npoint1, 64)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, npoint2, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 1, 512)

        # Feature Propagation (デコーダー)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, npoint2, 128)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, npoint1, 64)
        l0_points = self.fp1(xyz, l1_xyz, features, l1_points)  # (B, N, 64)

        # 最終分類
        feat = l0_points.permute(0, 2, 1)  # (B, 64, N)
        feat = self.drop1(F.relu(self.bn1(self.conv1(feat))))
        out = self.conv2(feat)  # (B, num_classes, N)
        out = out.permute(0, 2, 1)  # (B, N, num_classes)

        return out
