import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SemanticSTFDataset(Dataset):
    def __init__(self, root_dir, split="train", num_points=4096):
        """
        root_dir: /path/to/SemanticSTF
        split   : "train" / "val" / "test"
        num_points: number of points to sample (e.g., 2048 / 4096 / 8192)

        特徴量: x, y, z, intensity, range の5次元
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        # ───────────────────────────────
        # ファイルリスト作成
        # ───────────────────────────────
        velodyne_dir = os.path.join(root_dir, split, "velodyne")
        labels_dir = os.path.join(root_dir, split, "labels")

        self.velodyne_files = sorted(
            [
                os.path.join(velodyne_dir, f)
                for f in os.listdir(velodyne_dir)
                if f.endswith(".bin")
            ]
        )

        self.label_files = sorted(
            [
                os.path.join(labels_dir, f)
                for f in os.listdir(labels_dir)
                if f.endswith(".label")
            ]
        )

        assert len(self.velodyne_files) == len(
            self.label_files
        ), "点群とラベルの数が一致していません"

    # ───────────────────────────────
    # 1サンプル読み込み
    # ───────────────────────────────
    def __getitem__(self, idx):
        velo_path = self.velodyne_files[idx]
        label_path = self.label_files[idx]

        # 神読み込み（これで絶対に reshape エラー出ない！）
        with open(velo_path, "rb") as f:
            # 5列として読んで、余分は捨てる → 4列に強制整形
            points = np.fromfile(f, dtype=np.float32).reshape(-1, 5)[:, :4]

        # intensity を 0~1 に正規化
        points[:, 3] = points[:, 3] / 255.0

        # ★★★ range を計算して追加 (x, y, z, intensity, range) ★★★
        range_vals = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        range_vals = range_vals.reshape(-1, 1)  # (N, 1)
        points = np.concatenate([points, range_vals], axis=1)  # (N, 5)

        # ラベル読み込み（存在しなくても落ちない）
        if os.path.exists(label_path):
            labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            # 点数合わせ（念のため）
            if labels.shape[0] > points.shape[0]:
                labels = labels[: points.shape[0]]
            elif labels.shape[0] < points.shape[0]:
                # 稀に起こるので補完
                pad = np.zeros(points.shape[0] - labels.shape[0], dtype=np.uint32)
                labels = np.concatenate([labels, pad])
        else:
            labels = np.zeros(points.shape[0], dtype=np.uint32)

        # 二値ラベル：20 = noise（動いてる物体）
        bin_labels = (labels == 20).astype(np.int64)

        # サンプリング（従来通り）
        N = points.shape[0]
        if N == 0:
            return torch.zeros(self.num_points, 5), torch.zeros(
                self.num_points, dtype=torch.long
            )

        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            choice = np.random.choice(N, self.num_points, replace=True)

        points = points[choice]
        bin_labels = bin_labels[choice]

        return torch.from_numpy(points).float(), torch.from_numpy(bin_labels).long()

    def __len__(self):
        return len(self.velodyne_files)
