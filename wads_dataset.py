import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class WADSDataset(Dataset):
    def __init__(
        self, root_dir, split="train", num_points=4096, splits_file="splits.json"
    ):
        """
        WADS Dataset Loader

        Args:
            root_dir: /path/to/WADS (シーケンスフォルダが含まれるルートディレクトリ)
            split: "train" / "val" / "test"
            num_points: サンプリングする点数 (e.g., 2048 / 4096 / 8192 / 16384 / 32768)
            splits_file: データ分割を定義するJSONファイルのパス

        特徴量: x, y, z, intensity, range の5次元

        ディレクトリ構造:
        WADS/
        ├── 11/
        │   ├── velodyne/
        │   │   ├── 000000.bin
        │   │   ├── 000001.bin
        │   │   └── ...
        │   └── labels/
        │       ├── 000000.label
        │       ├── 000001.label
        │       └── ...
        ├── 12/
        └── ...
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        # ───────────────────────────────
        # splits.jsonを読み込み
        # ───────────────────────────────
        if not os.path.exists(splits_file):
            raise FileNotFoundError(f"Splits file not found: {splits_file}")

        with open(splits_file, "r") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(
                f"Split '{split}' not found in {splits_file}. Available: {list(splits.keys())}"
            )

        sequence_ids = splits[split]
        print(f"Loading {split} split with sequences: {sequence_ids}")

        # ───────────────────────────────
        # 各シーケンスからファイルを収集
        # ───────────────────────────────
        self.velodyne_files = []
        self.label_files = []

        for seq_id in sequence_ids:
            seq_str = str(seq_id).zfill(2)  # 11 -> "11", 2桁のゼロパディング
            seq_dir = os.path.join(root_dir, seq_str)

            if not os.path.exists(seq_dir):
                print(f"Warning: Sequence directory not found: {seq_dir}")
                continue

            velodyne_dir = os.path.join(seq_dir, "velodyne")
            labels_dir = os.path.join(seq_dir, "labels")

            if not os.path.exists(velodyne_dir):
                print(f"Warning: Velodyne directory not found: {velodyne_dir}")
                continue

            if not os.path.exists(labels_dir):
                print(f"Warning: Labels directory not found: {labels_dir}")
                continue

            # 各シーケンス内のファイルを収集
            velo_files = sorted(
                [
                    os.path.join(velodyne_dir, f)
                    for f in os.listdir(velodyne_dir)
                    if f.endswith(".bin")
                ]
            )

            label_files = sorted(
                [
                    os.path.join(labels_dir, f)
                    for f in os.listdir(labels_dir)
                    if f.endswith(".label")
                ]
            )

            # ファイル数の一致を確認
            if len(velo_files) != len(label_files):
                print(
                    f"Warning: Mismatch in sequence {seq_str}: "
                    f"{len(velo_files)} velodyne files, {len(label_files)} label files"
                )
                # 少ない方に合わせる
                min_len = min(len(velo_files), len(label_files))
                velo_files = velo_files[:min_len]
                label_files = label_files[:min_len]

            self.velodyne_files.extend(velo_files)
            self.label_files.extend(label_files)

            print(f"  Sequence {seq_str}: {len(velo_files)} files")

        if len(self.velodyne_files) == 0:
            raise RuntimeError(f"No data files found for split '{split}'")

        print(f"Total {split} samples: {len(self.velodyne_files)}")

    # ───────────────────────────────
    # 1サンプル読み込み
    # ───────────────────────────────
    def __getitem__(self, idx):
        velo_path = self.velodyne_files[idx]
        label_path = self.label_files[idx]

        # 点群読み込み（.binファイル: x, y, z, intensity の4列）
        with open(velo_path, "rb") as f:
            points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)

        # intensity を 0~1 に正規化
        points[:, 3] = points[:, 3] / 255.0

        # range を計算して追加 (x, y, z, intensity, range)
        range_vals = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        range_vals = range_vals.reshape(-1, 1)  # (N, 1)
        points = np.concatenate([points, range_vals], axis=1)  # (N, 5)

        # ラベル読み込み
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

        # 二値ラベル：110 = noise（動いてる物体）
        bin_labels = (labels == 110).astype(np.int64)

        # サンプリング
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
