import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VelodyneRangeProjection:
    """Range image projection for Velodyne HDL-64E"""

    def __init__(self, proj_H=64, proj_W=1024, fov_up=2.0, fov_down=-24.9):
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

    def project(self, points, remissions):
        """
        Project 3D points to range image

        Args:
            points: (N, 3) array of x, y, z coordinates
            remissions: (N,) array of intensity values

        Returns:
            proj_range: (H, W) range image
            proj_xyz: (H, W, 3) xyz coordinate image
            proj_remission: (H, W) intensity image
            proj_mask: (H, W) valid pixel mask
            proj_idx: (H, W) original point index mapping
        """
        # Initialize projection images
        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_xyz = np.zeros((self.proj_H, self.proj_W, 3), dtype=np.float32)
        proj_remission = np.zeros((self.proj_H, self.proj_W), dtype=np.float32)
        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        # FOV in radians
        fov_up = self.proj_fov_up / 180.0 * np.pi
        fov_down = self.proj_fov_down / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)

        # Calculate depth
        depth = np.linalg.norm(points, axis=1)

        # Remove invalid points
        valid = depth > 0
        depth = depth[valid]
        points = points[valid]
        remission = remissions[valid]

        if len(depth) == 0:
            proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)
            return proj_range, proj_xyz, proj_remission, proj_mask, proj_idx

        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # Calculate angles
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(np.clip(scan_z / depth, -1.0, 1.0))

        # Project to image coordinates [0, 1]
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov

        # Scale to image size
        proj_x *= self.proj_W
        proj_y *= self.proj_H

        # Discretize
        proj_x = np.floor(proj_x).astype(np.int32)
        proj_y = np.floor(proj_y).astype(np.int32)

        # Clip to valid range
        proj_x = np.clip(proj_x, 0, self.proj_W - 1)
        proj_y = np.clip(proj_y, 0, self.proj_H - 1)

        # Sort by depth (far to near) for occlusion handling
        indices = np.arange(len(depth))
        order = np.argsort(depth)[::-1]

        depth = depth[order]
        points = points[order]
        remission = remission[order]
        proj_x = proj_x[order]
        proj_y = proj_y[order]
        indices = indices[order]

        # Assign to range image
        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points
        proj_remission[proj_y, proj_x] = remission
        proj_idx[proj_y, proj_x] = indices

        # Create valid mask
        proj_mask = (proj_idx >= 0).astype(np.int32)

        return proj_range, proj_xyz, proj_remission, proj_mask, proj_idx


class KITTISSLDataset(Dataset):
    """
    KITTI Dataset for Self-Supervised Learning (SSL)

    This dataset loads clean weather data from KITTI directory structure.
    No labels are required for SSL training.
    """

    def __init__(
        self,
        root_dir,
        sequences,
        proj_H=64,
        proj_W=1024,
        fov_up=2.0,
        fov_down=-24.9,
    ):
        """
        Args:
            root_dir: Path to KITTI directory (e.g., './KITTI')
            sequences: List of sequence numbers to include (e.g., [1, 2, 5, 9])
            proj_H: Range image height
            proj_W: Range image width
            fov_up: Vertical FOV upper bound (degrees)
            fov_down: Vertical FOV lower bound (degrees)
        """
        self.root_dir = root_dir
        self.sequences = sequences

        # Range projection
        self.projector = VelodyneRangeProjection(proj_H, proj_W, fov_up, fov_down)

        # Collect all data files
        self.data_list = []
        for seq in sequences:
            # KITTI directory structure: KITTI/{sequence}/velodyne/*.bin
            seq_dir = os.path.join(root_dir, str(seq))
            velodyne_dir = os.path.join(seq_dir, "")

            if not os.path.exists(velodyne_dir):
                print(f"Warning: {velodyne_dir} does not exist")
                continue

            bin_files = sorted(
                [f for f in os.listdir(velodyne_dir) if f.endswith(".bin")]
            )

            for bin_file in bin_files:
                bin_path = os.path.join(velodyne_dir, bin_file)
                self.data_list.append(
                    {
                        "bin_path": bin_path,
                        "sequence": seq,
                        "filename": bin_file,
                    }
                )

        print(f"Loaded {len(self.data_list)} samples from sequences {sequences}")

    def __len__(self):
        return len(self.data_list)

    def load_bin(self, bin_path):
        """Load Velodyne .bin file"""
        points = np.fromfile(bin_path, dtype=np.float32)
        points = points.reshape((-1, 4))
        xyz = points[:, :3]
        intensity = points[:, 3]
        return xyz, intensity

    def __getitem__(self, idx):
        data_info = self.data_list[idx]

        # Load point cloud
        xyz, intensity = self.load_bin(data_info["bin_path"])

        # Project to range image
        proj_range, proj_xyz, proj_intensity, proj_mask, proj_idx = (
            self.projector.project(xyz, intensity)
        )

        # Stack input channels: x, y, z, intensity, range
        input_img = np.stack(
            [
                proj_xyz[:, :, 0],  # x
                proj_xyz[:, :, 1],  # y
                proj_xyz[:, :, 2],  # z
                proj_intensity,  # intensity
                proj_range,  # range
            ],
            axis=0,
        ).astype(np.float32)

        # Normalize inputs
        # Handle invalid pixels (where mask is 0)
        for c in range(5):
            channel = input_img[c]
            valid_pixels = proj_mask > 0
            if valid_pixels.sum() > 0:
                valid_values = channel[valid_pixels]
                mean = valid_values.mean()
                std = valid_values.std() + 1e-6
                channel[valid_pixels] = (valid_values - mean) / std
                channel[~valid_pixels] = 0  # Set invalid pixels to 0

        # Convert to tensors
        input_tensor = torch.from_numpy(input_img)
        mask_tensor = torch.from_numpy(proj_mask).float()

        return {
            "input": input_tensor,  # (5, H, W): x, y, z, intensity, range
            "mask": mask_tensor,  # (H, W): valid pixel mask
            "sequence": data_info["sequence"],
            "path": data_info["bin_path"],
            "filename": data_info["filename"],
        }


def create_ssl_dataloaders(
    root_dir,
    splits_json,
    batch_size=4,
    num_workers=4,
    proj_H=64,
    proj_W=1024,
):
    """
    Create train and val dataloaders for SSL

    Args:
        root_dir: Path to KITTI directory
        splits_json: Path to splits_ssl.json
        batch_size: Batch size
        num_workers: Number of data loading workers
        proj_H: Range image height
        proj_W: Range image width

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """

    # Load splits
    with open(splits_json, "r") as f:
        splits = json.load(f)

    # Create datasets
    train_dataset = KITTISSLDataset(root_dir, splits["train"], proj_H, proj_W)

    val_dataset = KITTISSLDataset(root_dir, splits["val"], proj_H, proj_W)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    root_dir = "./KITTI"
    splits_json = "./splits_ssl.json"

    train_loader, val_loader = create_ssl_dataloaders(
        root_dir, splits_json, batch_size=2, num_workers=0, proj_H=64, proj_W=1024
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    for batch in train_loader:
        print(f"Input shape: {batch['input'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"Sequences: {batch['sequence']}")
        break
