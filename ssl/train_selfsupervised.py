import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ssl.model_selfsupervised import SelfSupervisedModel
from ssl.data_loader_ssl import create_ssl_dataloaders


class MaskedL1Loss(nn.Module):
    """
    L1 Loss that only computes loss on valid and masked pixels,
    completely ignoring missing pixels.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: (B, 2, H, W) - predicted (range, intensity)
            targets: (B, 2, H, W) - ground truth (range, intensity)
            mask: (B, H, W) - pixel type mask
                  0: missing (ignore completely)
                  1: valid (compute loss)
                  2: masked (compute loss - this is what we want to reconstruct)

        Returns:
            loss: Mean L1 loss over valid and masked pixels only
        """
        # Create binary mask: 1 for valid/masked, 0 for missing
        # valid and masked pixels should have mask values >= 1
        valid_mask = (mask >= 1).float()  # (B, H, W)

        # Expand mask to match prediction channels
        valid_mask = valid_mask.unsqueeze(1)  # (B, 1, H, W)

        # Compute L1 loss
        l1_loss = torch.abs(predictions - targets)  # (B, 2, H, W)

        # Apply mask
        masked_loss = l1_loss * valid_mask

        # Compute mean over valid pixels only
        num_valid_pixels = valid_mask.sum() + 1e-6
        loss = masked_loss.sum() / num_valid_pixels

        return loss


def create_selfsupervised_batch(batch, mask_ratio=0.15, device="cpu"):
    """
    Create masked inputs for self-supervised learning from a batch.

    Args:
        batch: Original batch from dataloader
        mask_ratio: Ratio of valid pixels to mask (default: 0.15)
        device: Device to put tensors on

    Returns:
        ssl_input: (B, 4, H, W) - masked input
        target: (B, 2, H, W) - ground truth (range, intensity)
        ssl_mask: (B, H, W) - mask (0: missing, 1: valid, 2: masked)
    """
    # Original data
    inputs = batch["input"]  # (B, 5, H, W): x, y, z, intensity, range
    mask = batch["mask"]  # (B, H, W): valid pixel mask

    # Extract range and intensity
    range_img = inputs[:, 4:5, :, :].clone()  # (B, 1, H, W)
    intensity_img = inputs[:, 3:4, :, :].clone()  # (B, 1, H, W)
    xyz_mean = inputs[:, :3, :, :].mean(dim=1, keepdim=True)  # (B, 1, H, W)

    # Create mask for self-supervised learning
    # mask values: 0 = missing, 1 = valid, 2 = masked (to be reconstructed)
    ssl_mask = mask.clone()

    # For each sample in batch
    for b in range(inputs.shape[0]):
        valid_pixels = mask[b] > 0
        num_valid = valid_pixels.sum().item()

        if num_valid > 0:
            # Randomly select pixels to mask
            num_to_mask = int(num_valid * mask_ratio)

            if num_to_mask > 0:
                # Get valid pixel indices
                valid_indices = torch.nonzero(valid_pixels, as_tuple=False)

                # Randomly select indices to mask
                perm = torch.randperm(len(valid_indices))
                mask_indices = valid_indices[perm[:num_to_mask]]

                # Set selected pixels to masked (value = 2)
                ssl_mask[b, mask_indices[:, 0], mask_indices[:, 1]] = 2

                # Zero out the masked pixels in input
                range_img[b, 0, mask_indices[:, 0], mask_indices[:, 1]] = 0
                intensity_img[b, 0, mask_indices[:, 0], mask_indices[:, 1]] = 0

    # Create new input: (range, intensity, xyz_mean, ssl_mask)
    ssl_input = torch.cat(
        [range_img, intensity_img, xyz_mean, ssl_mask.unsqueeze(1).float()], dim=1
    )  # (B, 4, H, W)

    # Target is original range and intensity
    target = torch.cat(
        [
            inputs[:, 4:5, :, :],  # original range
            inputs[:, 3:4, :, :],  # original intensity
        ],
        dim=1,
    )  # (B, 2, H, W)

    return ssl_input.to(device), target.to(device), ssl_mask.to(device)


def train_epoch(
    model, dataloader, criterion, optimizer, device, epoch, mask_ratio=0.15
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")

    for batch in pbar:
        # Create self-supervised batch
        ssl_input, target, ssl_mask = create_selfsupervised_batch(
            batch, mask_ratio=mask_ratio, device=device
        )

        # Forward
        optimizer.zero_grad()
        outputs = model(ssl_input)  # (B, 2, H, W)

        # Compute loss (only on valid and masked pixels, ignore missing)
        loss = criterion(outputs, target, ssl_mask)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device, epoch, mask_ratio=0.15):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    # Metrics for masked pixels only
    total_range_mae = 0
    total_intensity_mae = 0
    num_masked_pixels = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")

        for batch in pbar:
            # Create self-supervised batch
            ssl_input, target, ssl_mask = create_selfsupervised_batch(
                batch, mask_ratio=mask_ratio, device=device
            )

            # Forward
            outputs = model(ssl_input)

            # Compute loss
            loss = criterion(outputs, target, ssl_mask)
            total_loss += loss.item()
            num_batches += 1

            # Compute MAE on masked pixels only
            masked_pixels = ssl_mask == 2  # Only the intentionally masked pixels

            if masked_pixels.sum() > 0:
                # Range MAE
                range_pred = outputs[:, 0:1, :, :]
                range_target = target[:, 0:1, :, :]
                range_mae = torch.abs(range_pred - range_target)[
                    masked_pixels.unsqueeze(1).expand_as(range_pred)
                ].mean()

                # Intensity MAE
                intensity_pred = outputs[:, 1:2, :, :]
                intensity_target = target[:, 1:2, :, :]
                intensity_mae = torch.abs(intensity_pred - intensity_target)[
                    masked_pixels.unsqueeze(1).expand_as(intensity_pred)
                ].mean()

                total_range_mae += range_mae.item() * masked_pixels.sum().item()
                total_intensity_mae += intensity_mae.item() * masked_pixels.sum().item()
                num_masked_pixels += masked_pixels.sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    avg_range_mae = total_range_mae / (num_masked_pixels + 1e-6)
    avg_intensity_mae = total_intensity_mae / (num_masked_pixels + 1e-6)

    return avg_loss, avg_range_mae, avg_intensity_mae


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    # Create dataloaders for SSL
    print("Loading SSL data...")
    train_loader, val_loader = create_ssl_dataloaders(
        args.data_root,
        args.splits_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        proj_H=args.proj_h,
        proj_W=args.proj_w,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("Creating Self-Supervised Model...")
    model = SelfSupervisedModel(
        base_channels=args.base_channels,
        bilinear=args.bilinear,
        attn_heads=args.attn_heads,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Loss and optimizer
    criterion = MaskedL1Loss()

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            mask_ratio=args.mask_ratio,
        )

        # Validate
        val_loss, val_range_mae, val_intensity_mae = validate(
            model, val_loader, criterion, device, epoch, mask_ratio=args.mask_ratio
        )

        # Update learning rate
        scheduler.step()

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Val/range_mae", val_range_mae, epoch)
        writer.add_scalar("Val/intensity_mae", val_intensity_mae, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"Val Range MAE: {val_range_mae:.4f} | Val Intensity MAE: {val_intensity_mae:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_range_mae": val_range_mae,
                    "val_intensity_mae": val_intensity_mae,
                },
                os.path.join(args.output_dir, "best_model_ssl.pth"),
            )
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(args.output_dir, f"checkpoint_ssl_epoch_{epoch}.pth"),
            )

    writer.close()
    print(f"\nTraining completed! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-Supervised Training for Range Image Inpainting"
    )

    # Data (changed for SSL dataset)
    parser.add_argument(
        "--data_root",
        type=str,
        default="./KITTI",
        help="Path to KITTI dataset directory",
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default="./splits_ssl.json",
        help="Path to splits_ssl.json",
    )

    # Model
    parser.add_argument("--proj_h", type=int, default=64, help="Range image height")
    parser.add_argument("--proj_w", type=int, default=1024, help="Range image width")
    parser.add_argument(
        "--base_channels", type=int, default=64, help="Base number of channels in U-Net"
    )
    parser.add_argument(
        "--bilinear",
        action="store_true",
        help="Use bilinear upsampling instead of transposed conv",
    )
    parser.add_argument(
        "--attn_heads",
        type=int,
        default=4,
        help="Number of attention heads in bottleneck",
    )

    # Self-supervised learning
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.15,
        help="Ratio of valid pixels to mask for reconstruction",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./outputs_ssl", help="Output directory"
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save checkpoint every N epochs"
    )

    args = parser.parse_args()

    main(args)
