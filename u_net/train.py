import argparse
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import create_dataloaders
from model_unet_trans import UNetDenoiser


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) labels
            mask: (B, H, W) valid pixel mask
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if mask is not None:
            focal_loss = focal_loss * mask
            if self.reduction == "mean":
                return focal_loss.sum() / (mask.sum() + 1e-6)
            elif self.reduction == "sum":
                return focal_loss.sum()
        else:
            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()

        return focal_loss


class MaskedCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss with mask support"""

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) labels
            mask: (B, H, W) valid pixel mask
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")

        if mask is not None:
            ce_loss = ce_loss * mask
            return ce_loss.sum() / (mask.sum() + 1e-6)
        else:
            return ce_loss.mean()


def compute_metrics(preds, labels, mask):
    """Compute evaluation metrics"""
    # Flatten and filter by mask
    mask_flat = mask.flatten().cpu().numpy() > 0
    preds_flat = preds.flatten().cpu().numpy()[mask_flat]
    labels_flat = labels.flatten().cpu().numpy()[mask_flat]

    if len(preds_flat) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou_noise": 0.0,
            "iou_clean": 0.0,
        }

    # Accuracy
    accuracy = (preds_flat == labels_flat).mean()

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, preds_flat, average="binary", zero_division=0
    )

    # IoU for each class
    cm = confusion_matrix(labels_flat, preds_flat, labels=[0, 1])

    iou_noise = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[1, 0] + 1e-6)
    iou_clean = cm[1, 1] / (cm[1, 1] + cm[1, 0] + cm[0, 1] + 1e-6)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_noise": iou_noise,
        "iou_clean": iou_clean,
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_masks = []

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch in pbar:
        inputs = batch["input"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)

        # Loss (only on valid pixels)
        loss = criterion(outputs, labels, masks)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())
        all_masks.append(masks.detach())

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    metrics = compute_metrics(all_preds, all_labels, all_masks)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics


def validate(model, dataloader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        for batch in pbar:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels, masks)

            # Metrics
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
            all_masks.append(masks)

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    metrics = compute_metrics(all_preds, all_labels, all_masks)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics


def plot_training_curves(history, output_dir):
    """Plot and save training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 2].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(epochs, history['train_precision'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(epochs, history['train_recall'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[1, 2].plot(epochs, history['val_iou_noise'], 'g-', label='IoU Noise', linewidth=2)
    axes[1, 2].plot(epochs, history['val_iou_clean'], 'orange', label='IoU Clean', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('IoU')
    axes[1, 2].set_title('Validation IoU')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    plt.close()


def save_epoch_log(epoch, train_loss, train_metrics, val_loss, val_metrics, lr, log_file):
    """Save epoch metrics to CSV file"""
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as f:
        fieldnames = [
            'epoch', 'lr',
            'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
            'train_iou_noise', 'train_iou_clean',
            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
            'val_iou_noise', 'val_iou_clean'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'epoch': epoch,
            'lr': lr,
            'train_loss': train_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'train_iou_noise': train_metrics['iou_noise'],
            'train_iou_clean': train_metrics['iou_clean'],
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_iou_noise': val_metrics['iou_noise'],
            'val_iou_clean': val_metrics['iou_clean'],
        })


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, _ = create_dataloaders(
        args.data_root,
        args.splits_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        proj_H=args.proj_h,
        proj_W=args.proj_w,
        noise_label=args.noise_label,
    )

    # Create model
    print("Creating U-Net model...")
    model = UNetDenoiser(
        in_channels=5,
        num_classes=2,
        base_channels=args.base_channels,
        bilinear=args.bilinear,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Loss and optimizer
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = MaskedCrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Initialize history for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'train_iou_noise': [], 'train_iou_clean': [],
        'val_iou_noise': [], 'val_iou_clean': []
    }

    # CSV log file
    log_file = os.path.join(args.output_dir, 'training_log.csv')

    # Training loop
    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update learning rate
        scheduler.step()

        # Save to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_iou_noise'].append(train_metrics['iou_noise'])
        history['train_iou_clean'].append(train_metrics['iou_clean'])
        history['val_iou_noise'].append(val_metrics['iou_noise'])
        history['val_iou_clean'].append(val_metrics['iou_clean'])

        # Save to CSV
        save_epoch_log(epoch, train_loss, train_metrics, val_loss, val_metrics, current_lr, log_file)

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        for key, value in train_metrics.items():
            writer.add_scalar(f"Train/{key}", value, epoch)

        for key, value in val_metrics.items():
            writer.add_scalar(f"Val/{key}", value, epoch)

        # Print metrics (コンソール出力)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )
        print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(
            f"Val IoU (noise/clean): {val_metrics['iou_noise']:.4f} / {val_metrics['iou_clean']:.4f}"
        )

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_metrics["f1"],
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print(f"Saved best model with F1: {best_val_f1:.4f}")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

    # Plot and save training curves
    plot_training_curves(history, args.output_dir)

    writer.close()
    print(f"\nTraining completed! Best Val F1: {best_val_f1:.4f}")
    print(f"Training log saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for LiDAR denoising")

    # Data
    parser.add_argument(
        "--data_root", type=str, default="./WADS/wads", help="Path to WADS dataset"
    )
    parser.add_argument(
        "--splits_json", type=str, default="./splits.json", help="Path to splits.json"
    )
    parser.add_argument(
        "--noise_label", type=int, default=110, help="Label value for noise points"
    )

    # Model (U-Net specific)
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

    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    # Loss
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use focal loss instead of CE loss",
    )
    parser.add_argument(
        "--focal_alpha", type=float, default=0.25, help="Focal loss alpha"
    )
    parser.add_argument(
        "--focal_gamma", type=float, default=2.0, help="Focal loss gamma"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save checkpoint every N epochs"
    )

    args = parser.parse_args()

    main(args)