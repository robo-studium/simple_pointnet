import argparse
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import create_dataloaders
from student_model import StudentUNet


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    1. Task loss (cross-entropy with ground truth)
    2. Distillation loss (KL divergence with teacher logits)
    3. Feature matching loss (MSE between intermediate features)
    
    The student learns to mimic attention effects through feature distillation
    without explicitly using attention modules.
    """
    
    def __init__(self, alpha=0.5, temperature=4.0, feature_weight=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for distillation loss
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
        # Feature projection layers (learnable, will be created on-the-fly)
        self.feature_projectors = nn.ModuleDict()
    
    def get_or_create_projector(self, student_channels, teacher_channels, key, device):
        """Create channel alignment projector if needed"""
        projector_key = f"{key}_{student_channels}_to_{teacher_channels}"
        
        if projector_key not in self.feature_projectors:
            projector = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False).to(device)
            self.feature_projectors[projector_key] = projector
        
        return self.feature_projectors[projector_key]
    
    def forward(self, student_outputs, teacher_outputs, targets, mask=None):
        """
        Args:
            student_outputs: dict with 'logits', 'x2', 'x3', 'x4', 'x5'
            teacher_outputs: dict with 'logits', 'x2', 'x3', 'x4', 'x5' (with attention)
            targets: ground truth labels (B, H, W)
            mask: valid pixel mask (B, H, W)
        """
        student_logits = student_outputs['logits']
        teacher_logits = teacher_outputs['logits']
        
        # 1. Task loss (student vs ground truth)
        task_loss = self.ce_loss(student_logits, targets)
        if mask is not None:
            task_loss = (task_loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            task_loss = task_loss.mean()
        
        # 2. Distillation loss (student vs teacher logits)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 3. Feature matching loss (learn attention effects implicitly)
        feature_loss = 0.0
        feature_keys = ['x2', 'x3', 'x4', 'x5']
        
        for key in feature_keys:
            if key in student_outputs and key in teacher_outputs:
                s_feat = student_outputs[key]
                t_feat = teacher_outputs[key]  # Teacher features include attention effects
                
                # Align channels if different
                if s_feat.shape[1] != t_feat.shape[1]:
                    projector = self.get_or_create_projector(
                        s_feat.shape[1], t_feat.shape[1], key, s_feat.device
                    )
                    s_feat = projector(s_feat)
                
                # Normalize features before matching
                s_feat_norm = F.normalize(s_feat, p=2, dim=1)
                t_feat_norm = F.normalize(t_feat, p=2, dim=1)
                
                # MSE loss to match attention-enhanced features
                feature_loss += self.mse_loss(s_feat_norm, t_feat_norm.detach())
        
        feature_loss = feature_loss / len(feature_keys)
        
        # Combined loss
        total_loss = (
            (1 - self.alpha) * task_loss + 
            self.alpha * distill_loss + 
            self.feature_weight * feature_loss
        )
        
        return {
            'total': total_loss,
            'task': task_loss,
            'distill': distill_loss,
            'feature': feature_loss
        }


def compute_metrics(preds, labels, mask):
    """Compute evaluation metrics"""
    mask_flat = mask.flatten().cpu().numpy() > 0
    preds_flat = preds.flatten().cpu().numpy()[mask_flat]
    labels_flat = labels.flatten().cpu().numpy()[mask_flat]
    
    if len(preds_flat) == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "iou_noise": 0.0, "iou_clean": 0.0,
        }
    
    accuracy = (preds_flat == labels_flat).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, preds_flat, average="binary", zero_division=0
    )
    
    cm = confusion_matrix(labels_flat, preds_flat, labels=[0, 1])
    iou_noise = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[1, 0] + 1e-6)
    iou_clean = cm[1, 1] / (cm[1, 1] + cm[1, 0] + cm[0, 1] + 1e-6)
    
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "iou_noise": iou_noise, "iou_clean": iou_clean,
    }


def train_epoch(student, teacher, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with distillation"""
    student.train()
    teacher.eval()  # Teacher always in eval mode
    
    total_losses = {'total': 0, 'task': 0, 'distill': 0, 'feature': 0}
    all_preds, all_labels, all_masks = [], [], []
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch in pbar:
        inputs = batch["input"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        student_outputs = student.forward_with_features(inputs)
        
        with torch.no_grad():
            teacher_outputs = teacher.forward_with_features(inputs)
        
        # Compute distillation loss
        losses = criterion(student_outputs, teacher_outputs, labels, masks)
        
        # Backward
        losses['total'].backward()
        optimizer.step()
        
        # Metrics
        preds = torch.argmax(student_outputs['logits'], dim=1)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())
        all_masks.append(masks.detach())
        
        for k in total_losses.keys():
            total_losses[k] += losses[k].item()
        
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'task': losses['task'].item(),
            'dist': losses['distill'].item()
        })
    
    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    metrics = compute_metrics(all_preds, all_labels, all_masks)
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    return avg_losses, metrics


def validate(student, teacher, dataloader, criterion, device, epoch):
    """Validate model with distillation loss"""
    student.eval()
    teacher.eval()
    
    total_losses = {'total': 0, 'task': 0, 'distill': 0, 'feature': 0}
    all_preds, all_labels, all_masks = [], [], []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        for batch in pbar:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            
            # Forward pass
            student_outputs = student.forward_with_features(inputs)
            teacher_outputs = teacher.forward_with_features(inputs)
            
            # Compute loss
            losses = criterion(student_outputs, teacher_outputs, labels, masks)
            
            # Metrics
            preds = torch.argmax(student_outputs['logits'], dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
            all_masks.append(masks)
            
            for k in total_losses.keys():
                total_losses[k] += losses[k].item()
            
            pbar.set_postfix({'loss': losses['total'].item()})
    
    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    metrics = compute_metrics(all_preds, all_labels, all_masks)
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    return avg_losses, metrics


def plot_training_curves(history, output_dir):
    """Plot and save training curves"""
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Student Training with Distillation (from Axial Attention Teacher)', 
                 fontsize=16, fontweight='bold')
    
    # Total Loss
    axes[0, 0].plot(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_total_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Task Loss
    axes[0, 1].plot(epochs, history['train_task_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_task_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Task Loss')
    axes[0, 1].set_title('Task Loss (CE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distillation Loss
    axes[0, 2].plot(epochs, history['train_distill_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val_distill_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Distillation Loss')
    axes[0, 2].set_title('Distillation Loss (KL)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, history['train_accuracy'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_accuracy'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 1].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score')
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
    save_path = os.path.join(output_dir, 'distillation_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    plt.close()


def save_epoch_log(epoch, train_losses, train_metrics, val_losses, val_metrics, lr, log_file):
    """Save epoch metrics to CSV file"""
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as f:
        fieldnames = [
            'epoch', 'lr',
            'train_total_loss', 'train_task_loss', 'train_distill_loss', 'train_feature_loss',
            'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
            'train_iou_noise', 'train_iou_clean',
            'val_total_loss', 'val_task_loss', 'val_distill_loss', 'val_feature_loss',
            'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
            'val_iou_noise', 'val_iou_clean'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'epoch': epoch,
            'lr': lr,
            'train_total_loss': train_losses['total'],
            'train_task_loss': train_losses['task'],
            'train_distill_loss': train_losses['distill'],
            'train_feature_loss': train_losses['feature'],
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'train_iou_noise': train_metrics['iou_noise'],
            'train_iou_clean': train_metrics['iou_clean'],
            'val_total_loss': val_losses['total'],
            'val_task_loss': val_losses['task'],
            'val_distill_loss': val_losses['distill'],
            'val_feature_loss': val_losses['feature'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_iou_noise': val_metrics['iou_noise'],
            'val_iou_clean': val_metrics['iou_clean'],
        })


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, _ = create_dataloaders(
        args.data_root, args.splits_json,
        batch_size=args.batch_size, num_workers=args.num_workers,
        proj_H=args.proj_h, proj_W=args.proj_w, noise_label=args.noise_label,
    )
    
    # Create Student model
    print("Creating Student model...")
    student = StudentUNet(
        in_channels=5, num_classes=2, base_channels=args.base_channels
    ).to(device)
    
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {student_params:,} ({student_params/1e6:.2f}M)")
    
    # Load Teacher model (with Axial Attention)
    print(f"Loading Teacher model (with Axial Attention) from: {args.teacher_checkpoint}")
    
    from model_unet_trans import UNetDenoiser as TeacherModel
    
    teacher = TeacherModel(
        in_channels=5, 
        num_classes=2, 
        base_channels=args.teacher_base_channels, 
        bilinear=args.teacher_bilinear,
        attn_heads=args.teacher_attn_heads
    ).to(device)
    
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher parameters: {teacher_params:,} ({teacher_params/1e6:.2f}M)")
    print(f"Compression ratio: {teacher_params/student_params:.2f}x")
    print(f"Teacher has Axial Attention at bottleneck - Student learns to mimic via distillation")
    
    # Loss and optimizer
    criterion = DistillationLoss(
        alpha=args.alpha,
        temperature=args.temperature,
        feature_weight=args.feature_weight
    )
    
    optimizer = optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Initialize history
    history = {
        'train_total_loss': [], 'val_total_loss': [],
        'train_task_loss': [], 'val_task_loss': [],
        'train_distill_loss': [], 'val_distill_loss': [],
        'train_feature_loss': [], 'val_feature_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'train_iou_noise': [], 'train_iou_clean': [],
        'val_iou_noise': [], 'val_iou_clean': []
    }
    
    log_file = os.path.join(args.output_dir, 'distillation_log.csv')
    
    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    early_stop = (args.early_stop_patience > 0)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_losses, train_metrics = train_epoch(
            student, teacher, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_losses, val_metrics = validate(
            student, teacher, val_loader, criterion, device, epoch
        )
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        # Update history
        for key in ['total', 'task', 'distill', 'feature']:
            history[f'train_{key}_loss'].append(train_losses[key])
            history[f'val_{key}_loss'].append(val_losses[key])
        
        for key in ['accuracy', 'precision', 'recall', 'f1', 'iou_noise', 'iou_clean']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        # Save to CSV
        save_epoch_log(epoch, train_losses, train_metrics, val_losses, val_metrics, current_lr, log_file)
        
        # Log to tensorboard
        writer.add_scalar("Loss/train_total", train_losses['total'], epoch)
        writer.add_scalar("Loss/val_total", val_losses['total'], epoch)
        writer.add_scalar("Loss/train_distill", train_losses['distill'], epoch)
        writer.add_scalar("Loss/val_distill", val_losses['distill'], epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        for key, value in train_metrics.items():
            writer.add_scalar(f"Train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"Val/{key}", value, epoch)
        
        # Print metrics
        print(f"Train Loss: {train_losses['total']:.4f} (Task: {train_losses['task']:.4f}, "
              f"Distill: {train_losses['distill']:.4f}, Feature: {train_losses['feature']:.4f})")
        print(f"Val Loss: {val_losses['total']:.4f} (Task: {val_losses['task']:.4f}, "
              f"Distill: {val_losses['distill']:.4f}, Feature: {val_losses['feature']:.4f})")
        print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"Val IoU (noise/clean): {val_metrics['iou_noise']:.4f} / {val_metrics['iou_clean']:.4f}")
        
        # Save best model
        if val_metrics["f1"] > best_val_f1 + args.early_stop_delta:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_metrics["f1"],
                "val_metrics": val_metrics,
            }, os.path.join(args.output_dir, "best_student_model.pth"))
            print(f"✓ Saved best model with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
        
        # Early stopping
        if early_stop and patience_counter >= args.early_stop_patience:
            print(f"\n=== Early Stopping at epoch {epoch} ===")
            print(f"Best Validation F1: {best_val_f1:.4f}")
            break
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(args.output_dir, f"student_checkpoint_epoch_{epoch}.pth"))
    
    # Plot and finish
    plot_training_curves(history, args.output_dir)
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best Val F1: {best_val_f1:.4f}")
    print(f"Student model saved to: {args.output_dir}")
    print(f"Training log: {log_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Student U-Net with Knowledge Distillation from Axial Attention Teacher"
    )
    
    # Data
    parser.add_argument("--data_root", type=str, default="./WADS/wads")
    parser.add_argument("--splits_json", type=str, default="./splits.json")
    parser.add_argument("--noise_label", type=int, default=110)
    parser.add_argument("--proj_h", type=int, default=64)
    parser.add_argument("--proj_w", type=int, default=1024)
    
    # Student Model
    parser.add_argument("--base_channels", type=int, default=24, 
                        help="Base channels for student (24 or 32 recommended)")
    
    # Teacher Model (with Axial Attention)
    parser.add_argument("--teacher_checkpoint", type=str, default="./outputs/best_model.pth",
                        help="Path to teacher model checkpoint")
    parser.add_argument("--teacher_base_channels", type=int, default=64,
                        help="Base channels of teacher model")
    parser.add_argument("--teacher_bilinear", action="store_true",
                        help="Whether teacher uses bilinear upsampling")
    parser.add_argument("--teacher_attn_heads", type=int, default=4,
                        help="Number of attention heads in teacher")
    
    # Distillation
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss (0-1)")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Temperature for distillation")
    parser.add_argument("--feature_weight", type=float, default=0.1,
                        help="Weight for feature matching loss")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs_student")
    parser.add_argument("--save_interval", type=int, default=10)
    
    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--early_stop_delta", type=float, default=0.0001)
    
    args = parser.parse_args()
    main(args)