import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader_stf import create_dataloaders
from model import UNetDenoiser


# =================================================
# Loss Functions
# =================================================


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if mask is not None:
            focal_loss = focal_loss * mask
            return focal_loss.sum() / (mask.sum() + 1e-6)

        return focal_loss.mean()


class MaskedCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss with mask support"""

    def forward(self, inputs, targets, mask=None):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")

        if mask is not None:
            ce_loss = ce_loss * mask
            return ce_loss.sum() / (mask.sum() + 1e-6)

        return ce_loss.mean()


# =================================================
# Metrics
# =================================================


def compute_metrics(preds, labels, mask):
    mask_flat = mask.flatten().cpu().numpy() > 0
    preds_flat = preds.flatten().cpu().numpy()[mask_flat]
    labels_flat = labels.flatten().cpu().numpy()[mask_flat]

    if len(preds_flat) == 0:
        return dict.fromkeys(
            ["accuracy", "precision", "recall", "f1", "iou_noise", "iou_clean"], 0.0
        )

    accuracy = (preds_flat == labels_flat).mean()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, preds_flat, average="binary", zero_division=0
    )

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


# =================================================
# Train / Val
# =================================================


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_masks = [], [], []

    for batch in tqdm(dataloader, desc=f"Train Epoch {epoch}"):
        inputs = batch["input"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels, masks)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())
        all_masks.append(masks.detach())

        total_loss += loss.item()

    metrics = compute_metrics(
        torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_masks)
    )

    return total_loss / len(dataloader), metrics


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_masks = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val Epoch {epoch}"):
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels, masks)

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
            all_masks.append(masks)

            total_loss += loss.item()

    metrics = compute_metrics(
        torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_masks)
    )

    return total_loss / len(dataloader), metrics


# =================================================
# Main
# =================================================


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    print("Loading data...")
    train_loader, val_loader, _ = create_dataloaders(
        args.data_root,
        # args.splits_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        proj_H=args.proj_h,
        proj_W=args.proj_w,
        noise_label=args.noise_label,
    )

    print("Creating model...")
    model = UNetDenoiser(
        in_channels=5,
        num_classes=2,
        base_channels=args.base_channels,
        bilinear=args.bilinear,
    ).to(device)

    # =================================================
    # 🔥 Load SSL pretrained weights
    # =================================================
    print("Loading SSL pretrained model...")
    ssl_ckpt = torch.load(
        os.path.join(args.ssl_ckpt_dir, "best_model_ssl.pth"),
        map_location="cpu",
        weights_only=False
    )

    ssl_state = ssl_ckpt["model_state_dict"]

    # ❌ remove output layer weights
    ssl_state = {k: v for k, v in ssl_state.items() if not k.startswith("outc")}

    model.load_state_dict(ssl_state, strict=False)
    print("SSL weights loaded (output layer reinitialized).")

    # =================================================

    if args.use_focal_loss:
        criterion = FocalLoss(args.focal_alpha, args.focal_gamma)
    else:
        criterion = MaskedCrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)

        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        for k, v in train_metrics.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

        # print(
        #     f"Epoch {epoch}: "
        #     f"Train F1={train_metrics['f1']:.4f}, "
        #     f"Val F1={val_metrics['f1']:.4f}"
        # )
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )
        print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(
            f"Val IoU (noise/clean): {val_metrics['iou_noise']:.4f} / {val_metrics['iou_clean']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print("Saved best model.")

    writer.close()
    print(f"Training finished. Best Val F1 = {best_val_f1:.4f}")


# =================================================
# Args
# =================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./semanticSTF")
    # parser.add_argument("--splits_json", type=str, default="./splits.json")
    parser.add_argument("--noise_label", type=int, default=20)

    parser.add_argument("--proj_h", type=int, default=64)
    parser.add_argument("--proj_w", type=int, default=1024)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--bilinear", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--ssl_ckpt_dir", type=str, default="./outputs_ssl")
    parser.add_argument("--output_dir", type=str, default="./outputs_finetune")

    args = parser.parse_args()
    main(args)

# outputのディレクトリと学習状況を出力させる
