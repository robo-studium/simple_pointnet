import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pointnet2 import PointNet2
from semanticstf_dataset_v2 import SemanticSTFDataset


def train():
    # ───────────────────────────────
    # 設定
    # ───────────────────────────────
    root = "./SemanticSTF"
    num_points = 32768
    batch_size = 4
    epochs = 20  # PointNet++は複雑なので少し長めに
    lr = 1e-3
    weight_decay = 1e-4  # 正則化を追加

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ───────────────────────────────
    # Dataset / DataLoader
    # ───────────────────────────────
    train_dataset = SemanticSTFDataset(root, split="train", num_points=num_points)
    val_dataset = SemanticSTFDataset(root, split="val", num_points=num_points)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ───────────────────────────────
    # Model
    # ───────────────────────────────
    model = PointNet2(in_dim=5, num_classes=2).to(device)

    # クラス不均衡対策：ノイズ点（クラス1）に重みを付ける
    # データセット全体のクラス比率を調べてから設定するのが理想的
    # ここでは仮にノイズ点が少ないと仮定して重み10倍に設定
    class_weights = torch.tensor([1.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler（オプション）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ───────────────────────────────
    # Training Loop
    # ───────────────────────────────
    best_val_loss = float("inf")

    # 学習曲線用のリスト
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_points = 0

        for i, (points, labels) in enumerate(train_loader):
            points = points.to(device)  # (B, N, 5)
            labels = labels.to(device)  # (B, N)

            optimizer.zero_grad()

            # Forward
            logits = model(points)  # (B, N, 2)
            logits = logits.permute(0, 2, 1)  # (B, 2, N) ← CE はこの形を要求

            # Loss
            loss = criterion(logits, labels)

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            preds = logits.argmax(dim=1)  # (B, N)
            total_correct += (preds == labels).sum().item()
            total_points += labels.numel()

            if (i + 1) % 20 == 0:
                acc = total_correct / total_points
                print(
                    f"[Epoch {epoch+1}/{epochs}] Step {i+1}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f} Acc: {acc:.4f}"
                )

        avg_train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_points
        train_losses.append(avg_train_loss)  # 学習曲線用に記録
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )

        # ───────────────────────────────
        # Validation
        # ───────────────────────────────
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        # クラスごとの精度を追跡
        class_correct = [0, 0]
        class_total = [0, 0]

        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device)
                labels = labels.to(device)

                logits = model(points)
                logits = logits.permute(0, 2, 1)

                loss = criterion(logits, labels)
                val_loss += loss.item()

                # accuracy（全点で計算）
                preds = logits.argmax(dim=1)  # (B, N)
                correct += (preds == labels).sum().item()
                total += labels.numel()

                # クラスごとの精度
                for c in range(2):
                    class_mask = labels == c
                    class_correct[c] += ((preds == labels) & class_mask).sum().item()
                    class_total[c] += class_mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)  # 学習曲線用に記録

        # クラスごとの精度を表示
        class0_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
        class1_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0

        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(
            f"  Class 0 (Normal) Acc: {class0_acc:.4f} | Class 1 (Noise) Acc: {class1_acc:.4f}"
        )

        # Learning rate scheduler step
        scheduler.step()

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            save_path = "checkpoints/pointnet2_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ★ Best model saved: {save_path}")

    # ───────────────────────────────
    # Save final model
    # ───────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/pointnet2_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved to {save_path}")

    # ───────────────────────────────
    # 学習曲線の描画
    # ───────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, epochs + 1), train_losses, label="Train Loss", marker="o", linewidth=2
    )
    plt.plot(
        range(1, epochs + 1),
        val_losses,
        label="Validation Loss",
        marker="s",
        linewidth=2,
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss Curve", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 学習曲線を保存
    os.makedirs("results", exist_ok=True)
    curve_path = "results/learning_curve.png"
    plt.savefig(curve_path, dpi=150)
    print(f"Learning curve saved to {curve_path}")

    # 学習曲線を表示
    plt.show()
    print("Learning curve displayed!")


if __name__ == "__main__":
    train()
