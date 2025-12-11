import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer_denoiser import TransformerPointCloudDenoiser, print_model_info
from wads_dataset_transformer import WADSDatasetTransformer


def train():
    # ───────────────────────────────
    # 設定
    # ───────────────────────────────
    root = "./WADS"
    splits_file = "splits.json"
    num_points = 10000  # Transformer用に変更
    batch_size = 8  # Transformerは軽いのでバッチサイズを増やせる
    epochs = 100
    lr = 1e-4
    weight_decay = 1e-4
    
    # Transformer設定
    num_patches = 100
    embed_dim = 128
    num_heads = 8
    num_layers = 6
    ffn_dim = 512
    pool_type = 'mean'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # CUDAメモリの最適化設定
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        print(f"Initial Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Initial Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # ───────────────────────────────
    # Dataset / DataLoader
    # ───────────────────────────────
    print("\n" + "=" * 60)
    print("Loading WADS Dataset (Transformer version)...")
    print("=" * 60)

    train_dataset = WADSDatasetTransformer(
        root, split="train", num_points=num_points, splits_file=splits_file
    )
    val_dataset = WADSDatasetTransformer(
        root, split="val", num_points=num_points, splits_file=splits_file
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ───────────────────────────────
    # Model
    # ───────────────────────────────
    print("\n" + "=" * 60)
    print("Initializing Transformer Model...")
    print("=" * 60)
    
    model = TransformerPointCloudDenoiser(
        num_points=num_points,
        num_patches=num_patches,
        in_dim=4,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        num_classes=2,
        pool_type=pool_type,
        dropout=0.1
    ).to(device)
    
    print_model_info(model)

    # クラス不均衡対策：ノイズ点（クラス1）に重みを付ける
    class_weights = torch.tensor([1.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ───────────────────────────────
    # 学習開始時刻を記録
    # ───────────────────────────────
    start_time = time.time()
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")

    # ───────────────────────────────
    # Training Loop
    # ───────────────────────────────
    best_val_loss = float("inf")
    best_val_acc = 0.0

    # 学習曲線用のリスト
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    learning_rates = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_points = 0

        # tqdmで進捗バーを表示
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=100)

        for i, (points, labels) in enumerate(pbar):
            points = points.to(device)  # (B, N, 4)
            labels = labels.to(device)  # (B, N)

            optimizer.zero_grad()

            # Forward
            logits = model(points)  # (B, N, 2)
            logits = logits.permute(0, 2, 1)  # (B, 2, N)

            # Loss
            loss = criterion(logits, labels)

            # Backprop
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            preds = logits.argmax(dim=1)  # (B, N)
            total_correct += (preds == labels).sum().item()
            total_points += labels.numel()

            # 進捗バーに現在の統計を表示
            current_loss = total_loss / (i + 1)
            current_acc = total_correct / total_points
            pbar.set_postfix(
                {"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"}
            )

            # メモリ解放
            del points, labels, logits, preds, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_points
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}"
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

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", ncols=100)

        with torch.no_grad():
            for points, labels in val_pbar:
                points = points.to(device)
                labels = labels.to(device)

                logits = model(points)
                logits = logits.permute(0, 2, 1)

                loss = criterion(logits, labels)
                val_loss += loss.item()

                # accuracy
                preds = logits.argmax(dim=1)  # (B, N)
                correct += (preds == labels).sum().item()
                total += labels.numel()

                # クラスごとの精度
                for c in range(2):
                    class_mask = labels == c
                    class_correct[c] += ((preds == labels) & class_mask).sum().item()
                    class_total[c] += class_mask.sum().item()

                # メモリ解放
                del points, labels, logits, preds, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # クラスごとの精度を表示
        class0_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
        class1_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0

        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(
            f"  Class 0 (Normal) Acc: {class0_acc:.4f} | Class 1 (Noise) Acc: {class1_acc:.4f}"
        )

        # Learning rate scheduler step
        scheduler.step()

        # Save best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            save_path = "checkpoints/transformer_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ★ Best model saved: {save_path} (Val Acc: {val_acc:.4f})")

    # ───────────────────────────────
    # Save final model
    # ───────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/transformer_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved to {save_path}")

    # ───────────────────────────────
    # 学習曲線データをJSONファイルに保存
    # ───────────────────────────────
    os.makedirs("results", exist_ok=True)
    metrics_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "learning_rates": learning_rates,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_points": num_points,
        "num_patches": num_patches,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ffn_dim": ffn_dim,
        "dataset": "WADS",
        "model": "Transformer"
    }

    json_path = "results/transformer_training_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Training metrics saved to {json_path}")

    # ───────────────────────────────
    # 学習曲線の描画
    # ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    axes[0].plot(
        range(1, epochs + 1), train_losses, label="Train Loss", marker="o", linewidth=2
    )
    axes[0].plot(
        range(1, epochs + 1),
        val_losses,
        label="Validation Loss",
        marker="s",
        linewidth=2,
    )
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss (Transformer)", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(
        range(1, epochs + 1), train_accs, label="Train Acc", marker="o", linewidth=2
    )
    axes[1].plot(
        range(1, epochs + 1),
        val_accs,
        label="Validation Acc",
        marker="s",
        linewidth=2,
    )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training and Validation Accuracy (Transformer)", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()

    # 学習曲線を保存
    curve_path = "results/transformer_learning_curve.png"
    plt.savefig(curve_path, dpi=150)
    print(f"Learning curve saved to {curve_path}")
    plt.close()

    # ───────────────────────────────
    # トータル学習時間を表示
    # ───────────────────────────────
    end_time = time.time()
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(
        f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time:.2f} seconds)"
    )
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train()