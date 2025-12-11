import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer_denoiser import TransformerPointCloudDenoiser
from wads_dataset_transformer import WADSDatasetTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
#   テスト評価関数
# ============================
def evaluate_test_set(model, test_loader, device):
    """
    テストデータセットに対して推論を実行し、評価指標を計算

    Args:
        model: 学習済みモデル
        test_loader: テストデータのDataLoader
        device: 実行デバイス (cuda/cpu)

    Returns:
        conf_matrix: 混同行列 (2x2 numpy array)
        metrics: 評価指標の辞書
    """
    model.eval()

    all_preds = []
    all_labels = []

    print("\n=== テストデータでの評価を開始 ===")

    with torch.no_grad():
        for points, labels in tqdm(test_loader, desc="推論中", unit="batch"):
            points = points.to(device)  # (B, N, 4)
            labels = labels.to(device)  # (B, N)

            # モデルの推論
            logits = model(points)  # (B, N, 2)
            preds = logits.argmax(dim=-1)  # (B, N)

            # バッチ内の全点を収集
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())

    # 全データを結合
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 混同行列を計算
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    # 各種評価指標を計算
    tn, fp, fn, tp = conf_matrix.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    
    # クラスごとの精度
    class0_correct = (all_preds[all_labels == 0] == 0).sum()
    class0_total = (all_labels == 0).sum()
    class0_acc = class0_correct / class0_total if class0_total > 0 else 0
    
    class1_correct = (all_preds[all_labels == 1] == 1).sum()
    class1_total = (all_labels == 1).sum()
    class1_acc = class1_correct / class1_total if class1_total > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'class0_accuracy': class0_acc,
        'class1_accuracy': class1_acc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    return conf_matrix, metrics


# ============================
#   結果表示関数
# ============================
def print_evaluation_results(conf_matrix, metrics):
    """
    混同行列と評価指標を見やすく表示

    Args:
        conf_matrix: 混同行列 (2x2 numpy array)
        metrics: 評価指標の辞書
    """
    print("\n" + "=" * 50)
    print("評価結果 (Transformer Model)")
    print("=" * 50)

    print("\n【混同行列】")
    print("                予測")
    print("              0 (正常)  1 (ノイズ)")
    print(f"実際 0 (正常)   {conf_matrix[0, 0]:8d}  {conf_matrix[0, 1]:8d}")
    print(f"     1 (ノイズ) {conf_matrix[1, 0]:8d}  {conf_matrix[1, 1]:8d}")

    print("\n【評価指標】")
    print(f"Accuracy (精度):     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Recall (再現率):     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"Precision (適合率):  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"F1 Score:           {metrics['f1_score']:.4f}")

    print("\n【クラスごとの精度】")
    print(f"Class 0 (正常) Accuracy:  {metrics['class0_accuracy']:.4f} ({metrics['class0_accuracy']*100:.2f}%)")
    print(f"Class 1 (ノイズ) Accuracy: {metrics['class1_accuracy']:.4f} ({metrics['class1_accuracy']*100:.2f}%)")

    print("\n【詳細統計】")
    print(f"True Positives (TP):  {metrics['tp']:8d}  (ノイズを正しくノイズと予測)")
    print(f"True Negatives (TN):  {metrics['tn']:8d}  (正常を正しく正常と予測)")
    print(f"False Positives (FP): {metrics['fp']:8d}  (正常を誤ってノイズと予測)")
    print(f"False Negatives (FN): {metrics['fn']:8d}  (ノイズを誤って正常と予測)")
    print("=" * 50 + "\n")


# ============================
#   メイン実行関数
# ============================
def run_inference(
    data_root: str = "./WADS",
    splits_file: str = "splits.json",
    checkpoint: str = "checkpoints/transformer_best.pth",
    num_points: int = 10000,
    batch_size: int = 8,
    num_workers: int = 4,
    save_results: bool = True,
    output_path: str = "results/transformer_test_results.txt",
    # Model parameters
    num_patches: int = 100,
    embed_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 6,
    ffn_dim: int = 512,
    pool_type: str = 'mean',
):
    """
    Transformer モデルでの推論のメイン処理
    """
    print(f"使用デバイス: {device}")

    # モデルの読み込み
    print(f"\nTransformer モデルを読み込み中: {checkpoint}")
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
    
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print("モデルの読み込み完了")

    # モデル情報表示
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # テストデータセットの準備
    print(f"\nテストデータセットを準備中: {data_root}")
    test_dataset = WADSDatasetTransformer(
        root_dir=data_root,
        split="test",
        num_points=num_points,
        splits_file=splits_file,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"テストデータ数: {len(test_dataset)} samples")

    # 評価実行
    conf_matrix, metrics = evaluate_test_set(model, test_loader, device)

    # 結果表示
    print_evaluation_results(conf_matrix, metrics)

    # 結果をファイルに保存
    if save_results:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("Transformer Model - Test Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("【モデル設定】\n")
            f.write(f"Checkpoint: {checkpoint}\n")
            f.write(f"Number of points: {num_points}\n")
            f.write(f"Number of patches: {num_patches}\n")
            f.write(f"Embedding dimension: {embed_dim}\n")
            f.write(f"Number of heads: {num_heads}\n")
            f.write(f"Number of layers: {num_layers}\n")
            f.write(f"FFN dimension: {ffn_dim}\n")
            f.write(f"Total parameters: {total_params:,}\n\n")
            
            f.write("【混同行列】\n")
            f.write(f"                予測\n")
            f.write(f"              0 (正常)  1 (ノイズ)\n")
            f.write(f"実際 0 (正常)   {conf_matrix[0, 0]:8d}  {conf_matrix[0, 1]:8d}\n")
            f.write(f"     1 (ノイズ) {conf_matrix[1, 0]:8d}  {conf_matrix[1, 1]:8d}\n\n")
            
            f.write("【評価指標】\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
            f.write(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
            f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n\n")
            
            f.write("【クラスごとの精度】\n")
            f.write(f"Class 0 (正常) Accuracy:  {metrics['class0_accuracy']:.4f}\n")
            f.write(f"Class 1 (ノイズ) Accuracy: {metrics['class1_accuracy']:.4f}\n\n")
            
            f.write("【詳細統計】\n")
            f.write(f"True Positives (TP):  {metrics['tp']:8d}\n")
            f.write(f"True Negatives (TN):  {metrics['tn']:8d}\n")
            f.write(f"False Positives (FP): {metrics['fp']:8d}\n")
            f.write(f"False Negatives (FN): {metrics['fn']:8d}\n")
            
        print(f"結果を保存しました: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Model Inference for WADS Dataset")
    
    parser.add_argument("--data_root", type=str, default="./WADS",
                        help="WADS dataset root directory")
    parser.add_argument("--splits_file", type=str, default="splits.json",
                        help="Path to splits JSON file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/transformer_best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Number of points to sample")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Save results to file")
    parser.add_argument("--output_path", type=str, default="results/transformer_test_results.txt",
                        help="Output file path for results")
    
    # Model parameters
    parser.add_argument("--num_patches", type=int, default=100,
                        help="Number of patches")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--ffn_dim", type=int, default=512,
                        help="FFN hidden dimension")
    parser.add_argument("--pool_type", type=str, default='mean', choices=['mean', 'max'],
                        help="Pooling type for patch tokenization")
    
    args = parser.parse_args()
    
    run_inference(
        data_root=args.data_root,
        splits_file=args.splits_file,
        checkpoint=args.checkpoint,
        num_points=args.num_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_results=args.save_results,
        output_path=args.output_path,
        num_patches=args.num_patches,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        pool_type=args.pool_type,
    )