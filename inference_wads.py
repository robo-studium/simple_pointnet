import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from pointnet2 import PointNet2
from wads_dataset import WADSDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
#   点群読み込み関数
# ============================
def load_pointcloud(filepath):
    """
    WADS/KITTI系の .bin ファイルを読み込み、range特徴量を追加
    → (N, 5): x, y, z, intensity, range
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith(".bin"):
        # 4列として読み込み
        with open(filepath, "rb") as f:
            pc = np.fromfile(f, dtype=np.float32).reshape(-1, 4)

        # intensity を 0~1 に正規化
        pc[:, 3] = pc[:, 3] / 255.0

        # range を計算して追加
        range_vals = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
        range_vals = range_vals.reshape(-1, 1)
        pc = np.concatenate([pc, range_vals], axis=1)  # (N, 5)

    elif filepath.endswith(".npy"):
        pc = np.load(filepath)
        if pc.shape[1] < 4:
            raise ValueError("npy file must have at least 4 channels (x,y,z,intensity)")
        pc = pc[:, :4].astype(np.float32)

        # .npyでintensityが0~255のままの可能性が高いので正規化
        if pc[:, 3].max() > 1.1:
            pc[:, 3] = pc[:, 3] / 255.0

        # range を計算して追加
        range_vals = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
        range_vals = range_vals.reshape(-1, 1)
        pc = np.concatenate([pc, range_vals], axis=1)  # (N, 5)

    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    return pc.astype(np.float32)


# ============================
#   ランダムサンプリング
# ============================
def sample_points(pc, num_points=32768):
    """
    点群とラベルを同じインデックスでサンプリングするため、idxも返す
    """
    N = pc.shape[0]
    if N == 0:
        return np.zeros((num_points, 5), dtype=np.float32), np.zeros(
            num_points, dtype=np.int64
        )

    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)

    return pc[idx], idx


# ============================
#   テスト評価関数
# ============================
def evaluate_test_set(model, test_loader, device):
    """
    テストデータセットに対して推論を実行し、混同行列と再現率を計算

    Args:
        model: 学習済みモデル
        test_loader: テストデータのDataLoader
        device: 実行デバイス (cuda/cpu)

    Returns:
        conf_matrix: 混同行列 (2x2 numpy array)
        recall: 再現率 (float)
    """
    model.eval()

    all_preds = []
    all_labels = []

    print("\n=== テストデータでの評価を開始 ===")

    with torch.no_grad():
        for points, labels in tqdm(test_loader, desc="推論中", unit="batch"):
            points = points.to(device)  # (B, N, 5)
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

    # 再現率を計算 (クラス1=ノイズ点に対する再現率)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)

    return conf_matrix, recall


# ============================
#   結果表示関数
# ============================
def print_evaluation_results(conf_matrix, recall):
    """
    混同行列と再現率を見やすく表示

    Args:
        conf_matrix: 混同行列 (2x2 numpy array)
        recall: 再現率 (float)
    """
    print("\n" + "=" * 50)
    print("評価結果")
    print("=" * 50)

    print("\n【混同行列】")
    print("                予測")
    print("              0 (正常)  1 (ノイズ)")
    print(f"実際 0 (正常)   {conf_matrix[0, 0]:8d}  {conf_matrix[0, 1]:8d}")
    print(f"     1 (ノイズ) {conf_matrix[1, 0]:8d}  {conf_matrix[1, 1]:8d}")

    # 各指標を計算
    tn, fp, fn, tp = conf_matrix.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("\n【評価指標】")
    print(f"Accuracy (精度):     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Recall (再現率):     {recall:.4f} ({recall*100:.2f}%)")
    print(f"Precision (適合率):  {precision:.4f} ({precision*100:.2f}%)")
    print(f"F1 Score:           {f1_score:.4f}")

    print("\n【詳細統計】")
    print(f"True Positives (TP):  {tp:8d}  (ノイズを正しくノイズと予測)")
    print(f"True Negatives (TN):  {tn:8d}  (正常を正しく正常と予測)")
    print(f"False Positives (FP): {fp:8d}  (正常を誤ってノイズと予測)")
    print(f"False Negatives (FN): {fn:8d}  (ノイズを誤って正常と予測)")
    print("=" * 50 + "\n")


# ============================
#   メイン実行関数
# ============================
def run_inference(
    data_root: str = "./WADS",
    splits_file: str = "splits.json",
    checkpoint: str = "checkpoints/pointnet2_best.pth",
    num_points: int = 16384,
    batch_size: int = 2,
    num_workers: int = 4,
    save_results: bool = False,
    output_path: str = "results/test_results.txt",
):
    """
    推論のメイン処理（argparse を使わない版）
    """
    print(f"使用デバイス: {device}")

    # モデルの読み込み
    print(f"\nモデルを読み込み中: {checkpoint}")
    model = PointNet2(in_dim=5, num_classes=2, num_points=num_points).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print("モデルの読み込み完了")

    # テストデータセットの準備
    print(f"\nテストデータセットを準備中: {data_root}")
    test_dataset = WADSDataset(
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
    conf_matrix, recall = evaluate_test_set(model, test_loader, device)

    # 結果表示
    print_evaluation_results(conf_matrix, recall)

    # 結果をファイルに保存（オプション）
    if save_results:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write("混同行列:\n")
            f.write(f"{conf_matrix}\n\n")
            f.write(f"再現率 (Recall): {recall:.4f}\n")
        print(f"結果を保存しました: {output_path}")


if __name__ == "__main__":
    # argparse を完全に削除して、直接デフォルト値で呼び出し
    run_inference()
