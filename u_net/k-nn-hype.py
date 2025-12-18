import argparse
import json
import os
from itertools import product  # 追加

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KDTree
from tqdm import tqdm

from data_loader import VelodyneRangeProjection, WADSDataset
from model_unet_trans import UNetDenoiser


class RangeImageBackProjection:
    """Back-project range image predictions to 3D point cloud"""

    def __init__(self, proj_H=64, proj_W=1024, fov_up=2.0, fov_down=-24.9):
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)

    def back_project_with_mapping(self, pred_img, proj_idx, num_points):
        """
        Back-project predictions using direct pixel-to-point mapping

        Args:
            pred_img: (H, W) prediction image (0=noise, 1=clean)
            proj_idx: (H, W) mapping from pixels to original point indices
            num_points: Total number of points in original cloud

        Returns:
            point_labels: (N,) predicted labels for all points
        """
        # Initialize all points as clean (1) by default
        point_labels = np.ones(num_points, dtype=np.int32)

        # Map predicted labels back to points
        for i in range(self.proj_H):
            for j in range(self.proj_W):
                point_idx = proj_idx[i, j]
                if point_idx >= 0 and point_idx < num_points:
                    point_labels[point_idx] = pred_img[i, j]

        return point_labels

    def back_project_with_knn(
        self, pred_img, proj_xyz, proj_idx, proj_mask, original_points, k=5
    ):
        """
        Back-project predictions using KNN for unmapped points

        Args:
            pred_img: (H, W) prediction image
            proj_xyz: (H, W, 3) xyz coordinates in range image
            proj_idx: (H, W) pixel to point mapping
            proj_mask: (H, W) valid pixel mask
            original_points: (N, 3) original point cloud
            k: Number of nearest neighbors

        Returns:
            point_labels: (N,) predicted labels for all points
        """
        num_points = len(original_points)
        point_labels = np.ones(num_points, dtype=np.int32)
        mapped_mask = np.zeros(num_points, dtype=bool)

        # First pass: Direct mapping
        for i in range(self.proj_H):
            for j in range(self.proj_W):
                if proj_mask[i, j] > 0:
                    point_idx = proj_idx[i, j]
                    if 0 <= point_idx < num_points:
                        point_labels[point_idx] = pred_img[i, j]
                        mapped_mask[point_idx] = True

        # Find unmapped points
        unmapped_indices = np.where(~mapped_mask)[0]

        if len(unmapped_indices) > 0:
            # Get mapped points and their labels
            mapped_indices = np.where(mapped_mask)[0]

            if len(mapped_indices) > 0:
                mapped_points = original_points[mapped_indices]
                mapped_labels = point_labels[mapped_indices]

                # Build KD-Tree with mapped points
                kdtree = KDTree(mapped_points)

                # Query k nearest neighbors for unmapped points
                unmapped_points = original_points[unmapped_indices]
                distances, indices = kdtree.query(
                    unmapped_points, k=min(k, len(mapped_points))
                )

                # Assign labels by majority vote
                for i, unmapped_idx in enumerate(unmapped_indices):
                    neighbor_labels = mapped_labels[indices[i]]
                    # Majority vote
                    point_labels[unmapped_idx] = np.bincount(neighbor_labels).argmax()

        return point_labels


def evaluate_pointcloud_predictions(pred_labels, gt_labels, noise_label=250):

    """
    Evaluate point cloud predictions

    Args:
        pred_labels: (N,) predicted labels (0=noise, 1=clean)
        gt_labels: (N,) ground truth semantic labels
        noise_label: Label value indicating noise in ground truth

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Convert ground truth to binary (0=noise, 1=clean)
    gt_binary = (gt_labels != noise_label).astype(np.int32)

    # Compute confusion matrix
    cm = confusion_matrix(gt_binary, pred_labels, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    # Compute metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Noise detection metrics
    noise_precision = (
        tn / (tn + fn) if (tn + fn) > 0 else 0
    )  # How many detected noise are actually noise
    noise_recall = (
        tn / (tn + fp) if (tn + fp) > 0 else 0
    )  # How many actual noise points were detected
    noise_f1 = (
        2 * noise_precision * noise_recall / (noise_precision + noise_recall)
        if (noise_precision + noise_recall) > 0
        else 0
    )

    # Clean point metrics
    clean_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    clean_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    clean_f1 = (
        2 * clean_precision * clean_recall / (clean_precision + clean_recall)
        if (clean_precision + clean_recall) > 0
        else 0
    )

    # IoU
    iou_noise = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
    iou_clean = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    # Noise detection rate (recall for noise class)
    noise_detection_rate = noise_recall

    metrics = {
        "accuracy": accuracy,
        "noise_precision": noise_precision,
        "noise_recall": noise_recall,
        "noise_f1": noise_f1,
        "noise_detection_rate": noise_detection_rate,
        "clean_precision": clean_precision,
        "clean_recall": clean_recall,
        "clean_f1": clean_f1,
        "iou_noise": iou_noise,
        "iou_clean": iou_clean,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_noise_gt": int(tn + fp),
        "total_clean_gt": int(tp + fn),
        "total_noise_pred": int(tn + fn),
        "total_clean_pred": int(tp + fp),
    }

    return metrics, cm


def evaluate_with_knn_params(
    model, dataset, device, args, output_dir, k_values=[3, 5, 7, 9, 11], weight_options=['uniform', 'distance']
):
    """
    複数のkとweightsの組み合わせを検証し、最適なハイパーパラメータを探索
    """
    model.eval()

    projector = VelodyneRangeProjection(
        args.proj_h, args.proj_w, args.fov_up, args.fov_down
    )
    back_projector = RangeImageBackProjection(
        args.proj_h, args.proj_w, args.fov_up, args.fov_down
    )

    results = []

    print("\n" + "="*80)
    print("KNN HYPERPARAMETER SEARCH ON VALIDATION SET")
    print("="*80)
    print(f"Testing k values: {k_values}")
    print(f"Testing weights: {weight_options}")
    print(f"Dataset size: {len(dataset)} samples")
    print("="*80)

    best_f1 = 0.0
    best_params = None
    best_metrics = None

    # すべての組み合わせを試す
    for k, weights in product(k_values, weight_options):
        print(f"\nTesting: k={k}, weights='{weights}'")

        all_metrics = []

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"k={k}, weights={weights}", leave=False):
                sample = dataset[idx]
                inputs = sample["input"].unsqueeze(0).to(device)
                bin_path = sample["path"]

                # Load point cloud
                points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                xyz = points[:, :3]
                intensity = points[:, 3]

                # Load GT labels
                label_path = bin_path.replace(".bin", ".label").replace("velodyne", "labels")
                gt_labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF

                # Project
                proj_range, proj_xyz, proj_intensity, proj_mask, proj_idx = projector.project(xyz, intensity)

                # Model inference
                outputs = model(inputs)
                pred_img = torch.argmax(outputs, dim=1)[0].cpu().numpy()

                # Back-project with custom KNN (weights対応版)
                point_preds = back_project_with_knn_weighted(
                    pred_img, proj_xyz, proj_idx, proj_mask, xyz,
                    k=k, weights=weights
                )

                # Evaluate
                metrics, _ = evaluate_pointcloud_predictions(point_preds, gt_labels, noise_label=args.noise_label)
                all_metrics.append(metrics)

        # Average metrics for this parameter set
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys() if isinstance(all_metrics[0][k], (int, float))}

        noise_f1 = avg_metrics['noise_f1']
        results.append({
            'k': k,
            'weights': weights,
            'noise_f1': noise_f1,
            'noise_detection_rate': avg_metrics['noise_detection_rate'],
            'iou_noise': avg_metrics['iou_noise'],
            'clean_f1': avg_metrics['clean_f1'],
            'accuracy': avg_metrics['accuracy']
        })

        print(f"  → Noise F1: {noise_f1:.4f} | Clean F1: {avg_metrics['clean_f1']:.4f} | IoU Noise: {avg_metrics['iou_noise']:.4f}")

        # Update best
        if noise_f1 > best_f1:
            best_f1 = noise_f1
            best_params = (k, weights)
            best_metrics = avg_metrics

    # Final results table
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    print(f"{'k':>4} | {'weights':<10} | {'Noise F1':>10} | {'Clean F1':>10} | {'IoU Noise':>10} | {'Accuracy':>10}")
    print("-"*80)
    for r in sorted(results, key=lambda x: x['noise_f1'], reverse=True):
        print(f"{r['k']:4d} | {r['weights']:<10} | {r['noise_f1']:10.4f} | {r['clean_f1']:10.4f} | {r['iou_noise']:10.4f} | {r['accuracy']:10.4f}")

    print("-"*80)
    print(f"BEST PARAMETERS: k={best_params[0]}, weights='{best_params[1]}' → Noise F1 = {best_f1:.4f}")

    # Save results
    search_results = {
        "best_k": best_params[0],
        "best_weights": best_params[1],
        "best_noise_f1": best_f1,
        "all_results": results
    }
    with open(os.path.join(output_dir, "knn_hyperparameter_search.json"), "w") as f:
        json.dump(search_results, f, indent=2)

    print(f"\nSearch results saved to {output_dir}/knn_hyperparameter_search.json")

    return best_params, best_metrics


def back_project_with_knn_weighted(
    pred_img, proj_xyz, proj_idx, proj_mask, original_points, k=5, weights='uniform'
):
    """
    weights対応版のKNNバックプロジェクション
    weights: 'uniform' or 'distance'
    """
    num_points = len(original_points)
    point_labels = np.ones(num_points, dtype=np.int32)
    mapped_mask = np.zeros(num_points, dtype=bool)

    # Direct mapping
    for i in range(pred_img.shape[0]):
        for j in range(pred_img.shape[1]):
            if proj_mask[i, j] > 0:
                point_idx = proj_idx[i, j]
                if 0 <= point_idx < num_points:
                    point_labels[point_idx] = pred_img[i, j]
                    mapped_mask[point_idx] = True

    unmapped_indices = np.where(~mapped_mask)[0]
    if len(unmapped_indices) == 0:
        return point_labels

    mapped_indices = np.where(mapped_mask)[0]
    mapped_points = original_points[mapped_indices]
    mapped_labels = point_labels[mapped_indices]

    kdtree = KDTree(mapped_points)
    unmapped_points = original_points[unmapped_indices]
    k = min(k, len(mapped_points))
    distances, indices = kdtree.query(unmapped_points, k=k)

    for i, unmapped_idx in enumerate(unmapped_indices):
        neighbor_labels = mapped_labels[indices[i]]
        neighbor_dists = distances[i]

        if weights == 'uniform':
            voted = np.bincount(neighbor_labels).argmax()
        elif weights == 'distance':
            # 距離の逆数で重み付け（ゼロ除算回避）
            inv_dists = 1.0 / (neighbor_dists + 1e-8)
            weights_sum = inv_dists.sum()
            prob_class_0 = (inv_dists[neighbor_labels == 0].sum()) / weights_sum
            prob_class_1 = (inv_dists[neighbor_labels == 1].sum()) / weights_sum
            voted = 0 if prob_class_0 > prob_class_1 else 1
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")

        point_labels[unmapped_idx] = voted

    return point_labels


# main関数を修正
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = UNetDenoiser(
        in_channels=5, num_classes=2,
        base_channels=args.base_channels, bilinear=args.bilinear
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load splits
    with open(args.splits_json, "r") as f:
        splits = json.load(f)

    # 検証用データセットを作成（ここではval_splitを使用、なければtestの半分などでも可）
    if "val" in splits and len(splits["val"]) > 0:
        val_split = splits["val"]
        print(f"Using validation split with {len(val_split)} samples")
    else:
        # valがなければtestの先頭20%を検証用に
        val_split = splits["test"][:max(1, len(splits["test"]) // 5)]
        print(f"No val split found. Using {len(val_split)} samples from test as validation")

    val_dataset = WADSDataset(
        args.data_root, val_split,
        proj_H=args.proj_h, proj_W=args.proj_w,
        fov_up=args.fov_up, fov_down=args.fov_down,
        noise_label=args.noise_label,
    )

    # === ハイパーパラメータ探索実行 ===
    best_k, best_weights = evaluate_with_knn_params(
        model, val_dataset, device, args, args.output_dir,
        k_values=[3, 5, 7, 9, 11, 15],
        weight_options=['uniform', 'distance']
    )[0]

    print(f"\nRecommended KNN parameters: k={best_k}, weights='{best_weights}'")
    print("You can now run evaluation on full test set with these parameters.")

    # 必要なら、ベストパラメータでテストセット全体を再評価するコードも追加可能

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Back-projection and evaluation for U-Net"
    )

    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs/best_model.pth",
        help="Path to model checkpoint",
    )

    # Data
    parser.add_argument("--data_root", type=str, default="./WADS/wads")
    parser.add_argument("--splits_json", type=str, default="./splits.json")
    parser.add_argument("--noise_label", type=int, default=110)

    # Model config (U-Net specific)
    parser.add_argument("--proj_h", type=int, default=64)
    parser.add_argument("--proj_w", type=int, default=1024)
    parser.add_argument(
        "--base_channels", type=int, default=64, help="Base number of channels in U-Net"
    )
    parser.add_argument(
        "--bilinear",
        action="store_true",
        help="Use bilinear upsampling instead of transposed conv",
    )
    parser.add_argument("--fov_up", type=float, default=2.0)
    parser.add_argument("--fov_down", type=float, default=-24.9)

    # Back-projection method
    parser.add_argument(
        "--use_knn",
        default=True,
        action="store_true",
        help="Use KNN for unmapped points",
    )
    parser.add_argument(
        "--knn_k", type=int, default=5, help="Number of nearest neighbors for KNN"
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predicted labels for each point cloud",
    )

    args = parser.parse_args()

    main(args)
