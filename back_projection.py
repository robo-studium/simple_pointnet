import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KDTree
from tqdm import tqdm

from data_loader import VelodyneRangeProjection, WADSDataset
from model import VisionTransformerDenoiser


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


def evaluate_test_set(model, test_dataset, device, args, output_dir):
    """Evaluate model on test set with back-projection"""

    model.eval()

    # Results storage
    all_metrics = []
    sequence_metrics = {}

    projector = VelodyneRangeProjection(
        args.proj_h, args.proj_w, args.fov_up, args.fov_down
    )

    back_projector = RangeImageBackProjection(
        args.proj_h, args.proj_w, args.fov_up, args.fov_down
    )

    print("Evaluating test set with back-projection...")

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Processing"):
            # Get data
            sample = test_dataset[idx]
            inputs = sample["input"].unsqueeze(0).to(device)
            sequence = sample["sequence"]
            bin_path = sample["path"]

            # Load original data
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            xyz = points[:, :3]
            intensity = points[:, 3]

            # Load ground truth labels
            label_path = bin_path.replace(".bin", ".label").replace(
                "velodyne", "labels"
            )
            gt_labels = np.fromfile(label_path, dtype=np.uint32)
            gt_labels = gt_labels & 0xFFFF  # Extract semantic label

            # Project to range image (to get proj_idx)
            proj_range, proj_xyz, proj_intensity, proj_mask, proj_idx = (
                projector.project(xyz, intensity)
            )

            # Model prediction
            outputs = model(inputs)
            pred_img = torch.argmax(outputs, dim=1)[0].cpu().numpy()

            # Back-project to point cloud
            if args.use_knn:
                point_preds = back_projector.back_project_with_knn(
                    pred_img, proj_xyz, proj_idx, proj_mask, xyz, k=args.knn_k
                )
            else:
                point_preds = back_projector.back_project_with_mapping(
                    pred_img, proj_idx, len(xyz)
                )

            # Evaluate
            metrics, cm = evaluate_pointcloud_predictions(
                point_preds, gt_labels, noise_label=args.noise_label
            )

            all_metrics.append(metrics)

            # Track by sequence
            if sequence not in sequence_metrics:
                sequence_metrics[sequence] = []
            sequence_metrics[sequence].append(metrics)

            # Save predictions (optional)
            if args.save_predictions:
                pred_dir = os.path.join(output_dir, "predictions", str(sequence))
                os.makedirs(pred_dir, exist_ok=True)

                # Save predicted labels
                pred_file = os.path.join(
                    pred_dir, os.path.basename(bin_path).replace(".bin", ".pred")
                )
                point_preds.astype(np.uint32).tofile(pred_file)

    # Aggregate metrics
    print("\n" + "=" * 70)
    print("POINT CLOUD EVALUATION RESULTS")
    print("=" * 70)

    # Overall metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float)):
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    print("\n--- Overall Metrics ---")
    print(f"Total samples: {len(all_metrics)}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"\n--- Noise Detection ---")
    print(f"Noise Detection Rate (Recall): {avg_metrics['noise_detection_rate']:.4f}")
    print(f"Noise Precision: {avg_metrics['noise_precision']:.4f}")
    print(f"Noise F1-Score: {avg_metrics['noise_f1']:.4f}")
    print(f"Noise IoU: {avg_metrics['iou_noise']:.4f}")
    print(f"\n--- Clean Point Detection ---")
    print(f"Clean Precision: {avg_metrics['clean_precision']:.4f}")
    print(f"Clean Recall: {avg_metrics['clean_recall']:.4f}")
    print(f"Clean F1-Score: {avg_metrics['clean_f1']:.4f}")
    print(f"Clean IoU: {avg_metrics['iou_clean']:.4f}")
    print(f"\n--- Point Statistics ---")
    print(f"Avg. GT Noise Points: {avg_metrics['total_noise_gt']:.0f}")
    print(f"Avg. GT Clean Points: {avg_metrics['total_clean_gt']:.0f}")
    print(f"Avg. Predicted Noise: {avg_metrics['total_noise_pred']:.0f}")
    print(f"Avg. Predicted Clean: {avg_metrics['total_clean_pred']:.0f}")

    # Per-sequence metrics
    print("\n--- Per-Sequence Metrics ---")
    for seq in sorted(sequence_metrics.keys()):
        seq_data = sequence_metrics[seq]
        seq_avg = {
            k: np.mean([m[k] for m in seq_data])
            for k in ["accuracy", "noise_detection_rate", "noise_f1"]
        }
        print(
            f"Sequence {seq}: Acc={seq_avg['accuracy']:.4f}, "
            f"Noise Detection Rate={seq_avg['noise_detection_rate']:.4f}, "
            f"Noise F1={seq_avg['noise_f1']:.4f}"
        )

    print("=" * 70)

    # Save results
    results = {
        "overall_metrics": avg_metrics,
        "per_sequence_metrics": {
            str(seq): {
                k: float(np.mean([m[k] for m in data]))
                for k in all_metrics[0].keys()
                if isinstance(all_metrics[0][k], (int, float))
            }
            for seq, data in sequence_metrics.items()
        },
        "config": vars(args),
    }

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/evaluation_results.json")

    return avg_metrics, sequence_metrics


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = VisionTransformerDenoiser(
        img_h=args.proj_h,
        img_w=args.proj_w,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        in_channels=5,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
        num_classes=2,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load test dataset
    print("Loading test dataset...")
    with open(args.splits_json, "r") as f:
        splits = json.load(f)

    test_dataset = WADSDataset(
        args.data_root,
        splits["test"],
        proj_H=args.proj_h,
        proj_W=args.proj_w,
        fov_up=args.fov_up,
        fov_down=args.fov_down,
        noise_label=args.noise_label,
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Evaluate
    avg_metrics, seq_metrics = evaluate_test_set(
        model, test_dataset, device, args, args.output_dir
    )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Back-projection and evaluation")

    # Model checkpoint
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Data
    parser.add_argument("--data_root", type=str, default="./WADS/wads")
    parser.add_argument("--splits_json", type=str, default="./splits.json")
    parser.add_argument("--noise_label", type=int, default=110)

    # Model config
    parser.add_argument("--proj_h", type=int, default=64)
    parser.add_argument("--proj_w", type=int, default=1024)
    parser.add_argument("--patch_h", type=int, default=8)
    parser.add_argument("--patch_w", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--fov_up", type=float, default=2.0)
    parser.add_argument("--fov_down", type=float, default=-24.9)

    # Back-projection method
    parser.add_argument(
        "--use_knn", action="store_true", help="Use KNN for unmapped points"
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
