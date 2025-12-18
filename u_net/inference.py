import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from data_loader import VelodyneRangeProjection, create_dataloaders
from model_unet_trans import UNetDenoiser


def visualize_prediction(input_img, label_img, pred_img, mask, save_path=None):
    """Visualize input range image, ground truth, and prediction"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Colormap for invalid pixels
    cmap_jet = cm.get_cmap("jet").copy()
    cmap_jet.set_bad(color="lightgray")

    # Extract range image (channel 4)
    range_img = input_img[4].cpu().numpy()
    range_img[mask.cpu().numpy() == 0] = np.nan

    # Extract intensity (channel 3)
    intensity_img = input_img[3].cpu().numpy()
    intensity_img[mask.cpu().numpy() == 0] = np.nan

    # Labels and predictions
    label_vis = label_img.cpu().numpy().astype(float)
    pred_vis = pred_img.cpu().numpy().astype(float)
    label_vis[mask.cpu().numpy() == 0] = np.nan
    pred_vis[mask.cpu().numpy() == 0] = np.nan

    # Plot range image
    im0 = axes[0, 0].imshow(range_img, cmap=cmap_jet)
    axes[0, 0].set_title("Range Image", fontsize=14)
    axes[0, 0].set_xlabel("Yaw")
    axes[0, 0].set_ylabel("Laser ID")
    plt.colorbar(im0, ax=axes[0, 0])

    # Plot intensity
    im1 = axes[0, 1].imshow(intensity_img, cmap=cmap_jet)
    axes[0, 1].set_title("Intensity Image", fontsize=14)
    axes[0, 1].set_xlabel("Yaw")
    axes[0, 1].set_ylabel("Laser ID")
    plt.colorbar(im1, ax=axes[0, 1])

    # Plot mask
    axes[0, 2].imshow(mask.cpu().numpy(), cmap="gray")
    axes[0, 2].set_title("Valid Mask", fontsize=14)
    axes[0, 2].set_xlabel("Yaw")
    axes[0, 2].set_ylabel("Laser ID")

    # Plot ground truth label
    im3 = axes[1, 0].imshow(label_vis, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1, 0].set_title("Ground Truth (0=Noise, 1=Clean)", fontsize=14)
    axes[1, 0].set_xlabel("Yaw")
    axes[1, 0].set_ylabel("Laser ID")
    plt.colorbar(im3, ax=axes[1, 0])

    # Plot prediction
    im4 = axes[1, 1].imshow(pred_vis, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1, 1].set_title("Prediction (0=Noise, 1=Clean)", fontsize=14)
    axes[1, 1].set_xlabel("Yaw")
    axes[1, 1].set_ylabel("Laser ID")
    plt.colorbar(im4, ax=axes[1, 1])

    # Plot difference
    diff = np.abs(label_vis - pred_vis)
    im5 = axes[1, 2].imshow(diff, cmap="Reds", vmin=0, vmax=1)
    axes[1, 2].set_title("Prediction Error", fontsize=14)
    axes[1, 2].set_xlabel("Yaw")
    axes[1, 2].set_ylabel("Laser ID")
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def save_denoised_pointcloud(input_path, pred_mask, proj_idx, output_path):
    """
    Save denoised point cloud by removing predicted noise points

    Args:
        input_path: Path to original .bin file
        pred_mask: (H, W) prediction mask (0=noise, 1=clean)
        proj_idx: (H, W) mapping from image pixels to original point indices
        output_path: Path to save denoised .bin file
    """
    # Load original point cloud
    points = np.fromfile(input_path, dtype=np.float32)
    points = points.reshape((-1, 4))

    # Get indices of clean points
    clean_pixels = pred_mask == 1
    clean_point_indices = proj_idx[clean_pixels]
    clean_point_indices = clean_point_indices[clean_point_indices >= 0]
    clean_point_indices = np.unique(clean_point_indices)

    # Filter points
    denoised_points = points[clean_point_indices]

    # Save
    denoised_points.tofile(output_path)

    print(f"Original points: {len(points)}, Denoised points: {len(denoised_points)}")
    print(f"Removed {len(points) - len(denoised_points)} noise points")



def evaluate_model(model, dataloader, device, output_dir, save_visualizations=True):
    """Evaluate model on dataset"""
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    all_metrics = []

    # 混同行列用の累積変数
    total_tp = total_tn = total_fp = total_fn = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inference")

        for i, batch in enumerate(pbar):
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            paths = batch["path"]

            # Forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # Compute metrics for each sample in batch
            batch_size = inputs.shape[0]
            for b in range(batch_size):
                pred = preds[b]
                label = labels[b]
                mask = masks[b]

                # Compute metrics
                mask_flat = mask.flatten().cpu().numpy() > 0
                pred_flat = pred.flatten().cpu().numpy()[mask_flat]
                label_flat = label.flatten().cpu().numpy()[mask_flat]

                if len(pred_flat) > 0:
                    accuracy = (pred_flat == label_flat).mean()

                    # True/False positives/negatives
                    tp = ((pred_flat == 1) & (label_flat == 1)).sum()
                    tn = ((pred_flat == 0) & (label_flat == 0)).sum()
                    fp = ((pred_flat == 1) & (label_flat == 0)).sum()
                    fn = ((pred_flat == 0) & (label_flat == 1)).sum()

                    # 全体の混同行列に累積
                    total_tp += tp
                    total_tn += tn
                    total_fp += fp
                    total_fn += fn

                    precision = tp / (tp + fp + 1e-6)
                    recall = tp / (tp + fn + 1e-6)
                    f1 = 2 * precision * recall / (precision + recall + 1e-6)

                    iou_noise = tn / (tn + fp + fn + 1e-6)
                    iou_clean = tp / (tp + fp + fn + 1e-6)

                    # Count noise/clean
                    pred_noise = (pred_flat == 0).sum()
                    pred_clean = (pred_flat == 1).sum()
                    label_noise = (label_flat == 0).sum()
                    label_clean = (label_flat == 1).sum()

                    metrics = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "iou_noise": iou_noise,
                        "iou_clean": iou_clean,
                        "pred_noise": pred_noise,
                        "pred_clean": pred_clean,
                        "label_noise": label_noise,
                        "label_clean": label_clean,
                    }

                    all_metrics.append(metrics)

                    # Save visualization
                    if save_visualizations and i < 10:
                        vis_path = os.path.join(vis_dir, f"sample_{i:04d}_{b:02d}.png")
                        visualize_prediction(
                            inputs[b], labels[b], preds[b], masks[b], save_path=vis_path
                        )

    # Aggregate metrics
    if len(all_metrics) > 0:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        # === 混同行列の出力 ===
        total_pixels = total_tp + total_tn + total_fp + total_fn
        confusion_matrix = np.array([[total_tn, total_fp],
                                     [total_fn, total_tp]])

        # 正規化版（行方向＝実際のクラスに対する正解率）
        confusion_matrix_norm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-6)

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Total valid pixels: {total_pixels:,}")
        print("\nConfusion Matrix (Noise=0, Clean=1):")
        print("                Predicted")
        print("                Noise    Clean")
        print(f"Actual Noise   {total_tn:8,}  {total_fp:8,}")
        print(f"Actual Clean   {total_fn:8,}  {total_tp:8,}")

        print("\nNormalized Confusion Matrix (row-wise):")
        print("                Predicted")
        print("                Noise    Clean")
        print(f"Actual Noise   {confusion_matrix_norm[0,0]:.4f}    {confusion_matrix_norm[0,1]:.4f}")
        print(f"Actual Clean   {confusion_matrix_norm[1,0]:.4f}    {confusion_matrix_norm[1,1]:.4f}")

        print("\nPer-class Accuracy:")
        print(f"  Noise Accuracy (TN / (TN+FP)): {total_tn / (total_tn + total_fp + 1e-6):.4f}")
        print(f"  Clean Accuracy (TP / (TP+FN)): {total_tp / (total_tp + total_fn + 1e-6):.4f}")

        print("\nOther Metrics:")
        print(f"Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"Precision (Clean): {avg_metrics['precision']:.4f}")
        print(f"Recall    (Clean): {avg_metrics['recall']:.4f}")
        print(f"F1 Score  (Clean): {avg_metrics['f1']:.4f}")
        print(f"IoU Noise:         {avg_metrics['iou_noise']:.4f}")
        print(f"IoU Clean:         {avg_metrics['iou_clean']:.4f}")
        print(f"Avg Predicted Noise: {avg_metrics['pred_noise']:.0f}")
        print(f"Avg Predicted Clean: {avg_metrics['pred_clean']:.0f}")
        print(f"Avg GT Noise:        {avg_metrics['label_noise']:.0f}")
        print(f"Avg GT Clean:        {avg_metrics['label_clean']:.0f}")
        print("=" * 60)

        # メトリクス保存（混同行列も追加）
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(f"TN: {total_tn}\n")
            f.write(f"FP: {total_fp}\n")
            f.write(f"FN: {total_fn}\n")
            f.write(f"TP: {total_tp}\n\n")
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value}\n")

        return avg_metrics

    return None
    """Evaluate model on dataset"""
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    all_metrics = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inference")

        for i, batch in enumerate(pbar):
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            paths = batch["path"]

            # Forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # Compute metrics for each sample in batch
            batch_size = inputs.shape[0]
            for b in range(batch_size):
                pred = preds[b]
                label = labels[b]
                mask = masks[b]

                # Compute metrics
                mask_flat = mask.flatten().cpu().numpy() > 0
                pred_flat = pred.flatten().cpu().numpy()[mask_flat]
                label_flat = label.flatten().cpu().numpy()[mask_flat]

                if len(pred_flat) > 0:
                    accuracy = (pred_flat == label_flat).mean()

                    # Count noise/clean
                    pred_noise = (pred_flat == 0).sum()
                    pred_clean = (pred_flat == 1).sum()
                    label_noise = (label_flat == 0).sum()
                    label_clean = (label_flat == 1).sum()

                    # True/False positives/negatives
                    tp = ((pred_flat == 1) & (label_flat == 1)).sum()
                    tn = ((pred_flat == 0) & (label_flat == 0)).sum()
                    fp = ((pred_flat == 1) & (label_flat == 0)).sum()
                    fn = ((pred_flat == 0) & (label_flat == 1)).sum()

                    precision = tp / (tp + fp + 1e-6)
                    recall = tp / (tp + fn + 1e-6)
                    f1 = 2 * precision * recall / (precision + recall + 1e-6)

                    iou_noise = tn / (tn + fp + fn + 1e-6)
                    iou_clean = tp / (tp + fp + fn + 1e-6)

                    metrics = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "iou_noise": iou_noise,
                        "iou_clean": iou_clean,
                        "pred_noise": pred_noise,
                        "pred_clean": pred_clean,
                        "label_noise": label_noise,
                        "label_clean": label_clean,
                    }

                    all_metrics.append(metrics)

                    # Save visualization
                    if save_visualizations and i < 10:  # Save first 10 batches
                        vis_path = os.path.join(vis_dir, f"sample_{i:04d}_{b:02d}.png")
                        visualize_prediction(
                            inputs[b], labels[b], preds[b], masks[b], save_path=vis_path
                        )

    # Aggregate metrics
    if len(all_metrics) > 0:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        print("\n" + "=" * 50)
        print("Evaluation Results:")
        print("=" * 50)
        print(f"Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"Recall:    {avg_metrics['recall']:.4f}")
        print(f"F1 Score:  {avg_metrics['f1']:.4f}")
        print(f"IoU Noise: {avg_metrics['iou_noise']:.4f}")
        print(f"IoU Clean: {avg_metrics['iou_clean']:.4f}")
        print(f"Avg Predicted Noise: {avg_metrics['pred_noise']:.0f}")
        print(f"Avg Predicted Clean: {avg_metrics['pred_clean']:.0f}")
        print(f"Avg GT Noise: {avg_metrics['label_noise']:.0f}")
        print(f"Avg GT Clean: {avg_metrics['label_clean']:.0f}")
        print("=" * 50)

        # Save metrics
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value}\n")

        return avg_metrics

    return None


def inference_single_file(model, bin_path, device, args, save_output=True):
    """Run inference on a single .bin file"""
    print(f"\nProcessing: {bin_path}")

    # Create projector
    projector = VelodyneRangeProjection(
        args.proj_h, args.proj_w, args.fov_up, args.fov_down
    )

    # Load point cloud
    points = np.fromfile(bin_path, dtype=np.float32)
    points = points.reshape((-1, 4))
    xyz = points[:, :3]
    intensity = points[:, 3]

    # Project to range image
    proj_range, proj_xyz, proj_intensity, proj_mask, proj_idx = projector.project(
        xyz, intensity
    )

    # Create input
    input_img = np.stack(
        [
            proj_xyz[:, :, 0],
            proj_xyz[:, :, 1],
            proj_xyz[:, :, 2],
            proj_intensity,
            proj_range,
        ],
        axis=0,
    ).astype(np.float32)

    # Normalize
    for c in range(5):
        channel = input_img[c]
        valid_pixels = proj_mask > 0
        if valid_pixels.sum() > 0:
            valid_values = channel[valid_pixels]
            mean = valid_values.mean()
            std = valid_values.std() + 1e-6
            channel[valid_pixels] = (valid_values - mean) / std
            channel[~valid_pixels] = 0

    # To tensor
    input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        preds = torch.argmax(outputs, dim=1)[0]

    # Save denoised point cloud
    if save_output:
        output_path = bin_path.replace(".bin", "_denoised.bin")
        save_denoised_pointcloud(bin_path, preds.cpu().numpy(), proj_idx, output_path)

    return preds, proj_mask, input_img


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading U-Net model...")
    model = UNetDenoiser(
        in_channels=5,
        num_classes=2,
        base_channels=args.base_channels,
        bilinear=args.bilinear,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    if args.mode == "eval":
        # Evaluate on test set
        print("Loading test data...")
        _, _, test_loader = create_dataloaders(
            args.data_root,
            args.splits_json,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            proj_H=args.proj_h,
            proj_W=args.proj_w,
            noise_label=args.noise_label,
        )

        evaluate_model(
            model,
            test_loader,
            device,
            args.output_dir,
            save_visualizations=args.save_vis,
        )

    elif args.mode == "single":
        # Process single file
        if not args.input_file:
            raise ValueError("--input_file must be specified for single file mode")

        preds, mask, input_img = inference_single_file(
            model, args.input_file, device, args, save_output=True
        )

        # Visualize
        if args.save_vis:
            vis_path = os.path.join(args.output_dir, "prediction.png")
            os.makedirs(args.output_dir, exist_ok=True)

            # Create dummy label (all clean) for visualization
            dummy_label = torch.ones_like(preds)
            visualize_prediction(
                torch.from_numpy(input_img),
                dummy_label,
                preds,
                torch.from_numpy(mask),
                save_path=vis_path,
            )

    print("\nInference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for U-Net LiDAR denoising")

    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["eval", "single"],
        help="Inference mode: eval on test set or single file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input .bin file (for single mode)",
    )

    # Data
    parser.add_argument("--data_root", type=str, default="./WADS/wads")
    parser.add_argument("--splits_json", type=str, default="./splits.json")
    parser.add_argument("--noise_label", type=int, default=110)

    # Model (U-Net specific)
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

    # Inference
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./inference_outputs")
    parser.add_argument("--save_vis", action="store_true", help="Save visualizations")

    args = parser.parse_args()

    main(args)
