# inference時間はPCのスペックにもよるし、使用状況にもよる

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from data_loader import WADSDataset
from model import UNetDenoiser


class InferenceTimer:
    """Measure inference time for model"""

    def __init__(self, model, device, warmup_runs=10):
        self.model = model
        self.device = device
        self.warmup_runs = warmup_runs
        self.times = {
            "total": [],
            "preprocessing": [],
            "model_forward": [],
            "postprocessing": [],
        }

    def warmup(self, sample_input):
        """Warmup GPU"""
        print(f"Warming up GPU with {self.warmup_runs} runs...")
        self.model.eval()

        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = self.model(sample_input)

        # Synchronize to ensure warmup is complete
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        print("Warmup complete!")

    def measure_single_inference(self, input_tensor):
        """
        Measure inference time for a single sample

        Args:
            input_tensor: (1, C, H, W) input tensor (batch_size=1)

        Returns:
            times: Dictionary with timing information
        """
        self.model.eval()

        # Start total time
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_total = time.perf_counter()

        # Preprocessing time (already done, so minimal)
        start_prep = time.perf_counter()
        input_tensor = input_tensor.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        prep_time = time.perf_counter() - start_prep

        # Model forward time
        start_forward = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward

        # Postprocessing time
        start_post = time.perf_counter()
        preds = torch.argmax(outputs, dim=1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        post_time = time.perf_counter() - start_post

        # Total time
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start_total

        return {
            "total": total_time,
            "preprocessing": prep_time,
            "model_forward": forward_time,
            "postprocessing": post_time,
        }

    def benchmark_dataset(self, dataset, num_samples=None):
        """
        Benchmark inference time on dataset

        Args:
            dataset: Dataset to benchmark
            num_samples: Number of samples to test (None = all)
        """
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))

        print(f"\nBenchmarking on {num_samples} samples...")

        # Warmup with first sample
        sample = dataset[0]
        input_tensor = sample["input"].unsqueeze(0).to(self.device)
        self.warmup(input_tensor)

        # Reset timing records
        self.times = {k: [] for k in self.times.keys()}

        # Measure inference time for each sample
        for idx in tqdm(range(num_samples), desc="Measuring inference time"):
            sample = dataset[idx]
            input_tensor = sample["input"].unsqueeze(0)

            times = self.measure_single_inference(input_tensor)

            for key, value in times.items():
                self.times[key].append(value)

        # Compute statistics
        stats = self.compute_statistics()

        return stats

    def compute_statistics(self):
        """Compute timing statistics"""
        stats = {}

        for key, times in self.times.items():
            if len(times) > 0:
                times_ms = np.array(times) * 1000  # Convert to milliseconds

                stats[key] = {
                    "mean_ms": float(np.mean(times_ms)),
                    "std_ms": float(np.std(times_ms)),
                    "min_ms": float(np.min(times_ms)),
                    "max_ms": float(np.max(times_ms)),
                    "median_ms": float(np.median(times_ms)),
                    "p95_ms": float(np.percentile(times_ms, 95)),
                    "p99_ms": float(np.percentile(times_ms, 99)),
                }

        # Compute FPS
        if len(self.times["total"]) > 0:
            mean_time = np.mean(self.times["total"])
            stats["fps"] = float(1.0 / mean_time) if mean_time > 0 else 0.0

        return stats

    def print_statistics(self, stats):
        """Print timing statistics"""
        print("\n" + "=" * 70)
        print("INFERENCE TIME MEASUREMENT RESULTS")
        print("=" * 70)

        print(f"\nDevice: {self.device}")
        print(f"Number of samples: {len(self.times['total'])}")

        if "fps" in stats:
            print(f"\n--- Throughput ---")
            print(f"FPS (Frames Per Second): {stats['fps']:.2f}")

        for stage in ["total", "preprocessing", "model_forward", "postprocessing"]:
            if stage in stats:
                print(f"\n--- {stage.replace('_', ' ').title()} ---")
                s = stats[stage]
                print(f"Mean:   {s['mean_ms']:.3f} ms")
                print(f"Std:    {s['std_ms']:.3f} ms")
                print(f"Min:    {s['min_ms']:.3f} ms")
                print(f"Max:    {s['max_ms']:.3f} ms")
                print(f"Median: {s['median_ms']:.3f} ms")
                print(f"95th percentile: {s['p95_ms']:.3f} ms")
                print(f"99th percentile: {s['p99_ms']:.3f} ms")

        # Breakdown percentage
        if "total" in stats and "model_forward" in stats:
            print(f"\n--- Time Breakdown ---")
            total_mean = stats["total"]["mean_ms"]
            for stage in ["preprocessing", "model_forward", "postprocessing"]:
                if stage in stats:
                    stage_mean = stats[stage]["mean_ms"]
                    percentage = (
                        (stage_mean / total_mean * 100) if total_mean > 0 else 0
                    )
                    print(
                        f"{stage.replace('_', ' ').title()}: {stage_mean:.3f} ms ({percentage:.1f}%)"
                    )

        print("=" * 70)


def measure_model_params(model):
    """Measure model parameters and memory"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024**2)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_size_mb,
    }


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("\nLoading U-Net model...")
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

    # Model info
    model_info = measure_model_params(model)
    print(f"\n--- Model Information ---")
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")

    # Load dataset
    print("\nLoading test dataset...")
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

    # Create timer
    timer = InferenceTimer(model, device, warmup_runs=args.warmup_runs)

    # Benchmark
    stats = timer.benchmark_dataset(test_dataset, num_samples=args.num_samples)

    # Print results
    timer.print_statistics(stats)

    # Save results
    results = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "model_info": model_info,
        "model_config": {
            "proj_h": args.proj_h,
            "proj_w": args.proj_w,
            "base_channels": args.base_channels,
            "bilinear": args.bilinear,
        },
        "timing_statistics": stats,
        "num_samples_tested": len(timer.times["total"]),
    }

    output_file = os.path.join(args.output_dir, "inference_time_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Create detailed timing log
    if args.save_detailed_log:
        detailed_log = {
            "times_ms": {
                key: [t * 1000 for t in values] for key, values in timer.times.items()
            }
        }

        log_file = os.path.join(args.output_dir, "detailed_timing_log.json")
        with open(log_file, "w") as f:
            json.dump(detailed_log, f, indent=2)

        print(f"Detailed log saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure inference time for U-Net")

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
    parser.add_argument("--noise_label", type=int, default=250)

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

    # Benchmark settings
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to test (None = all test set)",
    )
    parser.add_argument(
        "--warmup_runs", type=int, default=10, help="Number of warmup runs"
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="./timing_results")
    parser.add_argument(
        "--save_detailed_log",
        action="store_true",
        help="Save detailed timing log for all samples",
    )

    args = parser.parse_args()

    main(args)
