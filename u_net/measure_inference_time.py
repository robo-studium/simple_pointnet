# inference時間はPCのスペックにもよるし、使用状況にもよる

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from data_loader import WADSDataset
from student_model import StudentUNet
from model_unet_trans import UNetDenoiser


# ============================================================
# Inference Timer
# ============================================================
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

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        print("Warmup complete!")

    def measure_single_inference(self, input_tensor):
        """Measure inference time for a single sample"""
        self.model.eval()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_total = time.perf_counter()

        # preprocessing
        start_prep = time.perf_counter()
        input_tensor = input_tensor.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        prep_time = time.perf_counter() - start_prep

        # forward
        start_forward = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward

        # postprocessing
        start_post = time.perf_counter()
        _ = torch.argmax(outputs, dim=1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        post_time = time.perf_counter() - start_post

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
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))

        print(f"\nBenchmarking on {num_samples} samples...")

        sample = dataset[0]
        input_tensor = sample["input"].unsqueeze(0).to(self.device)
        self.warmup(input_tensor)

        self.times = {k: [] for k in self.times.keys()}

        for idx in tqdm(range(num_samples), desc="Measuring inference time"):
            sample = dataset[idx]
            input_tensor = sample["input"].unsqueeze(0)

            times = self.measure_single_inference(input_tensor)
            for key, value in times.items():
                self.times[key].append(value)

        return self.compute_statistics()

    def compute_statistics(self):
        stats = {}

        for key, times in self.times.items():
            if len(times) == 0:
                continue
            times_ms = np.array(times) * 1000
            stats[key] = {
                "mean_ms": float(np.mean(times_ms)),
                "std_ms": float(np.std(times_ms)),
                "min_ms": float(np.min(times_ms)),
                "max_ms": float(np.max(times_ms)),
                "median_ms": float(np.median(times_ms)),
                "p95_ms": float(np.percentile(times_ms, 95)),
                "p99_ms": float(np.percentile(times_ms, 99)),
            }

        if len(self.times["total"]) > 0:
            mean_time = np.mean(self.times["total"])
            stats["fps"] = float(1.0 / mean_time)

        return stats

    def print_statistics(self, stats):
        print("\n" + "=" * 70)
        print("INFERENCE TIME MEASUREMENT RESULTS")
        print("=" * 70)

        print(f"Device: {self.device}")
        print(f"Samples: {len(self.times['total'])}")

        print(f"\nFPS: {stats['fps']:.2f}")

        for k in ["total", "model_forward"]:
            s = stats[k]
            print(f"\n[{k}] Mean: {s['mean_ms']:.3f} ms  (p95: {s['p95_ms']:.3f} ms)")


# ============================================================
# Utils
# ============================================================
def measure_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)

    return {
        "total_params": total_params,
        "model_size_mb": size_mb,
    }


# ============================================================
# Main
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # Model selection
    # -------------------------
    if args.model_type == "student":
        print("Loading StudentUNet...")
        model = StudentUNet(
            in_channels=5,
            num_classes=2,
            base_channels=args.base_channels,
        ).to(device)

    elif args.model_type == "unet":
        print("Loading UNetDenoiser...")
        model = UNetDenoiser(
            in_channels=5,
            num_classes=2,
            base_channels=args.base_channels,
            bilinear=args.bilinear,
        ).to(device)

    else:
        raise ValueError("Invalid model_type")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded.")

    print(measure_model_params(model))

    # -------------------------
    # Dataset
    # -------------------------
    with open(args.splits_json) as f:
        splits = json.load(f)

    dataset = WADSDataset(
        args.data_root,
        splits["test"],
        proj_H=args.proj_h,
        proj_W=args.proj_w,
        fov_up=args.fov_up,
        fov_down=args.fov_down,
        noise_label=args.noise_label,
    )

    timer = InferenceTimer(model, device, args.warmup_runs)
    stats = timer.benchmark_dataset(dataset, args.num_samples)
    timer.print_statistics(stats)


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="student")
    # parser.add_argument("--model_type", choices=["student", "unet"], required=True)

    parser.add_argument("--checkpoint", type=str, default="./outputs_student/best_student_model.pth")
    # parser.add_argument("--checkpoint", type=str, required=True)


    parser.add_argument("--data_root", type=str, default="./WADS/wads")
    parser.add_argument("--splits_json", type=str, default="./splits.json")
    parser.add_argument("--noise_label", type=int, default=110)

    parser.add_argument("--proj_h", type=int, default=64)
    parser.add_argument("--proj_w", type=int, default=1024)
    parser.add_argument("--fov_up", type=float, default=2.0)
    parser.add_argument("--fov_down", type=float, default=-24.9)

    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--bilinear", action="store_true")

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--warmup_runs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./timing_results")

    args = parser.parse_args()
    main(args)
