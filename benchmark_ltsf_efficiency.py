"""Profile LTSF model efficiency on ETTm2-shaped batches.

This script does not train for accuracy. It instantiates the same paper-model
architectures used by the LTSF rerun scripts and measures reproducibility-oriented
efficiency quantities for a fixed dataset/horizon setting:

- trainable parameter count
- torch.profiler forward FLOPs/GFLOPs where available
- peak CUDA memory during a timed training loop
- training milliseconds per iteration
- inference milliseconds per batch

The default is ETTm2 across horizons 96/192/336/720 for AdaWarp-MVPF plus the
seven matched neural LTSF comparison models.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Callable, Sequence

import torch
from torch import nn

from adawarp_mvpf_plus import AdaWarpMVPFPlusForecaster
from adawarp_neural_baselines import make_neural_baseline
from benchmark_adawarp_ltsf import DATASET_FILES, dataset_path
from scripts.tacc.run_tslibrary_ltsf_baselines import DATASETS as TSLIB_DATASETS

PAPER_MODELS = (
    "AdaWarp-MVPF",
    "DLinear",
    "PatchTST",
    "TimesNet",
    "iTransformer",
    "TimeMixer",
    "FEDformer",
    "VPNet",
)
TSLIB_MODELS = {"DLinear", "PatchTST", "TimesNet", "iTransformer", "TimeMixer", "FEDformer"}
CUSTOM_MODELS = {"VPNet"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def time_feature_dim(dataset: str) -> int:
    meta = TSLIB_DATASETS.get(dataset)
    if not meta:
        return 4
    return {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}.get(meta["freq"], 4)


def infer_channel_count(data_root: Path, dataset: str) -> int:
    """Infer numeric channel count without importing pandas.

    The profiler only needs tensor shapes, not the full dataset. Standard LTSF
    CSVs have a leading date column followed by numeric variables, so counting
    parseable floating-point entries in the first data row is sufficient and
    avoids adding another dependency to this lightweight audit.
    """

    csv_path = dataset_path(data_root, dataset)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        _ = next(reader, None)
        for row in reader:
            if not row:
                continue
            count = 0
            for value in row:
                try:
                    float(value)
                except ValueError:
                    continue
                count += 1
            if count > 0:
                return count
    raise ValueError(f"Could not infer numeric channels from {csv_path}")


def build_tslib_args(args: argparse.Namespace, model_name: str, horizon: int, channels: int) -> SimpleNamespace:
    meta = TSLIB_DATASETS[args.dataset]
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id=f"eff_{args.dataset}_{horizon}",
        model=model_name,
        data=meta["data"],
        root_path=meta["root"] + "/",
        data_path=meta["path"],
        features="M",
        target="OT",
        freq=meta["freq"],
        checkpoints="./checkpoints/",
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=horizon,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=channels,
        dec_in=channels,
        c_out=channels,
        d_model=args.tslib_d_model,
        n_heads=args.tslib_n_heads,
        e_layers=args.tslib_e_layers,
        d_layers=args.tslib_d_layers,
        d_ff=args.tslib_d_ff,
        moving_avg=25,
        factor=3,
        distil=True,
        dropout=args.dropout,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=1,
        down_sampling_window=2,
        down_sampling_method="avg",
        seg_len=96,
        num_workers=0,
        itr=1,
        train_epochs=10,
        batch_size=args.batch_size,
        patience=3,
        learning_rate=1e-4,
        des="efficiency",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=False,
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0",
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=args.seed,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        patch_len=16,
        num_class=0,
    )


def build_model(args: argparse.Namespace, model_name: str, horizon: int, channels: int, device: torch.device) -> nn.Module:
    if model_name == "AdaWarp-MVPF":
        return AdaWarpMVPFPlusForecaster(
            args.seq_len,
            horizon,
            patch_lens=args.mvpf_patch_lens,
            width=args.mvpf_d_model,
            depth=args.mvpf_depth,
            dropout=args.dropout,
            num_prototypes=args.mvpf_num_prototypes,
            max_shift=args.mvpf_max_shift,
            reconstruction_weight=args.mvpf_reconstruction_weight,
            use_prototype_memory=False,
            use_frequency_gate=False,
            use_adaptive_shifts=True,
            use_adaptive_radius=True,
            use_component_gate=True,
            use_trend_decomposition=True,
            use_linear_field=True,
        ).to(device)
    if model_name in CUSTOM_MODELS:
        return make_neural_baseline(
            model_name,
            args.seq_len,
            horizon,
            width=args.custom_d_model,
            depth=args.custom_depth,
            blocks=args.custom_blocks,
            dropout=args.dropout,
            vpnet_patch_len=args.vpnet_patch_len,
        ).to(device)
    if model_name in TSLIB_MODELS:
        ts_root = Path(args.tslibrary_root).resolve()
        if str(ts_root) not in sys.path:
            sys.path.insert(0, str(ts_root))
        module = importlib.import_module(f"models.{model_name}")
        config = build_tslib_args(args, model_name, horizon, channels)
        return module.Model(config).float().to(device)
    raise ValueError(f"Unsupported efficiency model: {model_name}")


def make_batches(
    args: argparse.Namespace,
    model_name: str,
    horizon: int,
    channels: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, Callable[[nn.Module, tuple[torch.Tensor, ...]], torch.Tensor]]:
    batch = args.batch_size
    x_enc = torch.randn(batch, args.seq_len, channels, device=device)
    target = torch.randn(batch, horizon, channels, device=device)
    if model_name in TSLIB_MODELS:
        mark_dim = time_feature_dim(args.dataset)
        x_mark = torch.zeros(batch, args.seq_len, mark_dim, device=device)
        x_dec = torch.zeros(batch, args.label_len + horizon, channels, device=device)
        x_mark_dec = torch.zeros(batch, args.label_len + horizon, mark_dim, device=device)

        def forward_fn(model: nn.Module, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
            output = model(*tensors)
            if isinstance(output, tuple):
                output = output[0]
            return output[:, -horizon:, :]

        return (x_enc, x_mark, x_dec, x_mark_dec), target, forward_fn

    def forward_fn(model: nn.Module, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return model(tensors[0], None, None, None)

    return (x_enc,), target, forward_fn


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_forward_flops(model: nn.Module, tensors: tuple[torch.Tensor, ...], forward_fn: Callable, device: torch.device) -> float:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        model.eval()
        with torch.no_grad():
            with torch.profiler.profile(activities=activities, with_flops=True, profile_memory=True) as prof:
                _ = forward_fn(model, tensors)
        total_flops = sum(int(getattr(event, "flops", 0) or 0) for event in prof.key_averages())
        return float(total_flops) / 1.0e9
    except Exception:
        return float("nan")


def benchmark_inference(
    model: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    forward_fn: Callable,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_fn(model, tensors)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            _ = forward_fn(model, tensors)
        synchronize(device)
        elapsed = time.perf_counter() - started
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else float("nan")
    return 1000.0 * elapsed / max(1, iterations), peak_mb


def benchmark_training(
    model: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    forward_fn: Callable,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_fn(model, tensors)
        loss = criterion(prediction, target)
        if hasattr(model, "auxiliary_loss") and hasattr(model, "reconstruction_weight"):
            loss = loss + float(model.reconstruction_weight) * model.auxiliary_loss(tensors[0])
        loss.backward()
        optimizer.step()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_fn(model, tensors)
        loss = criterion(prediction, target)
        if hasattr(model, "auxiliary_loss") and hasattr(model, "reconstruction_weight"):
            loss = loss + float(model.reconstruction_weight) * model.auxiliary_loss(tensors[0])
        loss.backward()
        optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else float("nan")
    return 1000.0 * elapsed / max(1, iterations), peak_mb


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_root = ensure_dir(Path(args.output_root))
    rows: list[dict[str, object]] = []
    channels = infer_channel_count(Path(args.data_root), args.dataset)
    for horizon in args.horizons:
        for model_name in args.models:
            torch.manual_seed(args.seed)
            try:
                model = build_model(args, model_name, horizon, channels, device)
                tensors, target, forward_fn = make_batches(args, model_name, horizon, channels, device)
                params = count_parameters(model)
                gflops = profile_forward_flops(model, tensors, forward_fn, device)
                infer_ms, infer_peak = benchmark_inference(
                    model,
                    tensors,
                    forward_fn,
                    device,
                    args.warmup_iters,
                    args.inference_iters,
                )
                train_ms, train_peak = benchmark_training(
                    model,
                    tensors,
                    target,
                    forward_fn,
                    device,
                    args.warmup_iters,
                    args.training_iters,
                )
                row = {
                    "dataset": args.dataset,
                    "horizon": horizon,
                    "model": model_name,
                    "seq_len": args.seq_len,
                    "batch_size": args.batch_size,
                    "channels": channels,
                    "params": params,
                    "params_millions": params / 1.0e6,
                    "forward_gflops_per_batch": gflops,
                    "peak_gpu_memory_mb": train_peak,
                    "inference_peak_gpu_memory_mb": infer_peak,
                    "training_ms_per_iter": train_ms,
                    "inference_ms_per_batch": infer_ms,
                    "device": str(device),
                    "status": "ok",
                    "error": "",
                }
                print(
                    f"eff {model_name:<13} {args.dataset:<5} h={horizon:<3} "
                    f"params={params/1e6:.3f}M gflops={gflops:.3f} "
                    f"train_ms={train_ms:.2f} infer_ms={infer_ms:.2f} peak_mb={train_peak:.1f}",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    "dataset": args.dataset,
                    "horizon": horizon,
                    "model": model_name,
                    "seq_len": args.seq_len,
                    "batch_size": args.batch_size,
                    "channels": channels,
                    "params": "",
                    "params_millions": "",
                    "forward_gflops_per_batch": "",
                    "peak_gpu_memory_mb": "",
                    "inference_peak_gpu_memory_mb": "",
                    "training_ms_per_iter": "",
                    "inference_ms_per_batch": "",
                    "device": str(device),
                    "status": "failed",
                    "error": repr(exc),
                }
                print(f"eff FAILED {model_name} {args.dataset} h={horizon}: {exc!r}", flush=True)
            rows.append(row)
    fields = [
        "dataset",
        "horizon",
        "model",
        "seq_len",
        "batch_size",
        "channels",
        "params",
        "params_millions",
        "forward_gflops_per_batch",
        "peak_gpu_memory_mb",
        "inference_peak_gpu_memory_mb",
        "training_ms_per_iter",
        "inference_ms_per_batch",
        "device",
        "status",
        "error",
    ]
    write_csv(output_root / "metrics" / "ltsf_efficiency_ettm2.csv", rows, fields)
    with (output_root / "audit_efficiency.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="ETTm2")
    parser.add_argument("--models", nargs="+", choices=PAPER_MODELS, default=list(PAPER_MODELS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--training-iters", type=int, default=10)
    parser.add_argument("--inference-iters", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--tslib-d-model", type=int, default=128)
    parser.add_argument("--tslib-d-ff", type=int, default=256)
    parser.add_argument("--tslib-n-heads", type=int, default=8)
    parser.add_argument("--tslib-e-layers", type=int, default=2)
    parser.add_argument("--tslib-d-layers", type=int, default=1)
    parser.add_argument("--custom-d-model", type=int, default=256)
    parser.add_argument("--custom-depth", type=int, default=2)
    parser.add_argument("--custom-blocks", type=int, default=4)
    parser.add_argument("--vpnet-patch-len", type=int, default=16)
    parser.add_argument("--mvpf-d-model", type=int, default=128)
    parser.add_argument("--mvpf-depth", type=int, default=2)
    parser.add_argument("--mvpf-patch-lens", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--mvpf-num-prototypes", type=int, default=8)
    parser.add_argument("--mvpf-max-shift", type=int, default=2)
    parser.add_argument("--mvpf-reconstruction-weight", type=float, default=0.03)
    parser.add_argument("--data-root", default="TSLibrary/dataset")
    parser.add_argument("--tslibrary-root", default="TSLibrary")
    parser.add_argument("--output-root", default="results/ltsf_efficiency_ettm2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())