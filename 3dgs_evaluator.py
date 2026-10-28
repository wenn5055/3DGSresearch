"""Evaluate a trained 3DGS PLY against a held-out view set and report PSNR, SSIM and LPIPS.

Usage:
    python 3dgs_evaluator.py trained.ply data/lego
    python 3dgs_evaluator.py trained.ply data/lego --arch gpu --metrics psnr ssim

For every camera in the manifest this renders the PLY with the same tiled Warp pipeline training
uses, saves the render as a PNG, and writes one CSV row per view.

Three things are decided rather than duplicated into separate scripts:

Device. `--arch auto`, the default, uses CUDA when Warp reports it and CPU otherwise, so the same
command works on a laptop and on a GPU node. `--arch cpu` or `--arch gpu` forces the choice, and
forcing gpu without CUDA is an error rather than a silent fallback -- a run that quietly took
forty times longer than expected is worse than one that refused to start.

Appearance. A model trained with spherical harmonics writes a `<ply-stem>.sh.npz` sidecar. If one
is next to the PLY, evaluation uses its saved active degree (0-3). Otherwise the PLY's degree-0
colour is used. `--sh-degree 1/2/3` requires a sidecar trained to at least that degree;
`--sh-degree 0` ignores it, and `--sh-npz` points at one kept elsewhere.

Metrics. `--metrics` selects any of psnr, ssim and lpips; all three run by default. LPIPS needs
PyTorch, which is slow to import and not always installed, so it is imported only when asked for.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import warp as wp

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# Shared modules sit beside this file in the assignment repository.
_shared = _here / "shared"
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

from image_metrics import ssim_rgb
from gaussian_set import GaussianSet
from sh_appearance import load_sh_sidecar

# The base trainer is `3dgs_trainer_v1` in the course repo and `3dgs_trainer` in the assignment
# repo, where there is only one. Probing for both keeps this file identical in the two trees, the
# same way the `shared/` lookup above does.
for _candidate_module in ("3dgs_trainer_v1", "3dgs_trainer"):
    try:
        _base_trainer = importlib.import_module(_candidate_module)
        break
    except ModuleNotFoundError:
        continue
else:
    raise ModuleNotFoundError(
        "No trainer module found. This script expects 3dgs_trainer_v1.py or 3dgs_trainer.py "
        "beside it."
    )

DEFAULT_RESOLUTION = _base_trainer.DEFAULT_RESOLUTION
WarpImageTrainer = _base_trainer.WarpImageTrainer
load_views = _base_trainer.load_views

METRICS = ("psnr", "ssim", "lpips")


def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)


class LPIPSMetric:
    """Loads PyTorch and the LPIPS weights on first use, not at import.

    Kept lazy because most runs do not ask for LPIPS, `--help` never needs it, and importing
    torch costs seconds even when the metric is never called.
    """

    def __init__(self, net: str):
        try:
            import lpips
            import torch
        except ImportError as error:
            raise RuntimeError(
                "LPIPS needs PyTorch and lpips. Install them, or pass --metrics psnr ssim to "
                "skip it."
            ) from error
        self.torch = torch
        self.metric = lpips.LPIPS(net=net).eval()

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        def tensor(image):
            image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
            # LPIPS expects NCHW in [-1, 1].
            return self.torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

        with self.torch.inference_mode():
            return float(self.metric(tensor(prediction), tensor(target)).item())


def resolve_device(arch: str) -> str:
    if arch == "cpu":
        return "cpu"
    if arch == "gpu":
        if not wp.is_cuda_available():
            raise RuntimeError(
                "--arch gpu needs a CUDA-enabled Warp installation. Use --arch cpu, or --arch "
                "auto to pick whichever is available."
            )
        return "cuda:0"
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def build_trainer(args, scene, targets, cameras, focal, distance, device):
    """Validate the sidecar, then build the shared renderer. Returns (trainer, degree)."""
    capacity = len(scene.means)
    background = (
        np.ones(3, np.float32) if args.background == "white" else np.zeros(3, np.float32)
    )
    sh_path = args.sh_npz if args.sh_npz is not None else args.ply.with_suffix(".sh.npz")
    if args.sh_degree != 0 and not sh_path.exists() and (args.sh_degree is not None or args.sh_npz is not None):
        raise FileNotFoundError(
            f"Spherical-harmonic rendering requested, but no sidecar at {sh_path}."
        )
    degree, max_degree, values = 0, 0, None
    if args.sh_degree != 0 and sh_path.exists():
        values, max_degree, saved_degree = load_sh_sidecar(sh_path, capacity)
        degree = saved_degree if args.sh_degree is None else args.sh_degree
        if degree > saved_degree:
            raise ValueError(f"Requested SH degree {degree}, but {sh_path} only trained through degree {saved_degree}.")
    trainer = WarpImageTrainer(
        targets, cameras, focal, distance * 0.35, capacity, capacity, device, 0,
        init_scene=scene, background=background, tile_pair_capacity=args.tile_pair_capacity,
        sh_degree=max_degree if degree else 0,
    )
    if degree:
        trainer.sh.coefficients.assign(values.reshape(-1, 3))
        trainer.sh.degree = degree
    return trainer, degree


def evaluate(args: argparse.Namespace) -> None:
    wp.init()
    device = resolve_device(args.arch)
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else args.resolution

    data = args.data.expanduser().resolve()
    manifest_path = data / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist. A NeRF-synthetic download that only fetched "
            "transforms_train.json has no held-out split; fetch the full dataset, or pass "
            "--manifest transforms_train.json to evaluate against the training views instead."
        )

    scene = GaussianSet.from_ply(args.ply)
    background = (
        np.ones(3, np.float32) if args.background == "white" else np.zeros(3, np.float32)
    )
    targets, cameras, focal, distance, frame_paths = load_views(
        data, width, height, background, manifest_name=args.manifest
    )
    trainer, using_sh = build_trainer(args, scene, targets, cameras, focal, distance, device)

    wanted = list(dict.fromkeys(args.metrics))
    lpips_metric = LPIPSMetric(args.lpips_net) if "lpips" in wanted else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Evaluating {len(scene.means):,} splats from {args.ply} against {len(frame_paths)} "
        f"view(s) in {manifest_path} at {width}x{height} on Warp {device}; "
        f"appearance={f'degree-{using_sh} SH' if using_sh else 'degree-0 RGB'}; "
        f"metrics={', '.join(wanted)}.",
        flush=True,
    )

    fields = ["view", "file", "mse"] + [
        {"psnr": "psnr_db", "ssim": "ssim", "lpips": "lpips"}[m] for m in wanted
    ]
    rows = []
    for view_id, frame_path in enumerate(frame_paths):
        try:
            image, mse = trainer.render_single_view_image(view_id)
        except RuntimeError as error:
            raise RuntimeError(
                f"{error} (pass a larger --tile-pair-capacity to this script)"
            ) from error
        image = np.clip(image, 0.0, 1.0)
        row = {"view": view_id, "file": frame_path, "mse": mse}
        if "psnr" in wanted:
            row["psnr_db"] = psnr_from_mse(mse)
        if "ssim" in wanted:
            row["ssim"] = ssim_rgb(image, targets[view_id])
        if "lpips" in wanted:
            row["lpips"] = lpips_metric(image, targets[view_id])
        Image.fromarray(np.uint8(image * 255.0)).save(
            args.output_dir / f"{Path(frame_path).stem}.png"
        )
        rows.append(row)
        reported = " ".join(
            f"{label}={row[key]:{fmt}}"
            for label, key, fmt in (
                ("psnr", "psnr_db", "6.2f"), ("ssim", "ssim", ".5f"), ("lpips", "lpips", ".5f"),
            )
            if key in row
        )
        print(f"  [{view_id:4d}] {frame_path:30s} mse={mse:.6f} {reported}", flush=True)

    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    means = {key: float(np.mean([row[key] for row in rows])) for key in fields[2:]}
    summary = ", ".join(
        f"mean {key}={value:.5f}" if key != "psnr_db" else f"mean psnr={value:.2f} dB"
        for key, value in means.items()
    )
    print(
        f"\n{len(rows)} view(s): {summary}. "
        f"Renders in {args.output_dir}, per-view results in {args.csv}.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ply", type=Path, help="Trained splats to evaluate.")
    parser.add_argument("data", type=Path, help="Dataset directory, e.g. data/lego.")
    parser.add_argument(
        "--manifest", default="transforms_test.json",
        help="Manifest inside the dataset directory. Default: the held-out test split.",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=METRICS, default=list(METRICS),
        metavar="METRIC", help=f"Any of: {', '.join(METRICS)}. Default: all three.",
    )
    parser.add_argument(
        "--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex",
        help="Backbone for LPIPS. Ignored unless lpips is among --metrics.",
    )
    parser.add_argument(
        "--arch", choices=("auto", "cpu", "gpu"), default="auto",
        help="auto uses CUDA when Warp reports it, and CPU otherwise.",
    )
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--width", type=int, default=None, help="Overrides --resolution.")
    parser.add_argument("--height", type=int, default=None, help="Overrides --resolution.")
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument(
        "--sh-degree", type=int, choices=(0, 1, 2, 3), default=None,
        help="0 uses base RGB; 1-3 require a sidecar trained to at least that degree. "
             "Default: use the sidecar's saved active degree when one exists.",
    )
    parser.add_argument(
        "--sh-npz", type=Path, default=None,
        help="Sidecar path. Default: the PLY's name with a .sh.npz suffix.",
    )
    parser.add_argument("--tile-pair-capacity", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/eval/metrics.csv"))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
