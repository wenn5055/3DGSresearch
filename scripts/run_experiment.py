"""Record, train, and evaluate Gaussian or surfel experiments in isolated output folders."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata as packages
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile

import yaml

from update_output_index import refresh_output_index

ROOT = Path(__file__).resolve().parents[1]
SINGAPORE_TIMEZONE = timezone(timedelta(hours=8), "SGT")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
sys.path.insert(0, str(ROOT / "shared"))
from training_config import TrainingConfig


class StageFailed(RuntimeError):
    def __init__(self, name: str, returncode: int):
        super().__init__(f"{name} failed with exit code {returncode}; inspect {name}.log")
        self.returncode = returncode


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def capture(command: list[str], root: Path) -> dict:
    """Optional provenance commands must not prevent CPU runs or offline use."""
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=20)
        return {"command": command, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "exit_code": None, "error": str(error)}


def snapshot_provenance(root: Path, run_dir: Path) -> dict:
    git = {
        "commit": capture(["git", "rev-parse", "HEAD"], root),
        "status": capture(["git", "status", "--short"], root),
        "remotes": capture(["git", "remote", "-v"], root),
    }
    patch = capture(["git", "diff", "HEAD", "--binary", "--no-ext-diff"], root)
    (run_dir / "working-tree.patch").write_text(patch.get("stdout", ""), encoding="utf-8")
    git["patch_exit_code"] = patch["exit_code"]
    # Include new/untracked implementation files too; a Git diff alone misses them.
    sources = set(root.glob("*.py"))
    for directory in ("scripts", "shared", "src", "config"):
        for suffix in ("*.py", "*.yaml", "*.yml"):
            sources.update((root / directory).rglob(suffix))
    for name in ("requirements.txt", ".gitignore"):
        if (root / name).is_file():
            sources.add(root / name)
    source_records = []
    with zipfile.ZipFile(run_dir / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Source file resolves outside the repository: {path}")
            relative = path.relative_to(root).as_posix()
            archive.write(path, relative)
            source_records.append({**fingerprint(path), "path": relative})
    (run_dir / "source-files.json").write_text(
        json.dumps(source_records, indent=2), encoding="utf-8")
    versions = {dist.metadata["Name"]: dist.version for dist in packages.distributions()
                if dist.metadata["Name"]}
    (run_dir / "packages.json").write_text(
        json.dumps(dict(sorted(versions.items())), indent=2), encoding="utf-8")
    return {"git": git, "python": sys.version, "python_executable": sys.executable,
            "runtime_environment": {key: os.environ.get(key) for key in
                                    ("CUDA_VISIBLE_DEVICES", "WARP_CACHE_PATH", "OMP_NUM_THREADS")},
            "platform": platform.platform(), "gpu_snapshot": capture(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"], root)}


def inspect_inputs(config: dict, eval_manifest: str, run_dir: Path) -> dict:
    """Validate the selected scene, rather than calling the Lego-only setup check."""
    data = Path(config["paths"]["data"])
    names = ["transforms_train.json", config["reporting"]["eval_manifest"], eval_manifest]
    records, files = {}, set()
    for name in dict.fromkeys(item for item in names if item is not None):
        path = (data / name).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError(f"Manifest must be inside the dataset: {name}")
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        if not 0 < float(manifest["camera_angle_x"]) < math.pi:
            raise ValueError(f"Invalid camera_angle_x in {path}")
        frames = manifest["frames"]
        if not frames:
            raise ValueError(f"No frames in {path}")
        files.add(path)
        for frame in frames:
            pose = frame["transform_matrix"]
            if len(pose) != 4 or any(len(row) != 4 for row in pose):
                raise ValueError(f"Expected a 4x4 camera transform in {path}")
            base = data / frame["file_path"]
            candidates = ([base] if base.suffix else []) + [
                base.with_suffix(suffix) for suffix in (".png", ".jpg", ".jpeg")]
            image = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if image is None:
                raise FileNotFoundError(f"Missing image for {frame['file_path']} in {path}")
            files.add(image)
        records[name] = {**fingerprint(path), "frames": len(frames),
                         "frame_paths": [frame["file_path"] for frame in frames]}
    init_ply = config["paths"]["init_ply"]
    init_materials = None
    if init_ply is not None:
        files.add(Path(init_ply))
        if config.get('surfel') and config['model'].get('geometry') == 'surfel':
            sidecar = Path(init_ply).with_suffix('.surfel.npz')
            if sidecar.is_file():
                files.add(sidecar)
                init_materials = fingerprint(sidecar)
    fingerprints = [fingerprint(path) for path in sorted(files)]
    (run_dir / "input-files.json").write_text(
        json.dumps(fingerprints, indent=2), encoding="utf-8")
    return {"directory": str(data), "manifests": records, "initial_materials": init_materials,
            "initialization": "random" if init_ply is None else fingerprint(Path(init_ply))}


def write_record(run_dir: Path, record: dict) -> None:
    pending = run_dir / "run.json.tmp"
    pending.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(run_dir / "run.json")
    config = record["config"]
    paper_loss = config.get('surfel', {}).get('photometric_loss') == 'l1_dssim'
    image_loss = 'l1_dssim (from step 1)' if paper_loss else config['loss']['mode']
    lines = [f"# Experiment: {record['run_id']}", "",
             f"- Status: {record['status']}", f"- Started (UTC): {record['started_utc']}",
             f"- Finished (UTC): {record.get('finished_utc', 'pending')}",
             f"- Purpose: {record['note'] or 'Add your research question here.'}",
             f"- Dataset: {config['paths']['data']}",
             f"- Pipeline: {record.get('pipeline', 'gaussian')}",
             f"- Final evaluation manifest: {record['evaluation']['manifest']}",
             f"- Seed: {config['runtime']['seed']}",
             f"- Configured iterations: {config['training']['iterations']}",
             f"- Loss: {image_loss}; D-SSIM weight: {config['loss']['dssim_weight']}; "
             f"transition PSNR: {config['adaptive']['dssim_psnr_threshold']} dB",
             f"- Configured maximum SH degree: {config['model']['sh_degree']}",
             f"- Early stopping enabled: {config['convergence']['enabled']}",
             f"- Requested metrics: {', '.join(record['evaluation']['metrics'])}",
             "- Settings: [config.yaml](config.yaml); provenance and exact commands: [run.json](run.json)",
             "- Code: source.zip and source-files.json; inputs: input-files.json", ""]
    for stage in record["stages"]:
        lines.append(f"- {stage['name']}: {stage['status']}; exit={stage.get('exit_code')}; "
                     f"elapsed={stage.get('wall_seconds', 0):.2f}s; "
                     f"[log]({stage['name']}.log)")
    if config.get('geometry_loss'):
        geometry = config['geometry_loss']
        lines += ['', f"- Geometry loss weights: mask={geometry['mask_weight']}, normal-depth={geometry['normal_depth_weight']}",
                  f"- Normal-depth schedule: start={geometry['normal_start_iteration']}, ramp={geometry['normal_ramp_iterations']} steps",
                  '- Training loss includes enabled geometry terms; validation gates/metrics use image loss only.']
    if config.get('surfel', {}).get('photometric_loss') == 'l1_dssim':
        lines += ['', '- Actual image objective: L1 + D-SSIM from step 1 (overrides the inherited loss curriculum).',
                  '- Normal optimization: reflection gradients plus scheduled propagation when enabled; mask BCE retained.',
                  f"- Normal propagation settings: {config['surfel']['normal_propagation']}"]
    if config.get('surfel'):
        lines += [f"- Quiet refinement settings: {config['surfel'].get('refinement', {})}",
                  '- D-SSIM convention: mean((1 - SSIM) / 2).']
    if "training_result" in record:
        lines.extend(["", f"- Training result: {record['training_result']}"])
    for metric, value in record.get("mean_metrics", {}).items():
        lines.append(f"- Mean per-view {metric}: {value}")
    if "error" in record:
        lines.extend(["", f"Failure: {record['error']}"])
    lines.extend(["", "## Interpretation (fill in after inspection)", "",
                  "- Observations and image comparisons:", "- Limitations / confounders:",
                  "- Next action:", "",
                  "Process wall times include startup and logging. FPS, peak VRAM, and crop",
                  "metrics are not measured by this runner. GPU metadata is a hardware snapshot."])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    refresh_output_index(run_dir.parent.parent)


def run_stage(name: str, command: list[str], root: Path, run_dir: Path, record: dict) -> None:
    stage = {"name": name, "command": command, "cwd": str(root),
             "status": "running", "started_utc": utc_now()}
    record["stages"].append(stage)
    write_record(run_dir, record)
    print(f"\n[{name}] {subprocess.list2cmdline(command)}", flush=True)
    started = time.perf_counter()
    process = None
    try:
        with (run_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace", bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            finally:
                process.stdout.close()
            code = process.wait()
        stage.update(exit_code=code, status="completed" if code == 0 else "failed")
        if code:
            raise StageFailed(name, code)
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stage["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        if process is not None:
            stage["exit_code"] = process.returncode
        raise
    finally:
        stage.update(wall_seconds=time.perf_counter() - started, finished_utc=utc_now())
        write_record(run_dir, record)


def collect_results(run_dir: Path, record: dict) -> None:
    text = (run_dir / "train.log").read_text(encoding="utf-8")
    match = re.search(r"Training finished after ([\d,]+) iterations in ([\d.]+)s", text)
    if match:
        stopping = re.search(r"^Stopping at iteration .*", text, re.MULTILINE)
        completed = int(match[1].replace(",", ""))
        record["training_result"] = {
            "completed_iterations": completed, "trainer_seconds": float(match[2]),
            "stop_reason": stopping[0] if stopping else (
                "configured iteration limit" if completed == record["config"]["training"]["iterations"]
                else "unknown; inspect train.log")}
    else:
        record["training_result"] = {"completed_iterations": None,
                                     "stop_reason": "not found; inspect train.log"}
    stage = re.search(r"^Final loss stage: (mse|dssim|l1_dssim)$", text, re.MULTILINE)
    if stage:
        record["training_result"]["loss_stage"] = stage[1]
    sh_stage = re.search(r"^Final SH degree: (\d+) \(configured maximum: (\d+)\)$", text, re.MULTILINE)
    if sh_stage:
        record["training_result"]["sh_degree"] = int(sh_stage[1])
    surfel_stage = re.search(r"^Final surfel stage: (gaussian_rgb|rgb|deferred)$", text, re.MULTILINE)
    if surfel_stage:
        record["training_result"]["surfel_stage"] = surfel_stage[1]
    model = run_dir / "model.ply"
    record["model_artifacts"] = [fingerprint(path) for path in sorted(run_dir.glob("model.*"))
                                 if path.is_file()]
    if not model.is_file():
        raise FileNotFoundError(f"Trainer did not save {model}")
    if record["config"]["model"]["sh_degree"] and not model.with_suffix(".sh.npz").is_file():
        raise FileNotFoundError(f"SH training did not save {model.with_suffix('.sh.npz')}")
    if (record.get("pipeline") == "surfel" and record['config']['model']['geometry'] == 'surfel'
            and not model.with_suffix(".surfel.npz").is_file()):
        raise FileNotFoundError(f"Surfel training did not save {model.with_suffix('.surfel.npz')}")


def collect_metrics(run_dir: Path, record: dict) -> None:
    with (run_dir / "test-metrics.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = record["inputs"]["manifests"][record["evaluation"]["manifest"]]["frame_paths"]
    if [row["file"] for row in rows] != expected:
        raise ValueError("Evaluation CSV does not match the requested manifest's frames")
    means = {}
    for metric in record["evaluation"]["metrics"]:
        key = "psnr_db" if metric == "psnr" else metric
        value = math.fsum(float(row[key]) for row in rows) / len(rows)
        if math.isnan(value):
            raise ValueError(f"Evaluation produced NaN for {key}")
        means[key] = value if math.isfinite(value) else str(value)
    record["mean_metrics"] = means
    record["evaluation"]["views"] = len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Training YAML; its relative paths retain their original meaning.")
    parser.add_argument("--pipeline", choices=("gaussian", "surfel"), default="gaussian", help="Choose the baseline or independent surfel research passes.")
    parser.add_argument("--name", help="Optional suffix, e.g. cap1k gives scene-Oct28-1305-cap1k (Singapore time). Repeated names get -2, -3, etc.")
    parser.add_argument("--note", default="", help="Research question/purpose to save in the summary.")
    parser.add_argument("--metrics", nargs="+", choices=("psnr", "ssim", "lpips"), default=["psnr", "ssim", "lpips"])
    parser.add_argument("--eval-manifest", default="transforms_test.json")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--check-gradients", action="store_true", help="Run the existing gradient checker on the configured device before training.")
    parser.add_argument("--prepare-only", action="store_true", help="Validate inputs and snapshot the run without launching training/evaluation.")
    return parser


def run_experiment(args: argparse.Namespace, root: Path = ROOT) -> Path:
    source = args.config.expanduser().resolve()
    config_class = TrainingConfig
    if args.pipeline == "surfel":
        sys.path.insert(0, str(root / "src"))
        from cp4282_gs.surfels.config import SurfelTrainingConfig
        config_class = SurfelTrainingConfig
    config = config_class.load(source).printable()
    if not config["reporting"]["save_ply"]:
        raise ValueError("reporting.save_ply must be true for post-training evaluation")
    now = datetime.now(SINGAPORE_TIMEZONE)
    stamp = f"{MONTHS[now.month - 1]}{now.day:02d}-{now:%H%M}"
    scene = re.sub(r"[^A-Za-z0-9_-]", "_", Path(config["paths"]["data"]).name)
    if args.name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        raise ValueError("Name suffix must contain letters/digits/dots/dashes/underscores, starting with a letter or digit")
    name = f"{scene}-{stamp}" + (f"-{args.name}" if args.name else "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
        raise ValueError("Run name must be 1-120 letters/digits/dots/dashes/underscores, starting with a letter or digit")
    base_name = name
    outputs = (root / "outputs").resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    suffix = 2
    while True:
        run_dir = (outputs / name).resolve()
        if not run_dir.is_relative_to(outputs):
            raise ValueError("Run directory must be inside outputs/")
        try:
            run_dir.mkdir()  # Reserve atomically; never reuse an existing run.
            break
        except FileExistsError:
            name = f"{base_name}-{suffix}"
            suffix += 1
    original_output = config["paths"]["output"]
    config["paths"]["output"] = str(run_dir / "model.png")
    record = {"run_id": name, "status": "preparing", "started_utc": utc_now(), "pipeline": args.pipeline,
              "note": args.note, "source_config": str(source), "original_output": original_output,
              "runner_command": sys.argv, "config": config, "stages": [],
              "evaluation": {"manifest": args.eval_manifest,
                             "metrics": list(dict.fromkeys(args.metrics)), "lpips_net": args.lpips_net,
                             "aggregation": "arithmetic mean of per-view metric values"}}
    started = time.perf_counter()
    try:
        write_record(run_dir, record)
        shutil.copyfile(source, run_dir / "original-config.yaml")
        record["source_config_fingerprint"] = fingerprint(source)
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        record["inputs"] = inspect_inputs(config, args.eval_manifest, run_dir)
        record["environment"] = snapshot_provenance(root, run_dir)
        python = sys.executable
        train_script = root / ("scripts/train_surfels.py" if args.pipeline == "surfel" else "3dgs_trainer.py")
        is_surfel_geometry = args.pipeline == "surfel" and config['model']['geometry'] == 'surfel'
        eval_script = root / ("scripts/evaluate_surfels.py" if is_surfel_geometry else "3dgs_evaluator.py")
        train = [python, "-u", str(train_script), str(config_path)]
        evaluation = [python, "-u", str(eval_script), str(run_dir / "model.ply"),
                      config["paths"]["data"], "--manifest", args.eval_manifest,
                      "--arch", config["runtime"]["arch"],
                      "--width", str(config["model"]["width"]),
                      "--height", str(config["model"]["height"]),
                      "--background", config["model"]["background"],
                      "--lpips-net", args.lpips_net, "--output-dir", str(run_dir / "test-renders"),
                      "--csv", str(run_dir / "test-metrics.csv"),
                      "--metrics", *record["evaluation"]["metrics"]]
        if config["model"]["sh_degree"] == 0:
            evaluation.extend(["--sh-degree", "0"])
        if config["model"]["tile_pair_capacity"] is not None:
            evaluation.extend(["--tile-pair-capacity", str(config["model"]["tile_pair_capacity"])])
        planned = []
        if args.check_gradients:
            device = "cpu" if config["runtime"]["arch"] == "cpu" else "cuda:0"
            checker = root / ("scripts/check_surfel_gradients.py" if is_surfel_geometry else "3dgs_gradient_check_gpu.py")
            planned.append(("gradients", [python, "-u", str(checker), "--device", device]))
        planned.extend([("train", train), ("eval", evaluation)])
        record["planned_commands"] = dict(planned)
        print(f"Run directory: {run_dir}", flush=True)
        if args.prepare_only:
            record["status"] = "prepared"
        else:
            record["status"] = "running"
            for stage, command in planned:
                run_stage(stage, command, root, run_dir, record)
                if stage == "train":
                    collect_results(run_dir, record)
            collect_metrics(run_dir, record)
            record["status"] = "completed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = str(error) or type(error).__name__
        print(f"Run {record['status']}; evidence saved in {run_dir}", file=sys.stderr)
        raise
    finally:
        record.update(finished_utc=utc_now(), total_wall_seconds=time.perf_counter() - started)
        write_record(run_dir, record)
    print(f"Run {record['status']}. Summary: {run_dir / 'summary.md'}", flush=True)
    return run_dir


def main(default_pipeline="gaussian", default_eval_manifest="transforms_test.json") -> int:
    parser = build_parser()
    parser.set_defaults(pipeline=default_pipeline, eval_manifest=default_eval_manifest)
    args = parser.parse_args()
    try:
        run_experiment(args)
    except KeyboardInterrupt:
        return 130
    except StageFailed as error:
        print(error, file=sys.stderr)
        return error.returncode if 0 < error.returncode < 126 else 1
    except (OSError, ValueError, KeyError, csv.Error, yaml.YAMLError) as error:
        print(f"Experiment error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
