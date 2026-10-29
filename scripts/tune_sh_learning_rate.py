"""Compare SH learning rates on validation, reusing a matched baseline when supplied."""

from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

import run_experiment as runner


def training_sources(root):
    paths = sorted(set(root.glob("*.py")) | set((root / "shared").glob("*.py")))
    return {path.relative_to(root).as_posix(): runner.fingerprint(path)["sha256"] for path in paths}


def comparable(config):
    config = deepcopy(config)
    config["paths"].pop("output", None)
    config["learning_rates"].pop("feature_rest", None)
    return config


def read_record(directory):
    return json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))


def read_metrics(path, config):
    manifest = json.loads((Path(config["paths"]["data"]) / "transforms_val.json").read_text())
    expected = [frame["file_path"] for frame in manifest["frames"]]
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or [row["file"] for row in rows] != expected:
        raise ValueError("Evaluation CSV does not match the complete validation split.")
    means = {}
    for metric in ("psnr_db", "ssim"):
        values = [float(row[metric]) for row in rows]
        if any(math.isnan(value) for value in values):
            raise ValueError(f"Validation {metric} contains NaN.")
        value = math.fsum(values) / len(values)
        means[metric] = value if math.isfinite(value) else str(value)
    return means


def write_progress(directory, plan):
    plan["updated_utc"] = runner.utc_now()
    pending = directory / "sweep.json.tmp"
    pending.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(directory / "sweep.json")
    runner.refresh_output_index(directory.parent.parent)
    columns = ["rate", "status", "iterations", "sh_degree", "psnr_db", "ssim", "eligible", "run_dir", "metrics_csv"]
    with (directory / "validation-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plan["candidates"])
    lines = ["# SH learning-rate validation sweep", "",
             f"Status: **{plan['status']}**. Only `learning_rates.feature_rest` varies.", "",
             "Selection: highest mean validation PSNR; validation SSIM breaks exact ties. "
             "A candidate must complete the full iteration budget and reach the configured SH degree.", "",
             f"Fixed seed: {plan['base_config']['runtime']['seed']}; "
             f"iterations: {plan['base_config']['training']['iterations']}; "
             f"maximum SH degree: {plan['base_config']['model']['sh_degree']}.", "",
             "| SH learning rate | Status | Steps | SH degree | Validation PSNR | Validation SSIM | Eligible |",
             "| ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for candidate in plan["candidates"]:
        psnr = f"{float(candidate['psnr_db']):.4f}" if "psnr_db" in candidate else "-"
        ssim = f"{float(candidate['ssim']):.6f}" if "ssim" in candidate else "-"
        lines.append(f"| {candidate['rate']:g} | {candidate['status']} | {candidate.get('iterations', '-')} | "
                     f"{candidate.get('sh_degree', '-')} | {psnr} | {ssim} | {candidate.get('eligible', '-')} |")
    if plan.get("best_rate") is not None:
        lines += ["", f"Selected learning rate: **{plan['best_rate']:g}**.", "",
                  "Frozen settings: [best-config.yaml](best-config.yaml). This is a one-seed "
                  "selection; repeat promising settings with additional matched seeds before final claims."]
    else:
        lines += ["", "No winner selected until the sweep finishes with an eligible candidate."]
    if plan.get("error"):
        lines += ["", f"Error: {plan['error']}"]
    lines += ["", "Full provenance, per-candidate log paths, and evidence paths are in "
              "[sweep.json](sweep.json). Test scores are not used for selection."]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(args, root=runner.ROOT):
    config = runner.TrainingConfig.load(args.config).printable()
    rates = args.rates
    if not rates or any(not math.isfinite(value) or value <= 0 for value in rates) or len(set(rates)) != len(rates):
        raise ValueError("Rates must be unique, finite, positive numbers.")
    if config["model"]["sh_degree"] == 0:
        raise ValueError("SH learning-rate tuning requires model.sh_degree > 0.")
    if config["reporting"]["eval_manifest"] != "transforms_val.json":
        raise ValueError("This sweep requires validation-controlled training: reporting.eval_manifest: transforms_val.json.")
    baseline = args.baseline_run.resolve() if args.baseline_run else None
    sources = training_sources(root)
    baseline_rate = None
    if baseline:
        baseline_config = runner.TrainingConfig.load(baseline / "config.yaml").printable()
        if comparable(baseline_config) != comparable(config):
            raise ValueError("Baseline training settings differ from the sweep configuration.")
        baseline_rate = baseline_config["learning_rates"]["feature_rest"]
        if baseline_rate not in rates:
            raise ValueError("The baseline learning rate must be included in --rates.")
        saved_sources = {row["path"]: row["sha256"] for row in json.loads((baseline / "source-files.json").read_text())}
        if any(saved_sources.get(path) != digest for path, digest in sources.items()):
            raise ValueError("Baseline trainer/evaluator sources differ; start a fresh sweep without --baseline-run.")
    now = datetime.now(runner.SINGAPORE_TIMEZONE)
    stamp = f"{runner.MONTHS[now.month - 1]}{now.day:02d}-{now:%H%M}"
    scene = re.sub(r"[^A-Za-z0-9_-]", "_", Path(config["paths"]["data"]).name)
    name = f"{scene}-{stamp}-sh{config['model']['sh_degree']}-lr-sweep"
    outputs = (root / "outputs").resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    directory = outputs / name
    suffix = 2
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            directory = outputs / f"{name}-{suffix}"
            suffix += 1
    (directory / "configs").mkdir()
    plan = {"version": 1, "status": "prepared", "created_utc": runner.utc_now(),
            "base_config": config, "training_sources": sources,
            "wait_for": [str(path.resolve()) for path in args.wait_for], "candidates": []}
    if baseline and str(baseline) not in plan["wait_for"]:
        plan["wait_for"].append(str(baseline))
    # Reuse/evaluate the baseline first, then train the other rates sequentially.
    ordered_rates = sorted(rates, key=lambda value: (value != baseline_rate, value))
    for rate in ordered_rates:
        label = "lr-" + str(rate).replace(".", "p").replace("+", "")
        candidate_config = deepcopy(config)
        candidate_config["learning_rates"]["feature_rest"] = rate
        candidate_path = directory / "configs" / f"{label}.yaml"
        candidate_path.write_text(yaml.safe_dump(candidate_config, sort_keys=False), encoding="utf-8")
        candidate = {"rate": rate, "label": label, "config": str(candidate_path), "status": "pending"}
        if baseline and rate == baseline_rate:
            candidate["baseline_run"] = str(baseline)
        plan["candidates"].append(candidate)
    write_progress(directory, plan)
    return directory


def wait_for_runs(directory, plan, timeout=21600):
    deadline = time.monotonic() + timeout
    last_message = None
    while True:
        running = [path for path in plan["wait_for"] if read_record(path)["status"] in ("preparing", "running")]
        if not running:
            return
        message = ", ".join(Path(path).name for path in running)
        if message != last_message:
            print(f"Waiting for existing jobs: {message}", flush=True)
            last_message = message
        plan["status"] = "waiting"
        write_progress(directory, plan)
        if time.monotonic() >= deadline:
            raise TimeoutError("Existing runs did not finish within six hours; inspect their run.json and resume later.")
        time.sleep(10)


def validate_baseline(directory, candidate, config):
    baseline = Path(candidate["baseline_run"])
    record = read_record(baseline)
    if record["status"] != "completed":
        raise ValueError(f"Baseline did not complete: {baseline}")
    output = directory / "validation" / candidate["label"]
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "metrics.csv"
    command = [sys.executable, str(runner.ROOT / "3dgs_evaluator.py"), str(baseline / "model.ply"),
               config["paths"]["data"], "--manifest", "transforms_val.json",
               "--sh-degree", str(record["training_result"]["sh_degree"]),
               "--arch", config["runtime"]["arch"], "--width", str(config["model"]["width"]),
               "--height", str(config["model"]["height"]), "--background", config["model"]["background"],
               "--metrics", "psnr", "ssim", "--output-dir", str(output / "renders"), "--csv", str(csv_path)]
    if config["model"]["tile_pair_capacity"] is not None:
        command += ["--tile-pair-capacity", str(config["model"]["tile_pair_capacity"])]
    candidate["validation_command"] = command
    with (output / "eval.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=runner.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return baseline, csv_path


def finish_candidate(candidate, run_dir, csv_path, config):
    record = read_record(run_dir)
    if record["status"] != "completed":
        raise ValueError(f"Candidate run did not complete: {run_dir}")
    if (comparable(record["config"]) != comparable(config)
            or record["config"]["learning_rates"]["feature_rest"] != candidate["rate"]):
        raise ValueError(f"Candidate run settings do not match the frozen sweep: {run_dir}")
    result = record["training_result"]
    candidate.update(read_metrics(csv_path, config))
    candidate.update(status="completed", iterations=result.get("completed_iterations"),
                     sh_degree=result.get("sh_degree"), run_dir=str(run_dir), metrics_csv=str(csv_path))
    candidate["eligible"] = (candidate["iterations"] == config["training"]["iterations"]
                             and candidate["sh_degree"] == config["model"]["sh_degree"])


def execute(directory, root=runner.ROOT):
    directory = directory.resolve()
    if not directory.is_relative_to((root / "outputs").resolve()):
        raise ValueError("Sweep directory must be inside outputs/.")
    plan = json.loads((directory / "sweep.json").read_text(encoding="utf-8"))
    if plan["version"] != 1:
        raise ValueError("Unsupported sweep version.")
    if plan["status"] == "completed":
        print(f"Sweep already completed: {directory / 'summary.md'}", flush=True)
        return
    try:
        plan.pop("error", None)
        wait_for_runs(directory, plan)
        for candidate in plan["candidates"]:
            if training_sources(root) != plan["training_sources"]:
                raise ValueError("Trainer/evaluator sources changed during the sweep; keep methods matched and start a new sweep.")
            if candidate["status"] == "completed":
                continue
            config = runner.TrainingConfig.load(Path(candidate["config"])).printable()
            if comparable(config) != comparable(plan["base_config"]) or config["learning_rates"]["feature_rest"] != candidate["rate"]:
                raise ValueError("Candidate settings changed after preparation; create a new sweep.")
            candidate["status"] = "running"
            candidate.pop("error", None)
            plan["status"] = "running"
            write_progress(directory, plan)
            print(f"Validation candidate {candidate['rate']:g} starting", flush=True)
            if "baseline_run" in candidate:
                run_dir, csv_path = validate_baseline(directory, candidate, config)
            else:
                # A resumed failed candidate gets a fresh run folder; existing evidence stays intact.
                attempt = len(candidate.setdefault("attempt_logs", [])) + 1
                log_path = directory / f"{candidate['label']}-attempt{attempt}.log"
                candidate["attempt_logs"].append(str(log_path))
                write_progress(directory, plan)
                args = runner.build_parser().parse_args([
                    candidate["config"], "--name", f"sh{config['model']['sh_degree']}-tune-{candidate['label']}",
                    "--eval-manifest", "transforms_val.json", "--metrics", "psnr", "ssim",
                    "--note", f"Validation-only SH learning-rate tuning; sweep: {directory.name}"])
                with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    run_dir = runner.run_experiment(args, root)
                csv_path = run_dir / "test-metrics.csv"  # Its manifest is explicitly validation.
            if training_sources(root) != plan["training_sources"]:
                raise ValueError("Trainer/evaluator sources changed while a candidate was running; its scores were not ranked.")
            finish_candidate(candidate, run_dir, csv_path, config)
            write_progress(directory, plan)
            print(f"Candidate {candidate['rate']:g}: validation PSNR={candidate['psnr_db']}, "
                  f"SSIM={candidate['ssim']}, eligible={candidate['eligible']}", flush=True)
        eligible = [candidate for candidate in plan["candidates"] if candidate.get("eligible")]
        if eligible:
            best = max(eligible, key=lambda row: (float(row["psnr_db"]), float(row["ssim"])))
            plan["best_rate"] = best["rate"]
            (directory / "best-config.yaml").write_bytes(Path(best["config"]).read_bytes())
        plan["status"] = "completed"
        write_progress(directory, plan)
        print(f"Sweep completed: {directory / 'summary.md'}", flush=True)
    except BaseException as error:
        plan["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        plan["error"] = str(error) or type(error).__name__
        for candidate in plan["candidates"]:
            if candidate["status"] == "running":
                candidate["status"] = plan["status"]
                candidate["error"] = plan["error"]
        write_progress(directory, plan)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0000625, 0.000125, 0.00025])
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--wait-for", type=Path, nargs="+", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error("Supply a base config or --resume SWEEP_DIRECTORY.")
    directory = args.resume.resolve() if args.resume else prepare(args)
    print(f"Sweep directory: {directory}", flush=True)
    if not args.prepare_only:
        execute(directory)


if __name__ == "__main__":
    main()
