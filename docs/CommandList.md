# Experiment commands (PowerShell)

Run from the repository root with the project virtual environment activated.
These are instructions; creating this document does not run training or checks.
Use a fresh run ID for each experiment and complete a
[run record](run-record-template.md). See [TODO.md](TODO.md) for priorities.

## Prepare a recorded Lego baseline

Choose an unused ID. New-Item intentionally fails if the run directory exists;
choose another ID instead of overwriting an earlier experiment.

```powershell
$runId = 'lego-rgb-r001'
$runDir = Join-Path 'outputs' $runId
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop
Copy-Item -LiteralPath 'config/3dgs_training_gpu.yaml' -Destination "config/$runId.yaml" -ErrorAction Stop
```

Edit `config/lego-rgb-r001.yaml` before continuing. Set `paths.output` to
`../outputs/lego-rgb-r001/model.png`; keep the Lego data and initialization paths.
Set and record the intended budget, learning-rate schedule, seed, and stopping
policy. For a fixed completed-step comparison, set `convergence.enabled: false`.
Leave the runnable config under config/: its relative paths are resolved there.

Record provenance after finishing the config edits:

```powershell
git rev-parse HEAD | Set-Content "$runDir/commit.txt"
git status --short | Set-Content "$runDir/git-status.txt"
git diff HEAD --output="$runDir/working-tree.patch"
Copy-Item -LiteralPath "config/$runId.yaml" -Destination "$runDir/config.yaml"
Copy-Item -LiteralPath 'shared/training_config.py' -Destination "$runDir/training_config_source.py"
python --version 2>&1 | Set-Content "$runDir/python-version.txt"
python -m pip freeze | Set-Content "$runDir/packages.txt"
nvidia-smi | Set-Content "$runDir/gpu.txt"
Get-FileHash -LiteralPath 'data/lego/init.ply' -Algorithm SHA256 | Format-List | Out-File "$runDir/initialization-hash.txt"
```

The config snapshot plus defaults source records omitted settings. The snapshot
is for provenance, not a runnable config at its new location. Git diff excludes
untracked files: commit implementation/config changes or snapshot their contents
as well. Record the dataset version/source and exact commands in the run record.

## Check, train, then evaluate

Run each command separately and inspect the exit code and log. Stop and resolve
failures before proceeding. These checks validate existing code; new kernels
will require additional checks designed for their behavior.

```powershell
python scripts/check_setup.py 2>&1 | Tee-Object -FilePath "$runDir/setup.log"
$LASTEXITCODE | Set-Content "$runDir/setup-exit-code.txt"

python 3dgs_gradient_check_gpu.py --device cpu 2>&1 | Tee-Object -FilePath "$runDir/gradient-cpu.log"
$LASTEXITCODE | Set-Content "$runDir/gradient-cpu-exit-code.txt"

python 3dgs_gradient_check_gpu.py --device cuda:0 2>&1 | Tee-Object -FilePath "$runDir/gradient-gpu.log"
$LASTEXITCODE | Set-Content "$runDir/gradient-gpu-exit-code.txt"

$trainTimer = [System.Diagnostics.Stopwatch]::StartNew()
python -u 3dgs_trainer.py "config/$runId.yaml" 2>&1 | Tee-Object -FilePath "$runDir/train.log"
$LASTEXITCODE | Set-Content "$runDir/train-exit-code.txt"
$trainTimer.Stop()
$trainTimer.Elapsed.TotalSeconds | Set-Content "$runDir/train-process-seconds.txt"

python -u 3dgs_evaluator.py "$runDir/model.ply" data/lego --manifest transforms_test.json --arch gpu --resolution 256 --background white --sh-degree 0 --metrics psnr ssim lpips --output-dir "$runDir/test-renders" --csv "$runDir/test-metrics.csv" 2>&1 | Tee-Object -FilePath "$runDir/eval.log"
$LASTEXITCODE | Set-Content "$runDir/eval-exit-code.txt"
```

Match evaluation resolution/background to the actual run. LPIPS requires its
dependencies and pretrained weights; record the backbone (default: alex).
If deliberately using only PSNR/SSIM for a quick check, record LPIPS as omitted.
The evaluator produces per-view metrics and prints their means; it does not
currently benchmark FPS, peak VRAM, or reflection-crop metrics.

## Save the human-readable summary

```powershell
New-Item -ItemType Directory -Path 'docs/experiments' -Force | Out-Null
Copy-Item -LiteralPath 'docs/run-record-template.md' -Destination "docs/experiments/$runId.md"
```

Fill that record from the saved evidence. Keep compact summaries and reproducible
configs in Git; raw logs, renders, and trained models stay under outputs/. Back
up artifacts separately if they must be retained across machines.
