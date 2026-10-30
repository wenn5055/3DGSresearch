# Archive candidates

Reviewed 29 October 2026. Nothing moved or deleted.
Each candidate has a short reason and successor in its top-of-file comment.

| File | Why archive | Successor / keep instead |
| --- | --- | --- |
| `3dgs_renderer_v1.py` | Sequential teaching renderer; only used by old synthetic trainers | `3dgs_evaluator.py` for trained Gaussian models |
| `3dgs_1_syn_trainer.py` | One-splat synthetic teaching exercise | `3dgs_k_syn_trainer.py` was the next teaching step; `3dgs_trainer.py` is the full trainer |
| `3dgs_k_syn_trainer.py` | Small synthetic teaching exercise | `3dgs_trainer.py`; project extensions under `src/cp4282_gs/surfels/` |
| `shared/camera.py` | Used only by the teaching renderer/trainer chain | Active camera loading in `3dgs_trainer.py` |
| `shared/projected_gaussians.py` | Used only by the teaching renderer | `gaussian_first_tile_workspace_gpu.py` and `shared/tile_builder.py` |
| `scripts/check_setup.py` | Hard-coded Lego setup check | `scripts/run_experiment.py CONFIG --prepare-only` checks the configured dataset |
| `3dgs_trainer_annotated.md` | Describes the original unimplemented backward pass | `3dgs_trainer.py` and current workflow guides |
| `shared_annotated.md` | Teaching-era map predates D-SSIM, SH and surfels | Current `shared/` docstrings and `project-passes.md` |
| `docs/research-design-notes.md` | Original design rationale contains superseded status claims | `project-plan.md`, `project-passes.md`, `materials-red-rose-silver.md`, `findings.md` |

Archive the teaching renderer, both synthetic trainers and their two exclusive helpers together.
Keep the main Gaussian trainer/evaluator, tile workspace, all other shared modules,
surfel modules, tests and sweep tools: current workflows still use them.
Keep baseline/ablation configs, findings, consultation history, AI-use records and asset attribution.
`command-reference.md` still provides detailed manual logging/debugging instructions.
Before moving files, update remaining documentation links and compile/help command lists.

Python headers were applied after the deferred propagation sweep completed, preserving
its source checks. Archive notes do not change executable code.
