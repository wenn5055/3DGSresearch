"""Run a Gaussian control, fixed-RGB surfel, or deferred surfel project pass."""

from run_experiment import main


if __name__ == '__main__':
    raise SystemExit(main(default_pipeline='surfel', default_eval_manifest='transforms_val.json'))
