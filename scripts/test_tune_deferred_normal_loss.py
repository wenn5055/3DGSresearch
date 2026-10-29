"""Check deferred tuning isolation, eligibility, recovery and validation selection."""

import contextlib
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from PIL import Image
import yaml

import tune_deferred_normal_loss as tune


class DeferredNormalTuningTests(unittest.TestCase):
    def setUp(self):
        self.root = tune.runner.ROOT / 'outputs' / f'normal-tuning-test-{uuid.uuid4().hex}'
        self.root.mkdir()
        self.addCleanup(self.cleanup)
        for name in ('config', 'data', 'src', 'scripts'):
            (self.root / name).mkdir()
        (self.root / 'src/kernel.py').write_text('# fixture kernel\n')
        pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]]
        for split in ('train', 'val'):
            (self.root / 'data' / split).mkdir()
            frames = []
            for i in range(2):
                Image.new('RGBA', (16, 16), (100, 0, 0, 255)).save(self.root / 'data' / split / f'r_{i}.png')
                frames.append({'file_path': f'./{split}/r_{i}', 'transform_matrix': pose})
            (self.root / 'data' / f'transforms_{split}.json').write_text(json.dumps({
                'camera_angle_x': 0.7, 'frames': frames}))
        self.path = self.root / 'config/base.yaml'
        self.path.write_text(yaml.safe_dump({
            'paths': {'data': '../data', 'output': '../unused.png', 'init_ply': None},
            'runtime': {'arch': 'cpu', 'seed': 7},
            'model': {'width': 16, 'height': 16, 'capacity': 4, 'initial_splats': 1},
            'training': {'iterations': 6}, 'convergence': {'enabled': False},
            'reporting': {'log_every': 1},
            'surfel': {'mode': 'deferred', 'specular_start_iteration': 2},
            'geometry_loss': {'mask_weight': 0.1, 'normal_depth_weight': 0.01,
                              'normal_start_iteration': 1, 'normal_ramp_iterations': 1},
        }))
        self.args = tune.build_parser().parse_args([str(self.path)])
        self.calls = []

    def cleanup(self):
        if not self.root.resolve().is_relative_to((tune.runner.ROOT / 'outputs').resolve()):
            raise ValueError('Fixture cleanup must remain under outputs/.')
        shutil.rmtree(self.root)

    def plan(self, directory):
        return json.loads((directory / 'sweep.json').read_text())

    def fake_run(self, args, root, psnr=25, ssim=0.8, steps=6, stage='deferred', pixels=6):
        self.assertEqual(args.pipeline, 'surfel')
        self.assertEqual(args.eval_manifest, 'transforms_val.json')
        config = tune.SurfelTrainingConfig.load(args.config).printable()
        weight = config['geometry_loss']['normal_depth_weight']
        self.calls.append(weight)
        folder = root / 'outputs' / f'candidate-{len(self.calls)}'
        folder.mkdir()
        config['paths']['output'] = str(folder / 'model.png')
        (folder / 'run.json').write_text(json.dumps({
            'status': 'completed', 'pipeline': 'surfel', 'config': config,
            'training_result': {'completed_iterations': steps, 'surfel_stage': stage,
                                'loss_stage': 'l1_dssim' if config['surfel']['photometric_loss'] == 'l1_dssim' else 'mse'},
            'evaluation': {'manifest': 'transforms_val.json'},
            'mean_metrics': {'psnr_db': 999, 'ssim': 1},  # Deliberately ignored.
        }))
        (folder / 'train.log').write_text(f'normal_weight={weight:.5f}, normal_pixels={pixels}\n')
        with (folder / 'test-metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['file', 'psnr_db', 'ssim'])
            writer.writeheader()
            writer.writerows({'file': f'./val/r_{i}', 'psnr_db': psnr, 'ssim': ssim} for i in range(2))
        return folder

    def run_sweep(self, directory, launch):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(tune.runner, 'run_experiment', side_effect=launch):
            tune.execute(directory, self.root)

    def test_prepare_changes_only_normal_weight_and_never_reads_test_split(self):
        directory = tune.prepare(self.args, self.root)
        plan = self.plan(directory)
        base = tune.SurfelTrainingConfig.load(self.path).printable()
        self.assertEqual([c['weight'] for c in plan['candidates']], [0, 0.005, 0.01, 0.02])
        self.assertTrue(all('/test/' not in row['path'].replace('\\', '/') for row in plan['input_files']))
        self.assertFalse((directory / 'best-config.yaml').exists())
        for candidate in plan['candidates']:
            candidate_config = tune.SurfelTrainingConfig.load(Path(candidate['config'])).printable()
            self.assertEqual(tune.comparable(candidate_config), tune.comparable(base))

    def test_selection_uses_complete_validation_and_ssim_tiebreak(self):
        directory = tune.prepare(self.args, self.root)

        def launch(args, root):
            weight = yaml.safe_load(args.config.read_text())['geometry_loss']['normal_depth_weight']
            return self.fake_run(args, root, psnr=20 if weight == 0 else 30,
                                 ssim=0.9 if weight == 0.01 else 0.8,
                                 stage='rgb' if weight == 0.02 else 'deferred')

        self.run_sweep(directory, launch)
        plan = self.plan(directory)
        self.assertEqual(plan['best_weight'], 0.01)
        self.assertFalse(plan['candidates'][-1]['eligible'])
        self.assertEqual(yaml.safe_load((directory / 'best-config.yaml').read_text())['geometry_loss']['normal_depth_weight'], 0.01)
        with patch.object(tune.runner, 'run_experiment') as run:
            tune.execute(directory, self.root)
        run.assert_not_called()

    def test_zero_can_win_and_inactive_normal_candidates_are_not_ranked(self):
        directory = tune.prepare(self.args, self.root)
        self.run_sweep(directory, lambda args, root: self.fake_run(args, root, pixels=0, psnr=80))
        plan = self.plan(directory)
        self.assertEqual(plan['best_weight'], 0)
        self.assertTrue(plan['candidates'][0]['eligible'])
        self.assertTrue(all(not c['eligible'] for c in plan['candidates'][1:]))
        index = (self.root / 'docs/output-index.md').read_text(encoding='utf-8')
        self.assertIn('Deferred normal-loss weight', index)
        self.assertIn('Selected 0 by validation PSNR', index)

    def test_data_and_nested_sources_are_frozen(self):
        for relative in ('src/kernel.py', 'data/val/r_0.png'):
            with self.subTest(relative=relative):
                directory = tune.prepare(self.args, self.root)
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with patch.object(tune.runner, 'run_experiment') as run, self.assertRaises(ValueError):
                    tune.execute(directory, self.root)
                run.assert_not_called()
                self.assertFalse((directory / 'best-config.yaml').exists())
                path.write_bytes(original)

    def test_resume_keeps_completed_candidates_and_retries_failure_from_scratch(self):
        directory = tune.prepare(self.args, self.root)
        count = 0

        def fail_second(args, root):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('interrupted candidate')
            return self.fake_run(args, root)

        with self.assertRaises(RuntimeError):
            self.run_sweep(directory, fail_second)
        self.assertEqual(self.plan(directory)['status'], 'failed')
        self.assertFalse((directory / 'best-config.yaml').exists())
        self.run_sweep(directory, self.fake_run)
        plan = self.plan(directory)
        self.assertEqual(self.calls, [0, 0.005, 0.01, 0.02])
        self.assertEqual(len(plan['candidates'][1]['attempt_logs']), 2)
        self.assertEqual(plan['status'], 'completed')

    def test_rejects_test_evaluation_and_partial_validation_csv(self):
        directory = tune.prepare(self.args, self.root)
        candidate = self.plan(directory)['candidates'][0]
        config = tune.SurfelTrainingConfig.load(Path(candidate['config'])).printable()
        args = tune.runner.build_parser().parse_args([candidate['config'], '--pipeline', 'surfel',
                                                     '--eval-manifest', 'transforms_val.json'])
        folder = self.fake_run(args, self.root)
        record = tune.read_record(folder)
        record['evaluation']['manifest'] = 'transforms_test.json'
        (folder / 'run.json').write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            tune.finish_candidate(candidate, folder, config)
        record['evaluation']['manifest'] = 'transforms_val.json'
        (folder / 'run.json').write_text(json.dumps(record))
        (folder / 'test-metrics.csv').write_text('file,psnr_db,ssim\n./val/r_0,50,0.8\n')
        with self.assertRaises(ValueError):
            tune.finish_candidate(candidate, folder, config)

    def test_invalid_sweeps_are_rejected(self):
        for weights in ([0], [-1, 0.01], [0.01, 0.01], [0, float('nan')]):
            self.args.weights = weights
            with self.assertRaises(ValueError):
                tune.prepare(self.args, self.root)
        self.args.weights = [0, 0.01]
        original = yaml.safe_load(self.path.read_text())
        for section, key, value in [('surfel', 'mode', 'rgb'), ('convergence', 'enabled', True),
                                    ('training', 'iterations', 2)]:
            config = deepcopy(original)
            config[section][key] = value
            self.path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                tune.prepare(self.args, self.root)

    def test_waits_for_existing_gpu_jobs(self):
        config = yaml.safe_load(self.path.read_text())
        config['runtime']['arch'] = 'gpu'
        self.path.write_text(yaml.safe_dump(config))
        folder = self.root / 'outputs/active'
        folder.mkdir(parents=True)
        record = {'status': 'running', 'config': config}
        (folder / 'run.json').write_text(json.dumps(record))
        directory = tune.prepare(self.args, self.root)
        plan = self.plan(directory)
        self.assertIn(str(folder.resolve()), plan['wait_for'])

        def complete(_):
            record['status'] = 'completed'
            (folder / 'run.json').write_text(json.dumps(record))

        with contextlib.redirect_stdout(io.StringIO()), patch.object(tune.time, 'sleep', side_effect=complete) as sleep:
            tune.wait_for_runs(directory, plan, self.root)
        sleep.assert_called_once_with(10)

    def test_incomplete_runs_cannot_win(self):
        directory = tune.prepare(self.args, self.root)
        with self.assertRaisesRegex(ValueError, 'No eligible candidates'):
            self.run_sweep(directory, lambda args, root: self.fake_run(args, root, steps=5))
        self.assertFalse((directory / 'best-config.yaml').exists())
        self.assertEqual(self.plan(directory)['status'], 'failed')

    def test_paper_sweep_tunes_propagation_and_retains_mask(self):
        config = yaml.safe_load(self.path.read_text())
        config['geometry_loss']['normal_depth_weight'] = 0
        config['surfel'].update(photometric_loss='l1_dssim', normal_propagation={
            'enabled': True, 'interval': 1, 'until_iteration': 5})
        self.path.write_text(yaml.safe_dump(config))
        directory = tune.prepare(self.args, self.root)
        plan = self.plan(directory)
        self.assertEqual(plan['parameter'], 'propagation_scale')
        self.assertEqual([row['weight'] for row in plan['candidates']], [1.5, 1, 1.25])
        baseline = tune.SurfelTrainingConfig.load(self.path).printable()
        for row in plan['candidates']:
            settings = tune.SurfelTrainingConfig.load(Path(row['config'])).printable()
            self.assertEqual(tune.comparable(settings, 'propagation_scale'), tune.comparable(baseline, 'propagation_scale'))
            self.assertEqual(settings['geometry_loss']['mask_weight'], 0.1)
            self.assertEqual(settings['geometry_loss']['normal_depth_weight'], 0)

        def launch(args, root):
            settings = tune.SurfelTrainingConfig.load(args.config).printable()
            scale = settings['surfel']['normal_propagation']['scale_factor']
            folder = self.fake_run(args, root, psnr=40 if scale == 1.25 else 30)
            with (folder / 'train.log').open('a') as handle:
                handle.write('normal propagation: expanded=2, colours_reset=1\n')
            return folder

        self.run_sweep(directory, launch)
        plan = self.plan(directory)
        self.assertEqual(plan['best_scale'], 1.25)
        self.assertNotIn('best_weight', plan)
        index = (self.root / 'docs/output-index.md').read_text(encoding='utf-8')
        self.assertIn('normal-propagation scale', index)

    def test_paper_sweep_rejects_missing_propagation_events(self):
        config = yaml.safe_load(self.path.read_text())
        config['geometry_loss']['normal_depth_weight'] = 0
        config['surfel']['normal_propagation'] = {'enabled': True, 'until_iteration': 5}
        self.path.write_text(yaml.safe_dump(config))
        directory = tune.prepare(self.args, self.root)
        with self.assertRaisesRegex(ValueError, 'No eligible candidates'):
            self.run_sweep(directory, self.fake_run)


if __name__ == '__main__':
    unittest.main()
