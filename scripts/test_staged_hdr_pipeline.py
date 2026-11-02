"""Check scene isolation and the dependency between HDR training stages."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import run_staged_hdr as pipeline


class StagedHDRTests(unittest.TestCase):
    def test_new_scene_replaces_all_old_scene_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            path = work / 'frozen.yaml'
            pipeline.write_config(pipeline.ROOT / 'config/materials_chrome_frozen_hdr.yaml',
                                  path, work / 'new-data', work / 'new-initial.ply', work / 'new-model.png')
            config = pipeline.SurfelTrainingConfig.load(path).printable()
            self.assertEqual(config['paths']['data'], str(work / 'new-data'))
            self.assertEqual(config['paths']['init_ply'], str(work / 'new-initial.ply'))
            self.assertEqual(config['paths']['output'], str(work / 'new-model.png'))
            self.assertTrue(config['surfel']['freeze_geometry'])
            self.assertEqual(config['training']['iterations'], 10000)
            self.assertEqual(config['surfel']['shading_model'], 'hdr_angular')

    def exercise_chain(self, missing_sidecar=False):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            frozen = work / 'actual-timestamped-frozen-run'
            frozen.mkdir()
            (frozen / 'model.ply').touch()
            if not missing_sidecar:
                (frozen / 'model.surfel.npz').touch()
            (work / 'refine.yaml').write_text(yaml.safe_dump({'paths': {'init_ply': 'old-scene.ply'}}))
            state = {'initializer_command': ['initializer']}
            events = []

            def train(config, name, metrics):
                events.append(name)
                if len(events) == 2:
                    actual = yaml.safe_load(config.read_text())['paths']['init_ply']
                    self.assertEqual(actual, str(frozen / 'model.ply'))
                return frozen if len(events) == 1 else work / 'refined-run'

            with patch.object(pipeline.subprocess, 'run') as initialize, patch.object(pipeline, 'train_stage', side_effect=train):
                if missing_sidecar:
                    with self.assertRaisesRegex(FileNotFoundError, 'surfel.npz'):
                        pipeline.execute(work, state, ['psnr'])
                    self.assertEqual(events, ['frozen-hdr-10k'])
                    self.assertEqual(json.loads((work / 'pipeline.json').read_text())['status'], 'failed')
                else:
                    pipeline.execute(work, state, ['psnr'])
                    self.assertEqual(events, ['frozen-hdr-10k', 'staged-hdr-refine-5k'])
                    self.assertEqual(state['status'], 'completed')
                initialize.assert_called_once_with(['initializer'], cwd=pipeline.ROOT, check=True)

    def test_refinement_uses_actual_frozen_run(self):
        self.exercise_chain()

    def test_missing_learned_materials_blocks_refinement(self):
        self.exercise_chain(missing_sidecar=True)


if __name__ == '__main__':
    unittest.main()
