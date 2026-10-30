"""Check provenance rejection and point/global-state separation in compact exports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement
import yaml

from prune_frozen_splats import export_pruned


class FrozenPruningTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.run, self.importance, self.data = [self.root / name for name in ('parent', 'weights', 'data')]
        for path in (self.run, self.importance, self.data):
            path.mkdir()
        manifest = self.data / 'transforms_train.json'
        manifest.write_text(json.dumps({'frames': [{}, {}]}))
        config = {'paths': {'data': str(self.data), 'output': str(self.run / 'model.png')}}
        (self.run / 'config.yaml').write_text(yaml.safe_dump(config))
        (self.run / 'run.json').write_text(json.dumps({'status': 'completed', 'config': config,
            'source_config': 'original.yaml', 'training_result': {'trainer_seconds': 5}}))
        (self.run / 'source-files.json').write_text('[]')
        (self.run / 'train.log').write_text('Parent training log')
        vertices = np.array([(0.,), (1.,), (2.,), (3.,)], dtype=[('x', 'f4')])
        PlyData([PlyElement.describe(vertices, 'vertex')]).write(self.run / 'model.ply')
        self.sh = np.arange(36, dtype=np.float32).reshape(4, 3, 3)
        self.environment = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
        np.savez(self.run / 'model.sh.npz', format_version=1, sh_rest=self.sh,
                 sh_degree=1, max_sh_degree=1)
        np.savez(self.run / 'model.surfel.npz', format_version=1,
                 reflection_logits=np.arange(4, dtype=np.float32), environment_logits=self.environment)
        np.savez(self.importance / 'weights.npz', total=np.array([4., 0., 2., 0.]))
        fingerprint = lambda path: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.metadata = {'run': str(self.run), 'manifest': manifest.name, 'training_views': 2,
            'coverage_sum': 6., 'inputs': [fingerprint(manifest)],
            'checkpoints': [fingerprint(self.run / name) for name in
                            ('model.ply', 'model.sh.npz', 'model.surfel.npz')],
            'weights_sha256': hashlib.sha256((self.importance / 'weights.npz').read_bytes()).hexdigest()}
        self.write_metadata()

    def write_metadata(self):
        (self.importance / 'summary.json').write_text(json.dumps(self.metadata))

    def test_global_environment_is_preserved_when_height_equals_point_count(self):
        output = export_pruned(self.run, self.importance, .5, self.root / 'compact')
        np.testing.assert_array_equal(PlyData.read(output / 'model.ply')['vertex']['x'], [0., 2.])
        with np.load(output / 'model.sh.npz') as sh:
            np.testing.assert_array_equal(sh['sh_rest'], self.sh[[0, 2]])
        with np.load(output / 'model.surfel.npz') as state:
            np.testing.assert_array_equal(state['reflection_logits'], [0., 2.])
            np.testing.assert_array_equal(state['environment_logits'], self.environment)
        derived = json.loads((output / 'run.json').read_text())
        self.assertEqual(derived['status'], 'derived')
        self.assertEqual(derived['derivation']['optimization_steps_after_pruning'], 0)
        self.assertNotEqual(derived['source_config'], 'original.yaml')
        self.assertEqual((output / 'train.log').read_bytes(), (self.run / 'train.log').read_bytes())
        self.assertEqual(len(PlyData.read(self.run / 'model.ply')['vertex']), 4)

    def test_stale_checkpoint_is_rejected_before_export(self):
        with (self.run / 'model.ply').open('ab') as handle:
            handle.write(b'changed')
        output = self.root / 'must-not-exist'
        with self.assertRaisesRegex(ValueError, 'Stale importance'):
            export_pruned(self.run, self.importance, .5, output)
        self.assertFalse(output.exists())

    def test_hdr_global_angular_knots_survive_compaction(self):
        path = self.run / 'model.surfel.npz'
        knots = np.linspace(-1, 1, 16, dtype=np.float32)
        np.savez(path, format_version=2, shading_model='hdr_angular', angular_gain_logits=knots,
                 reflection_logits=np.arange(4, dtype=np.float32), environment_logits=self.environment)
        self.metadata['checkpoints'][-1]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_metadata()
        output = export_pruned(self.run, self.importance, .5, self.root / 'compact-hdr')
        with np.load(output / 'model.surfel.npz') as state:
            np.testing.assert_array_equal(state['angular_gain_logits'], knots)
            np.testing.assert_array_equal(state['reflection_logits'], [0., 2.])
            np.testing.assert_array_equal(state['environment_logits'], self.environment)
            self.assertEqual(int(state['format_version']), 2)

    def test_held_out_importance_and_changed_weights_are_rejected(self):
        output = self.root / 'must-not-exist'
        self.metadata['manifest'] = 'transforms_val.json'
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'all training views'):
            export_pruned(self.run, self.importance, .5, output)
        self.metadata['manifest'] = 'transforms_train.json'
        self.metadata['weights_sha256'] = 'wrong'
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Weights changed'):
            export_pruned(self.run, self.importance, .5, output)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
