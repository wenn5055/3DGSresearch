"""Train the independent flattened-surfel RGB or deferred-specular pass."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src')]
from cp4282_gs.surfels.workflow import train
from cp4282_gs.surfels.config import SurfelTrainingConfig
import warp as wp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = SurfelTrainingConfig.load(args.config)
    train(args.config)
    if config['runtime']['arch'] != 'cpu':
        wp.synchronize_device('cuda:0')
        # Fresh training process; includes Warp allocations, excludes driver/context
        # and allocations made outside Warp. This is not total device-wide VRAM.
        print(f'Training Warp pool high bytes: {wp.get_mempool_used_mem_high("cuda:0")}', flush=True)
