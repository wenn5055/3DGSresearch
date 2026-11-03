"""Compare original Blender lighting with a learned HDR map in the renderer's direction convention."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PROJECTION_SOURCE = 'https://github.com/blender/blender/blob/v5.0.1/intern/cycles/kernel/camera/projection.h'


def renderer_directions(width, height):
    y, x = np.mgrid[:height, :width]
    theta = np.pi*(y+.5)/height
    phi = 2*np.pi*((x+.5)/width-.5)
    return np.stack((np.sin(theta)*np.sin(phi), np.cos(theta), np.sin(theta)*np.cos(phi)), axis=-1)


def blender_uv(directions):
    d = directions/np.linalg.norm(directions, axis=-1, keepdims=True)
    return np.mod(.5-np.arctan2(d[..., 1], d[..., 0])/(2*np.pi), 1), 1-np.arccos(np.clip(d[..., 2], -1, 1))/np.pi


def sample_bottom_up(image, u, v):
    height, width = image.shape[:2]
    x, y = u*width-.5, np.clip(v*height-.5, 0, height-1)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    tx, ty = (x-ix)[..., None], (y-iy)[..., None]
    top = (1-tx)*image[iy, ix % width]+tx*image[iy, (ix+1) % width]
    bottom = (1-tx)*image[np.minimum(iy+1, height-1), ix % width]+tx*image[np.minimum(iy+1, height-1), (ix+1) % width]
    return (1-ty)*top+ty*bottom


def display(linear, exposure):
    value = np.maximum(linear*2**exposure, 0)
    srgb = np.where(value <= .0031308, 12.92*value, 1.055*value**(1/2.4)-.055)
    return Image.fromarray(np.uint8(np.clip(srgb, 0, 1)*255))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Completed HDR run directory.')
    parser.add_argument('--scene', type=Path, default=ROOT/'data/blender/justChromeBall.blend')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blender', type=Path, default=Path('C:/Program Files/Blender Foundation/Blender 5.0/blender.exe'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.run/'model.surfel.npz'
    with np.load(source) as state:
        if str(state['shading_model']) != 'hdr_angular':
            raise ValueError('This command expects the HDR/angular checkpoint.')
        learned = np.logaddexp(0, state['environment_logits'])
    command = [str(args.blender), '--background', '--disable-autoexec', str(args.scene.resolve()),
               '--python-exit-code', '1', '--python', str(ROOT/'scripts/export_blender_environment.py'),
               '--', '--output', str((args.output/'reference-source').resolve())]
    subprocess.run(command, cwd=ROOT, check=True)
    metadata = json.loads((args.output/'reference-source/environment.json').read_text())
    with np.load(args.output/'reference-source/environment.npz') as source_pixels:
        native = source_pixels['rgb_bottom_up']
    height, width = learned.shape[:2]
    u, v = blender_uv(renderer_directions(width, height))
    reference = sample_bottom_up(native, u, v)*metadata['world_strength']
    np.savez_compressed(args.output/'aligned-linear.npz', reference=reference, learned=learned)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    sheet = Image.new('RGB', (2*width, 2*(height+42)), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    for row, exposure in enumerate((0, -2)):
        for column, (label, values) in enumerate((('Original forest', reference), ('Learned environment', learned))):
            y = row*(height+42)
            draw.text((column*width+8, y+10), f'{label} | {exposure:+d} EV | {width}x{height}', font=font, fill='black')
            sheet.paste(display(values, exposure), (column*width, y+42))
    sheet.save(args.output/'comparison.png')
    report = {'status': 'completed', 'run': str(args.run.resolve()), 'reference': metadata,
              'checkpoint_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'aligned_resolution': [width, height], 'exposures_ev': [0, -2],
              'alignment': 'same world directions; renderer Y-polar latlong reprojected to Blender Z-polar latlong; no fitted rotation or brightness scaling',
              'reference_sampling': 'bilinear point samples from original-resolution linear pixels; same target resolution, no supersampled area filtering',
              'projection_source': PROJECTION_SOURCE,
              'interpretation': 'structural diagnostic; albedo/reflection/light ambiguity means radiance disagreement alone is not a calibrated reconstruction error',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('Open:', args.output/'comparison.png', flush=True)


if __name__ == '__main__':
    main()
