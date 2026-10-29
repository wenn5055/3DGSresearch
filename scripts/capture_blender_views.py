"""Run inside Blender to replay existing NeRF cameras on an edited scene.

blender --background --disable-autoexec SCENE.blend --python-exit-code 1 \
    --python scripts/capture_blender_views.py -- --output data/SCENE
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def check_png(path, resolution):
    with path.open('rb') as handle:
        header = handle.read(26)
    if (len(header) != 26 or header[:8] != b'\x89PNG\r\n\x1a\n'
            or struct.unpack('>II', header[16:24]) != (resolution, resolution)
            or header[24:26] != bytes([8, 6])):
        raise ValueError(f'Expected an {resolution}x{resolution} 8-bit RGBA PNG: {path}')


def select_device(requested):
    if requested == 'cpu':
        bpy.context.scene.cycles.device = 'CPU'
        return 'CPU'
    cache = ROOT / 'outputs/blender-optix-cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OPTIX_CACHE_PATH', str(cache))
    preferences = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX', 'CUDA'):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
        except (TypeError, RuntimeError):
            continue
        devices = [device for device in preferences.devices if device.type == backend]
        if devices:
            for device in preferences.devices:
                device.use = device.type == backend
            bpy.context.scene.cycles.device = 'GPU'
            return backend + ': ' + ', '.join(device.name for device in devices)
    raise RuntimeError('No Cycles OPTIX/CUDA device found; use --device cpu if needed.')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--templates', type=Path, default=ROOT / 'data/materials')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=800)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--limit-per-split', type=int, help='Small preview dataset only.')
    parser.add_argument('--resume', action='store_true')
    tokens = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    args = parser.parse_args(tokens)
    if (args.resolution < 1 or args.samples < 1 or args.seed < 0
            or (args.limit_per_split is not None and args.limit_per_split < 1)):
        parser.error('Resolution, samples and frame limit must be positive; seed must be nonnegative.')
    return args


def main():
    args = arguments()
    scene_file = Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath or not scene_file.is_file():
        raise ValueError('Load a saved .blend scene before running this script.')
    output, templates = args.output.resolve(), args.templates.resolve()
    if output == templates or output in templates.parents or templates in output.parents:
        raise ValueError('Output and template datasets must be separate directories.')
    manifests, template_hashes = {}, {}
    for split in ('train', 'val', 'test'):
        path = templates / f'transforms_{split}.json'
        manifest = json.loads(path.read_text(encoding='utf-8-sig'))
        if not 0 < float(manifest['camera_angle_x']) < math.pi:
            raise ValueError(f'Invalid horizontal field of view: {path}')
        frames = manifest['frames']
        if args.limit_per_split:
            frames = frames[:args.limit_per_split]
        if not frames:
            raise ValueError(f'No camera frames: {path}')
        for frame in frames:
            pose = frame['transform_matrix']
            if (len(pose) != 4 or any(len(row) != 4 for row in pose)
                    or not all(math.isfinite(float(v)) for row in pose for v in row)
                    or any(abs(float(pose[3][i]) - (1.0 if i == 3 else 0.0)) > 1e-6
                           for i in range(4))):
                raise ValueError(f'Invalid camera-to-world matrix: {path}')
        manifests[split] = dict(manifest, frames=frames)
        template_hashes[split] = fingerprint(path)

    scene = bpy.context.scene
    camera = scene.camera
    if camera is None or camera.type != 'CAMERA':
        raise ValueError('The scene must have an active camera.')
    device = select_device(args.device)
    specification = {
        'format_version': 1, 'script_sha256': fingerprint(Path(__file__)),
        'scene': str(scene_file), 'scene_sha256': fingerprint(scene_file),
        'templates': str(templates), 'template_sha256': template_hashes,
        'resolution': args.resolution, 'samples': args.samples, 'seed': args.seed,
        'device': device, 'blender_version': bpy.app.version_string,
        'limit_per_split': args.limit_per_split,
        'colour': {'transform': scene.view_settings.view_transform,
                   'look': scene.view_settings.look, 'exposure': scene.view_settings.exposure,
                   'gamma': scene.view_settings.gamma},
        'meshes': [obj.name for obj in scene.objects if obj.type == 'MESH' and not obj.hide_render],
        'camera_convention': 'Blender camera-to-world: +X right, +Y up, -Z forward',
        'image_format': '8-bit RGBA PNG; transparent film; scene colour management',
    }
    state_path = output / 'capture.json'
    if args.resume:
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['specification'] != specification:
            raise ValueError('Capture settings/source changed; use a new output directory.')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Output is not empty: {output}. Use --resume or a new directory.')
        output.mkdir(parents=True, exist_ok=True)
        state = {'specification': specification, 'images': {}, 'complete': False}
        write_json(state_path, state)

    # All changes are in memory. Never save over the user's Blender scene.
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.use_animated_seed = False
    scene.cycles.seed = args.seed
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.film_transparent = True
    scene.render.use_border = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_file_extension = True
    scene.render.use_persistent_data = True
    scene.render.dither_intensity = 0
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 15
    camera.parent = None
    camera.constraints.clear()
    camera.animation_data_clear()
    camera.data.animation_data_clear()
    camera.data.type = 'PERSP'
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.shift_x = camera.data.shift_y = 0
    camera.data.dof.use_dof = False

    for split, manifest in manifests.items():
        (output / split).mkdir(exist_ok=True)
        camera.data.lens = camera.data.sensor_width / (
            2 * math.tan(float(manifest['camera_angle_x']) / 2))
        if abs(camera.data.angle_x - float(manifest['camera_angle_x'])) > 1e-6:
            raise ValueError('Camera focal length does not reproduce the template field of view.')
        exported = dict(manifest, frames=[])
        for index, frame in enumerate(manifest['frames']):
            relative = f'{split}/r_{index}'
            image = output / f'{relative}.png'
            existing_hash = state['images'].get(relative)
            if existing_hash is not None:
                check_png(image, args.resolution)
                if fingerprint(image) != existing_hash:
                    raise ValueError(f'Captured image changed: {image}')
            else:
                if image.exists():
                    # A render may have finished immediately before interruption.
                    # It is deliberately re-rendered unless recorded in capture.json.
                    print(f'Re-rendering unrecorded frame: {image}', flush=True)
                camera.matrix_world = Matrix(frame['transform_matrix'])
                bpy.context.view_layer.update()
                actual = camera.matrix_world
                if max(abs(actual[r][c] - frame['transform_matrix'][r][c])
                       for r in range(4) for c in range(4)) > 1e-5:
                    raise ValueError('Camera drivers prevent replaying the requested pose.')
                scene.render.filepath = str(image)
                bpy.ops.render.render(write_still=True)
                check_png(image, args.resolution)
                state['images'][relative] = fingerprint(image)
                write_json(state_path, state)
            exported['frames'].append(dict(frame, file_path='./' + relative))
            print(f'CAPTURE {split} {index + 1}/{len(manifest["frames"])}: {image}', flush=True)
        # Only publish a split manifest when all of its images are ready.
        write_json(output / f'transforms_{split}.json', exported)
    state['complete'] = True
    write_json(state_path, state)
    print(f'Capture complete: {output}', flush=True)


if __name__ == '__main__':
    main()
