"""Trace ideal-mirror first bounces to distinguish environment and local geometry.

Run in background Blender with the scene loaded. Reference geometry is used
only for this diagnostic; no training/checkpoint files are changed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_blender_geometry import evaluated_geometry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--material', default='Chrome')
    parser.add_argument('--views', nargs='+', type=int, default=[0, 13, 73, 88])
    parser.add_argument('--max-per-view', type=int, default=50000)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output.exists():
        raise ValueError('Use a new diagnostic directory.')
    meta = json.loads((args.reference / 'geometry.json').read_text())
    scene = Path(bpy.data.filepath)
    if meta['scene_sha256'] != sha(scene) or meta['manifest_sha256'] != sha(args.manifest):
        raise ValueError('Reference scene/cameras differ from this audit.')
    manifest = json.loads(args.manifest.read_text())
    width, height = meta['resolution']
    focal = meta['focal']
    chrome = [obj for obj in meta['objects'] if args.material in obj['materials']]
    if not chrome or not set(args.views).issubset(meta['views']) or args.max_per_view < 1:
        raise ValueError('Missing material, views or valid sampling limit.')
    target_ids, target_names = [o['id'] for o in chrome], {o['name'] for o in chrome}
    # The shared BVH exporter filters camera-visible objects. For reflected rays,
    # substitute glossy visibility in memory only; never save the Blender file.
    visibility = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        visibility.append({'name': obj.name, 'hide_render': bool(obj.hide_render),
                           'visible_camera': bool(obj.visible_camera), 'visible_glossy': bool(obj.visible_glossy)})
        obj.hide_render = obj.hide_render or not obj.visible_glossy
        obj.visible_camera = True
    tree, vertices, faces, normals, object_ids, objects = evaluated_geometry()
    names = {obj['id']: obj['name'] for obj in objects}
    args.output.mkdir()
    rows, histograms = [], {}
    rng = np.random.default_rng(7)
    epsilon = 1e-4
    for view in args.views:
        frame = manifest['frames'][view]
        image = args.manifest.parent / (frame['file_path'].removeprefix('./') + '.png')
        if not image.exists():
            image = args.manifest.parent / frame['file_path']
        source = bpy.data.images.load(str(image.resolve()), check_existing=False)
        if tuple(source.size) != (width, height):
            raise ValueError('Audit requires images at the reference resolution.')
        pixels_rgba = np.empty(width * height * 4, np.float32)
        source.pixels.foreach_get(pixels_rgba)
        alpha = pixels_rgba.reshape(height, width, 4)[::-1, :, 3]
        bpy.data.images.remove(source)
        with np.load(args.reference / f'r_{view}.npz') as reference:
            z, n, ids = reference['depth'], reference['normal'], reference['object_id']
        target = np.isin(ids, target_ids) & (alpha >= .95)
        pixels = np.flatnonzero(target)
        total = len(pixels)
        if total > args.max_per_view:
            pixels = np.sort(rng.choice(pixels, args.max_per_view, replace=False))
        pose = Matrix(frame['transform_matrix'])
        rotation, origin = pose.to_3x3(), pose.translation
        labels = np.zeros((height, width), np.uint8)
        distances, hits = [], {}
        for pixel in pixels:
            y, x = divmod(int(pixel), width)
            point = origin + rotation @ Vector(((x + .5 - width / 2) * z[y, x] / focal,
                -(y + .5 - height / 2) * z[y, x] / focal, -z[y, x]))
            direction = (point - origin).normalized()
            normal = Vector(n[y, x]).normalized()
            reflected = (direction - 2 * direction.dot(normal) * normal).normalized()
            hit, hit_normal, index, distance = tree.ray_cast(point + epsilon * reflected, reflected)
            if index is None:
                labels[y, x] = 1
                continue
            name = names[object_ids[index]]
            hits[name] = hits.get(name, 0) + 1
            labels[y, x] = 2 if name in target_names else 3
            if name not in target_names:
                distances.append(float(distance))
        counts = {value: int(np.sum(labels == value)) for value in (1, 2, 3)}
        assert sum(counts.values()) == len(pixels)
        row = {'view': view, 'visible_material_pixels': total, 'sampled_pixels': len(pixels),
            'environment_fraction': counts[1] / len(pixels) if len(pixels) else None,
            'self_geometry_fraction': counts[2] / len(pixels) if len(pixels) else None,
            'other_geometry_fraction': counts[3] / len(pixels) if len(pixels) else None,
            'other_hit_median_scene_distance': float(np.median(distances)) if distances else None}
        rows.append(row)
        histograms[str(view)] = hits
        np.savez_compressed(args.output / f'r_{view}.npz', labels=labels)
        print(json.dumps(row), flush=True)
    with (args.output / 'scope.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'status': 'completed', 'material': args.material, 'views': args.views,
        'scene_sha256': sha(scene), 'manifest_sha256': sha(args.manifest),
        'reference_metadata_sha256': sha(args.reference / 'geometry.json'),
        'script_sha256': sha(Path(__file__)), 'helper_sha256': sha(Path(__file__).with_name('export_blender_geometry.py')),
        'visibility': visibility, 'hit_objects': histograms, 'rows': rows, 'ray_origin_epsilon': epsilon,
        'labels': {'0': 'outside material or unsampled', '1': 'environment escape', '2': 'same material geometry', '3': 'other geometry'},
        'scope': 'Ideal mirror first bounce using interpolated mesh normals; no shader bump/roughness/energy weighting or multibounce transport. Diagnostic only, never training input.'}
    (args.output / 'scope.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
