"""Export diagnostic ray-hit geometry from a saved Blender scene and NeRF poses.

Run inside background Blender. This reads the scene without changing or saving it.
Normals are interpolated evaluated-mesh normals, excluding shader bump maps.
The reference is for evaluation only, never a training target in the SH study.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluated_geometry():
    graph = bpy.context.evaluated_depsgraph_get()
    vertices, faces, normals, object_ids, objects = [], [], [], [], []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or obj.hide_render or not obj.visible_camera:
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            normal_matrix = evaluated.matrix_world.to_3x3().inverted().transposed()
            offset, object_id = len(vertices), len(objects) + 1
            objects.append({'id': object_id, 'name': obj.name,
                'materials': [slot.material.name if slot.material else None for slot in obj.material_slots]})
            vertices.extend([evaluated.matrix_world @ vertex.co for vertex in mesh.vertices])
            for triangle in mesh.loop_triangles:
                faces.append(tuple(offset + index for index in triangle.vertices))
                if mesh.polygons[triangle.polygon_index].use_smooth:
                    corner_normals = [mesh.corner_normals[index].vector for index in triangle.loops]
                else:
                    corner_normals = [triangle.normal] * 3
                normals.append(tuple((normal_matrix @ normal).normalized() for normal in corner_normals))
                object_ids.append(object_id)
        finally:
            evaluated.to_mesh_clear()
    if not faces:
        raise ValueError('No visible mesh triangles found.')
    return BVHTree.FromPolygons(vertices, faces, all_triangles=True), vertices, faces, normals, object_ids, objects


def barycentric(point, a, b, c):
    e1, e2, delta = b - a, c - a, point - a
    aa, ab, bb = e1.dot(e1), e1.dot(e2), e2.dot(e2)
    determinant = aa * bb - ab * ab
    if abs(determinant) < 1e-20:
        return 1., 0., 0.
    x = (bb * delta.dot(e1) - ab * delta.dot(e2)) / determinant
    y = (aa * delta.dot(e2) - ab * delta.dot(e1)) / determinant
    return 1. - x - y, x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--views', type=int, nargs='+', default=[0, 13, 21, 25])
    parser.add_argument('--all-views', action='store_true')
    parser.add_argument('--resolution', type=int, default=400)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if args.resolution < 1:
        parser.error('Resolution must be positive.')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty geometry output directory.')
    scene_path = Path(bpy.data.filepath)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.all_views:
        args.views = list(range(len(manifest['frames'])))
    if any(view < 0 or view >= len(manifest['frames']) for view in args.views):
        raise ValueError('View index outside the requested manifest.')
    args.output.mkdir(parents=True, exist_ok=True)
    tree, vertices, faces, normals, object_ids, objects = evaluated_geometry()
    size = args.resolution
    focal = .5 * size / math.tan(.5 * manifest['camera_angle_x'])
    metadata = {'status': 'running', 'scene': str(scene_path.resolve()), 'scene_sha256': sha(scene_path),
        'manifest': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
        'script_sha256': sha(Path(__file__)), 'blender_version': bpy.app.version_string,
        'resolution': [size, size], 'focal': focal, 'views': args.views, 'objects': objects,
        'triangles': len(faces), 'normal_convention': 'unit world-space, camera-facing interpolated evaluated-mesh corner normals; excludes shader bump',
        'depth_convention': 'positive camera-z distance, not Euclidean ray distance',
        'pixel_convention': 'top-left origin, half-integer centers, Blender c2w +X right +Y up -Z forward',
        'usage': 'diagnostic evaluation only; no reference geometry used by SH training'}
    metadata_path = args.output / 'geometry.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for view in args.views:
        started = perf_counter()
        camera = Matrix(manifest['frames'][view]['transform_matrix'])
        rotation, inverse, origin = camera.to_3x3(), camera.inverted(), camera.translation
        reference_normal = np.zeros((size, size, 3), np.float32)
        reference_depth = np.zeros((size, size), np.float32)
        reference_object = np.zeros((size, size), np.int16)
        for y in range(size):
            for x in range(size):
                ray = (rotation @ Vector(((x + .5 - size / 2) / focal, -(y + .5 - size / 2) / focal, -1.))).normalized()
                point, face_normal, index, distance = tree.ray_cast(origin, ray)
                if index is None:
                    continue
                a, b, c = [vertices[item] for item in faces[index]]
                wa, wb, wc = barycentric(point, a, b, c)
                na, nb, nc = normals[index]
                normal = (wa * na + wb * nb + wc * nc).normalized()
                if normal.dot(ray) > 0:
                    normal.negate()
                reference_normal[y, x] = normal
                reference_depth[y, x] = -(inverse @ point).z
                reference_object[y, x] = object_ids[index]
        np.savez_compressed(args.output / f'r_{view}.npz', normal=reference_normal,
                            depth=reference_depth, object_id=reference_object)
        print(f'View {view}: {np.count_nonzero(reference_object)} hit pixels, {perf_counter() - started:.2f}s', flush=True)
    metadata['status'] = 'completed'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Geometry reference complete:', args.output, flush=True)


if __name__ == '__main__':
    main()
