"""Read effective object materials without rendering or saving the Blender scene.

blender --background --disable-autoexec SCENE.blend --python-exit-code 1 \
    --python scripts/inspect_blender_materials.py -- --output outputs/SCENE-materials.json

Object-linked material slots override mesh datablock materials; inspect the former.
Node socket values are defaults, not evaluations of linked procedural shaders.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def socket_value(socket):
    if not hasattr(socket, 'default_value'):
        return None
    value = socket.default_value
    if isinstance(value, (float, int, str, bool)):
        return value
    try:
        return list(value)
    except TypeError:
        return None


def inspect_material(material, seen=None):
    if material is None:
        return None
    tree = material.node_tree
    return {'name': material.name, 'node_tree': inspect_tree(tree, set() if seen is None else seen)}


def inspect_tree(tree, seen):
    if tree is None or tree.name in seen:
        return None
    seen.add(tree.name)
    nodes = []
    for node in tree.nodes:
        record = {'name': node.name, 'type': node.type,
                  'inputs': {socket.name: {'default': socket_value(socket), 'linked': socket.is_linked}
                             for socket in node.inputs}}
        if node.type == 'TEX_ENVIRONMENT' and node.image:
            record['image'] = node.image.name
        if node.type == 'GROUP':
            record['group'] = inspect_tree(node.node_tree, seen)
        nodes.append(record)
    return {'name': tree.name, 'nodes': nodes,
            'links': [[link.from_node.name, link.from_socket.name, link.to_node.name, link.to_socket.name]
                      for link in tree.links]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    scene_path = Path(bpy.data.filepath)
    if not scene_path.is_file():
        raise ValueError('Open a saved scene first.')
    result = {'scene': str(scene_path.resolve()), 'scene_sha256': hashlib.sha256(scene_path.read_bytes()).hexdigest(),
              'blender_version': bpy.app.version_string,
              'world': inspect_tree(bpy.context.scene.world.node_tree, set()), 'objects': []}
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        bounds = [list(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
        result['objects'].append({'name': obj.name, 'location': list(obj.location),
            'world_bounds': bounds, 'dimensions': list(obj.dimensions),
            'materials': [inspect_material(slot.material) for slot in obj.material_slots],
            'modifiers': [{'name': item.name, 'type': item.type} for item in obj.modifiers]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Saved material inspection: {args.output}', flush=True)


if __name__ == '__main__':
    main()
