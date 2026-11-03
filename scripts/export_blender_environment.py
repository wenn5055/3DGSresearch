"""Export the packed world environment as scene-linear, bottom-up pixels; never save the scene."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    scene = bpy.context.scene
    tree = scene.world.node_tree
    output = next(n for n in tree.nodes if n.type == 'OUTPUT_WORLD' and n.is_active_output)
    background = output.inputs['Surface'].links[0].from_node
    if background.type != 'BACKGROUND' or background.inputs['Strength'].is_linked:
        raise ValueError('This diagnostic expects a constant-strength Background world.')
    texture = background.inputs['Color'].links[0].from_node
    if texture.type != 'TEX_ENVIRONMENT' or texture.projection != 'EQUIRECTANGULAR' or texture.inputs['Vector'].is_linked:
        raise ValueError('Expected an unmapped equirectangular environment image directly linked to Background.')
    mapping = texture.texture_mapping
    if (any(abs(v) > 1e-8 for v in mapping.rotation) or any(abs(v) > 1e-8 for v in mapping.translation)
            or any(abs(v-1) > 1e-8 for v in mapping.scale)):
        raise ValueError('Nonidentity texture mapping is not supported by this diagnostic.')
    image = texture.image
    width, height = image.size
    pixels = np.empty(width*height*4, np.float32)
    image.pixels.foreach_get(pixels)
    rgb = pixels.reshape(height, width, 4)[:, :, :3].copy()
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'environment.npz', rgb_bottom_up=rgb)
    metadata = {'scene': str(Path(bpy.data.filepath).resolve()),
                'scene_sha256': hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
                'image': image.name, 'resolution': [width, height], 'color_space': image.colorspace_settings.name,
                'packed': bool(image.packed_file), 'pixel_order': 'bottom-up, scene-linear float RGB',
                'world_strength': float(background.inputs['Strength'].default_value),
                'texture_interpolation': texture.interpolation, 'blender_version': bpy.app.version_string,
                'usage': 'evaluation only; reference lighting must not enter training',
                'pixel_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
    (args.output/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Exported:', args.output, flush=True)


if __name__ == '__main__':
    main()
