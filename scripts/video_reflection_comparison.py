"""Encode reference/model test views from a completed all-renders comparison.

This is a visual, 8-bit presentation artifact. Scientific metrics remain the
float-render scores in the comparison report. Frames retain manifest order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--models', type=int, nargs='+', help='Zero-based report model columns; default keeps all.')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--output', type=Path, help='New MP4 file; default report/comparison.mp4.')
    args = parser.parse_args()
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError('Install the optional encoder with: pip install imageio-ffmpeg') from error
    protocol = json.loads((args.report / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['status'] != 'completed' or not protocol.get('save_all_renders'):
        raise ValueError('Use a completed report produced with --save-all-renders.')
    chosen = args.models if args.models is not None else list(range(len(protocol['runs'])))
    if not chosen or len(set(chosen)) != len(chosen) or any(i < 0 or i >= len(protocol['runs']) for i in chosen):
        raise ValueError('Choose unique valid model columns.')
    if not 1 <= args.fps <= 60:
        raise ValueError('Use 1–60 FPS.')
    config = yaml.safe_load((Path(protocol['runs'][0]) / 'config.yaml').read_text(encoding='utf-8'))
    data = Path(config['paths']['data'])
    manifest_path = data / protocol['manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    frames = manifest['frames']
    width, height = protocol['resolution']
    for index in chosen:
        for view in range(len(frames)):
            path = args.report / f'model-{index}-{view}.png'
            if not path.is_file():
                raise ValueError('Missing render: ' + str(path))
    output = args.output or args.report / 'comparison.mp4'
    if output.suffix.lower() != '.mp4':
        raise ValueError('Use an MP4 output filename.')
    if output.exists():
        raise ValueError('Preserve existing videos; choose a new output filename.')
    total_width, total_height = width * (1 + len(chosen)), height + 50
    total_width += total_width % 2
    total_height += total_height % 2
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-n', '-nostdin', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{total_width}x{total_height}',
        '-r', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2', str(output)]
    started = perf_counter()
    with output.with_suffix('.encode.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            for view, frame in enumerate(frames):
                path = data / frame['file_path']
                if not path.suffix:
                    path = path.with_suffix('.png')
                with Image.open(path) as source:
                    rgba = source.convert('RGBA').resize((width, height), Image.Resampling.LANCZOS)
                reference = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
                canvas = Image.new('RGB', (total_width, total_height), 'white')
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 10), 'Reference', fill='#222222', font=font)
                canvas.paste(reference, (0, 50))
                for column, index in enumerate(chosen, 1):
                    with Image.open(args.report / f'model-{index}-{view}.png') as source:
                        canvas.paste(source.convert('RGB'), (column * width, 50))
                    draw.text((column * width + 12, 10), protocol['labels'][index], fill='#222222', font=font)
                process.stdin.write(canvas.tobytes())
                if (view + 1) % 20 == 0:
                    print(f'Video [{view + 1}/{len(frames)}]', flush=True)
            process.stdin.close()
            if process.wait(timeout=60):
                raise RuntimeError('FFmpeg failed; see the encoding log.')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    result = {'report': str(args.report.resolve()), 'models': chosen, 'labels': [protocol['labels'][i] for i in chosen],
        'frames': len(frames), 'fps': args.fps, 'frame_order': 'manifest order', 'resolution': [total_width, total_height],
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'report_protocol_sha256': hashlib.sha256((args.report / 'protocol.json').read_bytes()).hexdigest(),
        'command': command, 'seconds': perf_counter() - started, 'bytes': output.stat().st_size,
        'scope': 'Quantized visual artifact; use report float scores for quantitative comparisons.'}
    output.with_suffix('.video.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(output, flush=True)


if __name__ == '__main__':
    main()
