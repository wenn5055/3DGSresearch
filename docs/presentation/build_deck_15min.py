"""Build the 15-minute research talk and seven visual Q&A slides.

Creates editable PowerPoint, offline HTML, PNG previews and exact speaker notes.
Uses existing evidence only. No model evaluation, external uploads or PDF build.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from playwright.sync_api import sync_playwright

import build_deck as drawing

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/presentation-15min'
STEM = 'reconstructing-reflections-15min'
C = drawing.C
W, H = 1600, 900
DURATIONS = [45, 70, 75, 100, 90, 80, 60, 120, 60, 80, 60, 60]
SCRIPT = 'docs/presentation/speaker-script-15min.md'
APPENDIX = 'docs/presentation/qna-visual-appendix.md'
MAIN_NOTES = {}
BACKUP_NOTES = {}
SLIDES = drawing.SLIDES
T, R, P = drawing.text, drawing.rect, drawing.picture


def source(relative):
    return drawing.source(relative)


def asset(name, relative, box=None, white=False):
    return drawing.crop(name, relative, box, white)


def clock(seconds):
    return f'{seconds // 60}:{seconds % 60:02d}'


def read_scripts():
    script = source(SCRIPT).read_text(encoding='utf-8')
    for number, title, body in re.findall(r'^## Slide (\d+) — ([^\n]+)\n(.*?)(?=^## |\Z)', script, re.M | re.S):
        MAIN_NOTES[int(number)] = body.split('**Say**', 1)[1].strip()
    appendix = source(APPENDIX).read_text(encoding='utf-8')
    for number, title, body in re.findall(r'^## A(\d+) — ([^\n]+)\n(.*?)(?=^## |\Z)', appendix, re.M | re.S):
        body = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', body)
        body = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', body).replace('**', '')
        BACKUP_NOTES[int(number)] = body.strip()
    assert len(MAIN_NOTES) == 12 and len(BACKUP_NOTES) == 7
    assert sum(DURATIONS) == 900


def slide(title, section, citation, *, dark=False, backup=False):
    number = len(SLIDES) + 1
    seconds = 0 if backup else DURATIONS[number - 1]
    notes = BACKUP_NOTES[number - 12] if backup else MAIN_NOTES[number]
    s = drawing.new(title, section, seconds / 60, notes, dark=dark, backup=backup)
    s['citation'] = citation
    s['seconds'] = seconds
    s['id'] = f'A{number - 12}' if backup else str(number)
    return s


def header(s, eyebrow, title, subtitle=None, size=58, height=130):
    T(s, 72, 36, 1456, 32, eyebrow.upper(), 19, C['lime'] if s['dark'] else C['green'], True)
    T(s, 72, 88, 1456, height, title, size, font='Georgia')
    if subtitle:
        T(s, 75, 218, 1450, 62, subtitle, 25, '#C3D1C3' if s['dark'] else C['muted'])


def footer(s, number):
    drawing.line(s, 72, 833, 1456, '#46605A' if s['dark'] else C['line'])
    T(s, 72, 850, 1335, 34, s['citation'], 17, '#C3D1C3' if s['dark'] else C['muted'])
    T(s, 1420, 849, 108, 34, s['id'] if s['backup'] else f'{number:02d} / 12',
      20, C['lime'] if s['dark'] else C['green'], True, align='right')


def column_images(s, images, labels, y=300, height=380, x=72, width=1456, label_size=24):
    gap = 26
    cell = (width - gap * (len(images) - 1)) / len(images)
    for i, (image, label) in enumerate(zip(images, labels)):
        xx = x + i * (cell + gap)
        T(s, xx, y - 44, cell, 44, label, label_size,
          C['lime'] if s['dark'] else C['green'], True)
        P(s, image, xx, y, cell, height)


def card(s, x, y, width, title, value, detail, *, fill=None, value_size=52):
    R(s, x, y, width, 130, fill or C['white'])
    T(s, x + 20, y + 12, width - 40, 30, title, 21, C['muted'], True)
    T(s, x + 20, y + 45, width - 40, 63, value, value_size, C['green'], True)
    if detail:
        T(s, x + 20, y + 108, width - 40, 27, detail, 19, C['muted'])


def build_content():
    # Complete 400px display panels from the archived original 800px test renders.
    full = 'docs/reports/assets/2026-11-12-full-views/source-test-rgb.png'
    test = {}
    for row, view in enumerate([0, 13, 73, 88]):
        for col, model in enumerate(['reference', 'SH3', 'HDR-1.0']):
            x, y = 12 + col * 412, 68 + row * 442
            test[view, model] = asset(f'test-{view}-{model}-full', full, (x, y, x + 400, y + 400))
    crop_sheet = 'docs/reports/assets/2026-11-03-sphere-study/test-reflection-crops.png'
    crops = [asset(f'test-0-{name}-crop', crop_sheet, (12 + col * 312, 68, 312 + col * 312, 368))
             for col, name in enumerate(['target', 'SH3', 'HDR-1.0', 'HDR-0.75'])]
    hero = asset('hero-reference', 'data/justChromeBall/val/r_0.png', (160, 160, 640, 640), True)
    materials = 'docs/reports/assets/2026-11-11-report-revision/materials-initial-exploration.png'
    material_images = [asset(f'materials-{i}', materials, (i * 360, 46, (i + 1) * 360, 406)) for i in range(3)]
    stage = 'docs/reports/assets/2026-11-02-sphere/reflection-crops.png'
    stages = [[asset(f'stage-{row}-{col}', stage, (col * 360, 46 + row * 402, (col + 1) * 360, 406 + row * 402))
               for col in range(3)] for row in range(2)]
    failure = asset('refinement-failure', 'docs/reports/assets/2026-11-02-sphere/regression-view-51.png')
    normal = asset('hdr1-normal', 'docs/reports/assets/2026-11-03-sphere-study/normal-comparison.png', (836, 68, 1236, 468))
    env = asset('hdr1-environment', 'docs/reports/assets/2026-11-03-sphere-study/learned-environments.png', (12, 168, 412, 368))
    swap = 'docs/reports/assets/2026-11-03-reference-normal-swap/reflection-crops.png'
    swaps = {}
    for row, view in [(0, 0), (4, 51)]:
        swaps[view] = [asset(f'swap-{view}-{col}', swap, (12 + col * 312, 68 + row * 342,
                                                       312 + col * 312, 368 + row * 342)) for col in range(5)]
    weights = 'docs/reports/assets/2026-11-07-normal-weight-closeout/reflection-crops.png'
    weight_images = [asset(f'weight-crop-{col}', weights, (col * 360, 42, (col + 1) * 360, 402)) for col in range(3)]
    loss = 'docs/reports/assets/2026-11-07-refinement-loss/reflection-highlight-view-0.png'
    loss_images = [asset(f'loss-crop-{col}', loss, (col * 360, 42, (col + 1) * 360, 402)) for col in range(3)]

    s = slide('Reconstructing reflections', 'Opening',
              'Actual Blender target · sphere · validation view 0 | CP4282 research presentation · 12 November 2026', dark=True)
    T(s, 72, 52, 800, 38, 'CP4282 / 15-MINUTE RESEARCH TALK', 21, C['lime'], True)
    T(s, 72, 152, 840, 260, 'Reconstructing\nreflections.', 100, font='Georgia')
    T(s, 77, 445, 780, 104, 'Deferred Gaussian surfels\nfor a chrome sphere', 38)
    T(s, 77, 573, 790, 62, 'Tang Xiao Di Wendy', 29, C['lime'], True)
    R(s, 77, 672, 780, 118, C['green'])
    T(s, 98, 692, 737, 88, 'Can explicit reflection shading\nimprove on SH3?', 34, C['white'], True)
    R(s, 932, 153, 596, 596, C['white'])
    P(s, hero, 947, 168, 566, 566)

    s = slide('Materials motivated a focused experiment', 'Initial exploration',
              'Original Materials · test view 13 · early RGB/SH3, staged MSE/D-SSIM | separate from the sphere study')
    header(s, '01 / Initial exploration', 'Materials exposed several problems at once.')
    column_images(s, material_images, ['BLENDER TARGET', 'EARLY FIXED RGB', 'EARLY SH3'], y=276, height=330)
    R(s, 72, 653, 704, 139, C['white']); R(s, 800, 653, 728, 139, C['lime'])
    T(s, 94, 670, 660, 98, 'Nearby-object reflections\n+ hidden recessed geometry', 32, bold=True)
    T(s, 822, 670, 680, 98, 'Focus the main question:\none isolated, smooth sphere', 32, C['green'], True)
    T(s, 72, 608, 1456, 34, 'Early test PSNR: fixed RGB 24.94 dB → SH3 26.07 dB', 23, C['muted'])

    s = slide('SH3 and explicit reflection shading', 'Appearance models',
              'Examples: sphere · test view 0 · SH3 / HDR-1.0 reflection crops | course Unit 11; report §2')
    header(s, '02 / Appearance', 'Two ways to predict view-dependent colour.')
    for x, title, lines, picture in [(72, 'SH3', 'View direction → colour\nDirectional coefficients per surfel', crops[1]),
                                     (820, 'REFLECTION MODEL', 'View + normal → reflected direction\nSample shared surroundings', crops[2])]:
        R(s, x, 273, 708, 471, C['white'])
        T(s, x + 25, 295, 655, 48, title, 30, C['green'], True)
        T(s, x + 25, 360, 655, 86, lines, 29)
        P(s, picture, x + 25, 469, 250, 250)
        T(s, x + 304, 493, 375, 173,
          'Limited angular detail' if x == 72 else 'A common source\nof reflected detail', 34, C['muted'])
    T(s, 74, 775, 1450, 38, 'The comparison changes the appearance approach; both main methods use surfels.', 25, C['green'])

    s = slide('Blend attributes, then compute reflections', 'Method',
              'Schematic: implemented renderer | actual HDR-1.0 normal/environment maps · validation 0 | Gaussian Surfels; 3DGS-DR')
    header(s, '03 / Method', 'Blend the surface. Then compute its reflection.',
           'Implemented in the course’s Warp framework; the environment is learned from the images.')
    for x, n, title, body in [(72, '01', 'Surfels', 'Position, orientation,\ncolour, reflection strength'),
                               (447, '02', 'Blend attributes', 'Normal, depth,\ncolour and coverage'),
                               (822, '03', 'Reflect the ray', 'Camera ray +\ncombined surface normal'),
                               (1197, '04', 'Sample & shade', 'Learned HDR map\n+ base colour')]:
        R(s, x, 323, 330, 235, C['white'])
        T(s, x + 20, 340, 290, 35, n, 25, C['green'], True)
        T(s, x + 20, 395, 290, 45, title, 31, bold=True)
        T(s, x + 20, 466, 290, 79, body, 24, C['muted'])
        if x != 1197:
            T(s, x + 334, 414, 39, 48, '→', 35, C['green'])
    P(s, normal, 75, 608, 176, 176)
    T(s, 277, 634, 345, 116, 'Learned surface normal\nDirection encoded as RGB', 26, C['muted'])
    P(s, env, 690, 604, 360, 180)
    T(s, 1074, 634, 452, 116, 'Learned surroundings\n512 × 256 HDR map', 27, C['muted'])

    s = slide('A shared surface and training schedule', 'Training and comparison',
              'Same 100k surfels · seed 7 · 800 × 800 · 100 train / 100 validation / 200 test | RGB + alpha training only')
    header(s, '04 / Training', 'Same starting surface. Same training budget.')
    R(s, 72, 275, 938, 140, C['ink']); R(s, 1034, 275, 494, 140, C['lime'])
    T(s, 97, 289, 880, 62, '10,000 steps', 47, C['lime'], True)
    T(s, 99, 361, 875, 43, 'Fit appearance; geometry and opacity fixed', 30, C['white'])
    T(s, 1059, 289, 445, 62, '5,000 steps', 47, C['green'], True)
    T(s, 1059, 361, 445, 43, 'Joint refinement', 30)
    for x, title, body in [(72, 'RGB matching', 'MSE for both methods'),
                            (565, 'Mask constraint', 'Protect foreground coverage'),
                            (1058, 'Normal–depth consistency', 'Compare two model predictions')]:
        drawing.line(s, x, 474, 470)
        T(s, x, 496, 470, 82, title, 31, bold=True)
        T(s, x, 598, 470, 84, body, 29, C['muted'])
    R(s, 72, 727, 1456, 72, C['soft'])
    T(s, 95, 743, 1408, 45, 'No reference normals or environment map supply fitting gradients.', 31, C['green'], True)

    s = slide('Reflections improve every test view', 'Main result',
              'Sphere · test view 0 full frames | means: all 200 test views · SH3 vs HDR-1.0 · same 100k / 15k budget')
    header(s, '05 / Main result', 'The reflection model improves every test view.')
    column_images(s, [test[0, m] for m in ['reference', 'SH3', 'HDR-1.0']],
                  ['BLENDER REFERENCE', 'SH3', 'HDR-1.0'], y=275, height=400)
    card(s, 72, 690, 470, 'PSNR ↑', '+4.31 dB', '20.27 → 24.57 dB')
    card(s, 565, 690, 470, 'SSIM ↑', '0.821 → 0.881', '', value_size=43)
    card(s, 1058, 690, 470, 'LPIPS ↓', '35.1% lower', 'All three improve on 200 / 200 views', fill=C['lime'], value_size=44)

    s = slide('Refinement improves visible structure', 'Training stages',
              'Sphere · validation 0 / 13; failure 51 · HDR-1.0 | 10k frozen vs 15k total includes extra steps + geometry release')
    header(s, '06 / Refinement', 'Refinement helps. Fine details still fail.')
    for col, title in enumerate(['TARGET', 'FROZEN · 10k', 'REFINED · 15k TOTAL']):
        T(s, 72 + col * 340, 234, 318, 40, title, 23, C['green'], True)
    for row, view in enumerate([0, 13]):
        y = 285 + row * 269
        T(s, 72, y - 20, 1000, 29, f'Validation view {view}', 18, C['muted'])
        for col, im in enumerate(stages[row]):
            P(s, im, 72 + col * 340, y + 12, 312, 232)
    R(s, 1130, 274, 398, 220, C['white'])
    T(s, 1148, 289, 362, 40, 'PRESERVED FAILURE · VIEW 51', 20, C['orange'], True)
    P(s, failure, 1148, 343, 362, 136)
    T(s, 1148, 515, 362, 83, 'PSNR\n22.50 → 24.13 dB', 30, C['green'], True)
    T(s, 1148, 638, 362, 88, 'LPIPS\n0.1198 → 0.0940', 30, C['green'], True)
    T(s, 1148, 752, 362, 49, 'Validation means · 100 views', 22, C['muted'])

    s = slide('Lighting and normal errors interact', 'Diagnostic substitutions',
              'Sphere · validation 0 shown; five-view diagnostic · same HDR-0.75 checkpoint | reference inputs used only for evaluation')
    header(s, '07 / Diagnostic', 'Lighting and normal errors interact.',
           'Same saved model: replace lighting, normals, or both. All four combinations are shown.')
    T(s, 73, 326, 350, 40, 'BLENDER TARGET', 24, C['green'], True)
    P(s, swaps[0][0], 72, 381, 330, 330)
    T(s, 73, 744, 390, 68, 'Best structure ≠\nhighest PSNR', 28, C['green'], True)
    for x, label in [(650, 'LEARNED NORMALS'), (1080, 'REFERENCE NORMALS')]:
        T(s, x, 250, 350, 35, label, 23, C['green'], True)
    for yy, label in [(329, 'LEARNED\nLIGHTING'), (616, 'REFERENCE\nLIGHTING')]:
        T(s, 459, yy, 176, 82, label, 23, C['muted'], True)
    for index, x, y in [(1, 650, 293), (3, 1080, 293), (2, 650, 562), (4, 1080, 562)]:
        P(s, swaps[0][index], x, y, 250, 250)

    s = slide('Trial 1: increase normal–depth consistency', 'Loss trial 1',
              'Sphere · validation 0 shown | image scores: 100 views; normal error: five views | same HDR-1.0 parent + 5k refinement')
    header(s, '08 / Trial 1', 'Stronger consistency does not improve accuracy.')
    R(s, 72, 277, 433, 360, C['white'])
    T(s, 96, 299, 385, 74, 'NORMAL–DEPTH\nWEIGHT', 25, C['muted'], True)
    T(s, 96, 412, 385, 76, '0.01 → 0.03', 50, C['green'], True)
    T(s, 96, 526, 385, 93, 'Everything else held fixed\nwithin this trial', 28, C['muted'])
    column_images(s, weight_images, ['TARGET', 'WEIGHT 0.01', 'WEIGHT 0.03'], y=329, height=300,
                  x=563, width=965, label_size=23)
    R(s, 72, 690, 1456, 111, C['soft'])
    T(s, 96, 707, 1408, 83, 'Reference-normal error: 1.573° → 1.594°\nInternal agreement is not independent accuracy.', 31, C['green'], True)

    s = slide('Trial 2: L1 + D-SSIM', 'Loss trial 2',
              'Sphere · validation 0 crops | 100-view image means; five-view normal checks | same HDR-1.0 parent, 5k refinement')
    header(s, '09 / Trial 2', 'A better LPIPS score, at a PSNR cost.',
           'Refinement image objective: MSE → 0.8 L1 + 0.2 D-SSIM')
    column_images(s, loss_images, ['TARGET', 'MSE REFINEMENT', 'L1 + D-SSIM REFINEMENT'], y=331, height=328)
    card(s, 72, 690, 470, 'LPIPS ↓', '10.1% lower', 'Improves on 100 / 100 validation views', fill=C['lime'], value_size=44)
    card(s, 565, 690, 470, 'PSNR', '−0.56 dB', 'Lower pixel accuracy', value_size=48)
    card(s, 1058, 690, 470, 'NORMAL ACCURACY', 'Unchanged', 'Missing thin reflections remain', value_size=42)

    s = slide('Scope and next experiments', 'Interpretation',
              'One scene and one seed; cameras are not independent training repetitions | report §§7–8')
    header(s, '10 / What the evidence establishes', 'A focused result, with specific next questions.')
    for x, n, title, body in [(72, '01', 'One fitted model pair', '200 camera views show a consistent gain.\nThey do not replace repeated seeds.'),
                               (820, '02', 'A combined appearance change', 'The treatment improves reconstruction.\nIts components are not isolated here.')]:
        R(s, x, 294, 708, 287, C['white'])
        T(s, x + 25, 314, 655, 50, n, 39, C['green'], True)
        T(s, x + 25, 390, 655, 72, title, 33, bold=True)
        T(s, x + 25, 488, 655, 83, body, 27, C['muted'])
    R(s, 72, 644, 1456, 153, C['lime'])
    T(s, 96, 658, 1408, 35, 'NEXT EXPERIMENTS', 22, C['green'], True)
    T(s, 96, 713, 1408, 75, 'Separate the image-loss components.\nRepeat with another seed and another controlled reflective scene.', 33)

    s = slide('Reconstruction quality needs more than one score', 'Conclusion',
              'Main evidence: test metrics + reflection images + independent geometry checks | Questions', dark=True)
    header(s, '11 / Conclusion', 'Better reflection images need\nmore than one score.', size=70, height=178)
    for y, n, body in [(307, '01', 'Explicit reflections improve the SH3 comparison.'),
                        (415, '02', 'Lighting and normal errors jointly limit detail.'),
                        (523, '03', 'Better image scores do not establish correct geometry.')]:
        T(s, 73, y, 90, 64, n, 42, C['lime'], True)
        T(s, 188, y + 3, 1320, 82, body, 37)
    T(s, 76, 653, 1450, 48, 'Questions', 36, C['lime'], True)
    links = [('Comparisons', 12), ('RGB error', 13), ('Normals', 14), ('Depth', 15),
             ('Lighting', 16), ('Losses', 17), ('Scope', 18)]
    for i, (label, target) in enumerate(links):
        x = 76 + i * 208
        R(s, x, 735, 194, 57, C['green'])
        T(s, x + 12, 748, 170, 34, label, 23, C['white'], True)
        s['objects'][-1]['target'] = target

    s = slide('A1: More full-view comparisons', 'Q&A / comparisons',
              'Sphere · test views 13 and 88 · reference / SH3 / HDR-1.0 | same framing and 100k / 15k budget', backup=True)
    header(s, 'Q&A / A1', 'More cameras, the same comparison.')
    for col, title in enumerate(['REFERENCE', 'SH3', 'HDR-1.0']):
        T(s, 72 + col * 493, 232, 470, 34, title, 24, C['green'], True)
    for row, view in enumerate([13, 88]):
        y = 303 + row * 268
        T(s, 73, y - 26, 1450, 32, f'Test view {view}', 21, C['muted'])
        for col, model in enumerate(['reference', 'SH3', 'HDR-1.0']):
            P(s, test[view, model], 72 + col * 493, y, 470, 240)

    s = slide('A2: RGB differences', 'Q&A / image error',
              'Sphere · validation 0 / 51 · mean absolute RGB error [0,1] · shared scale 0–0.25, clipped | saved PNGs', backup=True)
    header(s, 'Q&A / A2', 'Where do the image predictions disagree?',
           'The same absolute-error scale is used for SH3 and HDR-1.0. These are visual PNG differences.')
    image = asset('qna-rgb-errors', 'docs/presentation/assets/2026-11-12-qna/rgb-differences.png')
    P(s, image, 72, 286, 1456, 520)

    s = slide('A3: Normal maps and angular error', 'Q&A / normals',
              'Sphere · validation 0 / 51 · reference / SH3 / HDR-1.0 | error 0–5°, clipped; white excluded | not five-view means', backup=True)
    header(s, 'Q&A / A3', 'Similar normal colours can hide local errors.',
           'World-space XYZ encoded as RGB; the error map compares HDR-1.0 with the Blender mesh.')
    image = asset('qna-normal-errors', 'docs/presentation/assets/2026-11-12-qna/normal-comparison-errors.png')
    P(s, image, 72, 287, 1120, 515)
    T(s, 1220, 349, 304, 70, '1.61°', 64, C['green'], True)
    T(s, 1225, 432, 298, 55, 'View 0 mean error', 25, C['muted'])
    T(s, 1220, 554, 304, 70, '1.54°', 64, C['green'], True)
    T(s, 1225, 637, 298, 75, 'View 51 mean error\nHDR-1.0', 25, C['muted'])

    s = slide('A4: Depth and absolute depth error', 'Q&A / depth',
              'Sphere · validation 0 / 51 · positive camera-z depth | same range across both views / all models; white excluded', backup=True)
    header(s, 'Q&A / A4', 'Depth is another check on the surface.',
           'Shared scene-unit depth scale; absolute depth error is displayed from 0 to 0.02, clipped.')
    image = asset('qna-depth-errors', 'docs/presentation/assets/2026-11-12-qna/depth-comparison-errors.png')
    P(s, image, 72, 290, 1456, 508)

    s = slide('A5: Lighting and normal substitutions', 'Q&A / lighting',
              'Sphere · HDR-0.75 · validation 51 crops + matched 0 EV environment maps | five-view diagnostic, no optimisation', backup=True)
    header(s, 'Q&A / A5', 'Sharper lighting alone leaves distorted reflections.')
    environment = 'docs/reports/assets/2026-11-03-environment-check/comparison.png'
    original = asset('forest-reference-map', environment, (0, 42, 512, 298))
    learned = asset('forest-learned-map', environment, (512, 42, 1024, 298))
    column_images(s, [original, learned], ['ORIGINAL FOREST', 'LEARNED ENVIRONMENT'], y=260, height=194,
                  x=72, width=1015, label_size=23)
    T(s, 1130, 286, 398, 146, 'Both reference inputs\nrestore more structure.\nContrast still differs.', 29, C['green'], True)
    column_images(s, swaps[51], ['TARGET', 'LEARNED / LEARNED', 'REF LIGHT / LEARNED N',
                               'LEARNED LIGHT / REF N', 'REFERENCE / REFERENCE'],
                  y=548, height=244, label_size=18)
    T(s, 73, 469, 1450, 36, 'Validation view 51 · lighting / normals', 23, C['muted'])

    s = slide('A6: Do losses improve the normals?', 'Q&A / loss trials',
              'Sphere · validation 0 normals shown; five-view mean errors | XYZ encoded as RGB; reference normals used for evaluation', backup=True)
    header(s, 'Q&A / A6', 'The independent normal check changes the conclusion.')
    for x, title, value, body in [(72, 'TRIAL 1 · NORMAL–DEPTH WEIGHT', '1.573° → 1.594°', 'No normal-accuracy improvement'),
                                   (820, 'TRIAL 2 · IMAGE OBJECTIVE', '1.573° → 1.576°', 'Normal accuracy essentially unchanged')]:
        R(s, x, 277, 708, 526, C['white'])
        T(s, x + 22, 293, 664, 36, title, 23, C['green'], True)
        T(s, x + 22, 349, 664, 70, value, 46, C['green'], True)
        T(s, x + 22, 749, 664, 46, body, 27, C['muted'])
    normal_sources = [('docs/reports/assets/2026-11-07-normal-weight-closeout/normal-comparison.png', 72,
                       ['WEIGHT 0.01', 'WEIGHT 0.03']),
                      ('docs/reports/assets/2026-11-07-refinement-loss/normal-normal-comparison.png', 820,
                       ['MSE', 'L1 + D-SSIM'])]
    for n, (path, x, labels) in enumerate(normal_sources):
        ims = [asset(f'trial-normal-{n}-{col}', path, (12 + col * 412, 68, 412 + col * 412, 468)) for col in [1, 2]]
        column_images(s, ims, labels, y=470, height=250, x=x + 22, width=664, label_size=23)

    s = slide('A7: Why simplify to a sphere?', 'Q&A / scope',
              'Separate diagnostics: enlarged Chrome · validation 18; original Materials · validation 13 | not pooled with sphere results', backup=True)
    header(s, 'Q&A / A7', 'Earlier failures explain the scope decision.')
    logo = asset('logo-failure', 'docs/reports/assets/2026-11-02-logo-position5x/logo-comparison.png')
    rays = asset('nearby-reflections', 'docs/reports/assets/2026-10-30/reflection-scope.png', (0, 440, 1120, 880))
    T(s, 73, 243, 805, 42, 'HIDDEN RECESS: RGB AND NORMALS', 25, C['green'], True)
    P(s, logo, 72, 300, 805, 475)
    T(s, 933, 243, 594, 42, 'LOCAL REFLECTION TRANSPORT', 25, C['green'], True)
    P(s, rays, 932, 304, 596, 264)
    T(s, 942, 583, 570, 82, '23%', 70, C['orange'], True)
    T(s, 942, 677, 570, 112, 'of traced Chrome rays hit other\ngeometry in this predefined view.\nThis is not a percentage of image error.', 26, C['muted'])

    assert len(SLIDES) == 19
    for number, s in enumerate(SLIDES, 1):
        footer(s, number)


def export_pptx():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W / 120), Inches(H / 120)
    prs.core_properties.title = 'Reconstructing reflections with deferred Gaussian surfels'
    prs.core_properties.subject = 'CP4282 | 15 minutes | 12 main slides + 7 Q&A slides'
    prs.core_properties.author = 'Tang Xiao Di Wendy'
    prs.core_properties.keywords = 'Gaussian surfels, reflections, SH3, CP4282'
    target_shapes = []
    for spec in SLIDES:
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.background.fill.solid(); s.background.fill.fore_color.rgb = drawing.rgb(spec['bg'])
        for ob in spec['objects']:
            x, y, w, h = [Inches(ob[k] / 120) for k in ('x', 'y', 'w', 'h')]
            if ob['kind'] == 'rect':
                shape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
                shape.fill.solid(); shape.fill.fore_color.rgb = drawing.rgb(ob['color']); shape.line.fill.background()
            elif ob['kind'] == 'image':
                shape = s.shapes.add_picture(str(ob['path']), x, y, width=w, height=h)
            else:
                shape = s.shapes.add_textbox(x, y, w, h)
                tf = shape.text_frame; tf.word_wrap = True
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                for i, value in enumerate(ob['value'].split('\n')):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    p.text = value; p.font.name = ob['font']; p.font.size = Pt(ob['size'] * .6)
                    p.font.bold = ob['bold']; p.font.color.rgb = drawing.rgb(ob['color'])
                    p.space_before = p.space_after = Pt(0); p.line_spacing = 1.12
                    if ob['align'] == 'right':
                        p.alignment = PP_ALIGN.RIGHT
            if 'target' in ob:
                target_shapes.append((shape, ob['target']))
        if spec['backup']:
            note_header = f"{spec['id']} | Q&A only; outside the timed talk"
        else:
            start = sum(DURATIONS[:len(prs.slides) - 1]); end = start + spec['seconds']
            note_header = f"Slide {len(prs.slides)} | {clock(start)}–{clock(end)} | target {clock(spec['seconds'])}"
        s.notes_slide.notes_text_frame.text = f"{note_header}\n\n{spec['notes']}\n\nVisual/source: {spec['citation']}"
    for shape, target in target_shapes:
        shape.click_action.target_slide = prs.slides[target]
    path = OUT / f'{STEM}.pptx'; prs.save(path)
    return path


def export_html():
    slides = []
    for i, spec in enumerate(SLIDES):
        objects = []
        for ob in spec['objects']:
            style = f"left:{ob['x']}px;top:{ob['y']}px;width:{ob['w']}px;height:{ob['h']}px;"
            if ob['kind'] == 'rect':
                objects.append(f'<div class="ob" style="{style}background:{ob["color"]}"></div>')
            elif ob['kind'] == 'image':
                b64 = base64.b64encode(ob['path'].read_bytes()).decode()
                objects.append(f'<img class="ob evidence" alt="{html.escape(ob["path"].stem)}" style="{style}" '
                               f'src="data:image/png;base64,{b64}" onclick="zoom(this.src)">')
            else:
                style += f"font-size:{ob['size']}px;color:{ob['color']};font-family:{ob['font']};font-weight:{700 if ob['bold'] else 400};text-align:{ob['align']};"
                jump = f' onclick="jump({ob["target"]})" tabindex="0" role="button"' if 'target' in ob else ''
                objects.append(f'<div class="ob txt" style="{style}"{jump}>{html.escape(ob["value"])}</div>')
        slides.append(f'<section class="slide" aria-label="{html.escape(spec["title"])}" style="background:{spec["bg"]}">{"".join(objects)}</section>')
    options = ''.join(f'<option value="{i}">{s["id"]}. {html.escape(s["title"])}</option>' for i, s in enumerate(SLIDES))
    notes = json.dumps([s['notes'] for s in SLIDES], ensure_ascii=False).replace('</', '<\\/')
    doc = '''<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reconstructing reflections — 15-minute talk</title><style>
*{box-sizing:border-box}body{margin:0;background:#101B19;color:#F3F2EC;font-family:Arial}#stage{position:relative;margin:auto}
.slide{width:1600px;height:900px;position:relative;overflow:hidden;display:none;transform-origin:top left}.slide.active{display:block}
.ob{position:absolute}.txt{line-height:1.12;white-space:pre-line}.evidence{cursor:zoom-in}.controls{padding:12px 20px;display:flex;align-items:center;gap:12px;font-size:13px}
button,select{background:#D6E8AF;color:#172D29;border:0;padding:8px 12px;border-radius:4px}select{max-width:370px}button{cursor:pointer}#count{margin-left:auto}
#notes{white-space:pre-wrap;max-width:1100px;margin:0 auto 25px;padding:20px 30px;background:#253C35;line-height:1.6;display:none}
dialog{border:0;padding:30px;background:#fff;max-width:98vw;max-height:98vh}dialog::backdrop{background:#000c}dialog img{max-width:92vw;max-height:87vh;object-fit:contain}dialog button{position:absolute;right:4px;top:4px}
@media print{.controls,#notes,dialog{display:none}.slide{display:block!important;transform:none!important;break-after:page}#stage{width:1600px!important;height:auto!important}}
</style></head><body><nav class="controls"><button onclick="go(-1)">← Previous</button><button onclick="go(1)">Next →</button><select id="picker" onchange="jump(Number(this.value))">'''+options+'''</select><button onclick="toggleNotes()">Notes (N)</button><button onclick="fullscreen()">Fullscreen (F)</button><span>12 main + 7 Q&A · click an image to enlarge</span><span id="count"></span></nav><main id="stage">'''+''.join(slides)+'''</main><aside id="notes"></aside><dialog id="zoom"><button onclick="document.getElementById('zoom').close()">Close</button><img alt="Enlarged experiment figure"></dialog><script>
const notes='''+notes+''';const ss=[...document.querySelectorAll('.slide')];let index=Math.max(0,Math.min(ss.length-1,Number(location.hash.slice(1)||1)-1));
function fit(){const scale=Math.min(innerWidth/1600,(innerHeight-65)/900);const st=document.getElementById('stage');st.style.width=1600*scale+'px';st.style.height=900*scale+'px';ss.forEach(s=>s.style.transform='scale('+scale+')')}
function show(){ss.forEach((s,i)=>s.classList.toggle('active',i===index));document.getElementById('count').textContent=(index+1)+' / '+ss.length;document.getElementById('picker').value=index;document.getElementById('notes').textContent=notes[index];history.replaceState(null,'','#'+(index+1));fit()}
function jump(i){index=Math.max(0,Math.min(ss.length-1,i));show()}function go(d){jump(index+d)}function toggleNotes(){const n=document.getElementById('notes');n.style.display=n.style.display==='block'?'none':'block'}
function fullscreen(){document.documentElement.requestFullscreen()}function zoom(src){const d=document.getElementById('zoom');d.querySelector('img').src=src;d.showModal()}
addEventListener('keydown',e=>{if(document.getElementById('zoom').open)return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(-1)}if(e.key.toLowerCase()==='n')toggleNotes();if(e.key.toLowerCase()==='f')fullscreen();if(e.key==='Home')jump(0);if(e.key==='End')jump(ss.length-1);if(e.key.toLowerCase()==='q')jump(12)});addEventListener('resize',fit);show();
</script></body></html>'''
    path = OUT / f'{STEM}.html'; path.write_text(doc, encoding='utf-8')
    return path


def render_and_check(path):
    candidates = [Path(os.environ.get(v, 'C:/')) / p for v in ('PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA')
                  for p in ('Google/Chrome/Application/chrome.exe', 'Microsoft/Edge/Application/msedge.exe')]
    executable = next(p for p in candidates if p.exists())
    records = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=str(executable), headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 965}, device_scale_factor=1)
        page.goto(path.as_uri(), wait_until='load')
        page.evaluate('async()=>{await document.fonts.ready;await Promise.all([...document.images].filter(i=>i.currentSrc||i.getAttribute("src")).map(i=>i.decode()))}')
        preview_style = page.add_style_tag(content='.controls,#notes{display:none!important}#stage{width:1600px!important;height:900px!important}.slide{transform:none!important}')
        for i, s in enumerate(SLIDES):
            page.evaluate('i=>jump(i)', i)
            active = page.locator('.slide.active')
            issues = active.evaluate("s=>[...s.querySelectorAll('.txt')].filter(e=>e.scrollHeight>e.clientHeight+3||e.scrollWidth>e.clientWidth+3).map(e=>({text:e.textContent,height:e.clientHeight,required:e.scrollHeight,width:e.clientWidth,requiredWidth:e.scrollWidth}))")
            records.append({'slide': i + 1, 'id': s['id'], 'title': s['title'], 'text_overflow': issues})
            active.screenshot(path=str(OUT / f'slide-{i + 1:02d}.png'))
        preview_style.evaluate('(element)=>element.remove()')
        # Exercise appendix navigation, speaker notes and image enlargement.
        page.evaluate('jump(11)')
        page.locator('.slide.active [role="button"]').first.click()
        assert page.evaluate('index') == 12
        page.evaluate('toggleNotes()')
        assert page.locator('#notes').is_visible()
        page.locator('.slide.active img').first.click()
        assert page.locator('dialog').is_visible()
        page.locator('dialog button').click()
        browser.close()
    (OUT / 'validation.json').write_text(json.dumps({'slides': records, 'main_slides': 12, 'qna_slides': 7,
                                                   'main_seconds': sum(DURATIONS), 'images_decoded': True,
                                                   'appendix_navigation': True, 'speaker_notes': True,
                                                   'image_enlargement': True}, indent=2) + '\n', encoding='utf-8')
    overview(range(12), 'overview-main.png', 4)
    overview(range(12, 19), 'overview-qna.png', 3)
    problems = [(r['slide'], r['text_overflow']) for r in records if r['text_overflow']]
    if problems:
        raise RuntimeError(f'Text overflow: {problems}')


def overview(indices, filename, columns):
    indices = list(indices); cell_w, cell_h = 400, 250
    board = Image.new('RGB', (columns * cell_w + 20, math.ceil(len(indices) / columns) * cell_h + 20), '#DDE3DA')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    for local, number in enumerate(indices):
        with Image.open(OUT / f'slide-{number + 1:02d}.png') as source_image:
            im = source_image.resize((380, 214), Image.Resampling.LANCZOS)
        x, y = 10 + local % columns * cell_w, 10 + local // columns * cell_h
        board.paste(im, (x, y)); draw.text((x, y + 218), f'{SLIDES[number]["id"]}  {SLIDES[number]["section"]}', fill=C['ink'], font=font)
    board.save(OUT / filename)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args(); OUT = args.output.resolve()
    drawing.OUT = OUT
    SLIDES.clear(); drawing.PROVENANCE.clear()
    (OUT / 'assets').mkdir(parents=True, exist_ok=True)
    source('docs/presentation/build_deck_15min.py')
    source('docs/presentation/build_deck.py')
    source('docs/presentation/requirements.txt')
    read_scripts(); drawing.verify_numbers(); build_content()
    pptx = export_pptx(); html_path = export_html()
    (OUT / 'speaker-notes.txt').write_text('\n\n'.join(
        f"{s['id']}. {s['title']}\nTarget: {clock(s['seconds']) if not s['backup'] else 'Q&A only'}\n\n{s['notes']}" for s in SLIDES), encoding='utf-8')
    (OUT / 'asset-manifest.json').write_text(json.dumps(list(drawing.PROVENANCE.values()), indent=2) + '\n', encoding='utf-8')
    render_and_check(html_path)
    prs = Presentation(pptx)
    assert len(prs.slides) == 19
    assert prs.core_properties.author == 'Tang Xiao Di Wendy'
    for i, s in enumerate(prs.slides):
        assert s.notes_slide.notes_text_frame.text.strip()
        if i < 12:
            assert MAIN_NOTES[i + 1] in s.notes_slide.notes_text_frame.text
        for shape in s.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= prs.slide_width + 10, (i + 1, shape.name)
            assert shape.top + shape.height <= prs.slide_height + 10, (i + 1, shape.name)
    (OUT / 'build-manifest.json').write_text(json.dumps({'author': prs.core_properties.author,
        'main_slides': 12, 'qna_slides': 7, 'main_minutes': 15, 'format': '16:9', 'pdf_built': False,
        'artifacts': [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in [pptx, html_path, OUT / 'speaker-notes.txt']]}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'slides': 19, 'main_minutes': 15,
                      'powerpoint': str(pptx), 'slideshow': str(html_path)}, indent=2))


if __name__ == '__main__':
    main()
