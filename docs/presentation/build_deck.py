"""Build the preliminary research talk from tracked evidence; never runs experiments.

Outputs: editable PPTX, offline HTML, PDF, slide PNGs, notes and an asset manifest.
Install requirements in an isolated environment; see README.md.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/presentation-preliminary"
ASSETS = ROOT / "docs/reports/assets"
W, H = 1600, 900
C = dict(bg="#F3F2EC", ink="#172D29", green="#365E4B", muted="#61716B",
         lime="#D6E8AF", soft="#E3E8DD", white="#FFFFFF", orange="#B34F2F", line="#CDD5CB")
SLIDES = []
PROVENANCE = {}


def rgb(value):
    return RGBColor.from_string(value.lstrip("#"))


def source(relative):
    path = ROOT / relative
    PROVENANCE[relative] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return path


def crop(name, relative, bounds=None, white=False):
    path = source(relative)
    im = Image.open(path)
    if bounds:
        im = im.crop(bounds)
    if white:
        bg = Image.new("RGBA", im.size, "white")
        bg.alpha_composite(im.convert("RGBA"))
        im = bg.convert("RGB")
    target = OUT / "assets" / (name + ".png")
    im.save(target)
    PROVENANCE[relative].setdefault("uses", []).append({"output": target.name, "crop_xyxy": bounds,
                                                       "alpha_composited_on_white": white})
    return target


def new(title, section, minutes, note, dark=False, backup=False):
    slide = {"title": title, "section": section, "minutes": minutes, "notes": note,
             "bg": C["ink"] if dark else C["bg"], "dark": dark, "backup": backup, "objects": []}
    SLIDES.append(slide)
    return slide


def rect(s, x, y, w, h, color):
    s["objects"].append(dict(kind="rect", x=x, y=y, w=w, h=h, color=color))


def text(s, x, y, w, h, value, size=30, color=None, bold=False, font="Arial", align="left"):
    s["objects"].append(dict(kind="text", x=x, y=y, w=w, h=h, value=value, size=size,
                             color=color or (C["bg"] if s["dark"] else C["ink"]), bold=bold,
                             font=font, align=align))


def picture(s, path, x, y, w, h):
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    fw, fh = iw * scale, ih * scale
    s["objects"].append(dict(kind="image", x=x+(w-fw)/2, y=y+(h-fh)/2, w=fw, h=fh, path=path))


def line(s, x, y, w, color=None):
    rect(s, x, y, w, 2, color or C["line"])


def header(s, eyebrow, headline, subtitle=None):
    text(s, 72, 42, 1456, 34, eyebrow.upper(), 20, C["lime"] if s["dark"] else C["green"], True)
    text(s, 72, 98, 1456, 150, headline, 61, font="Georgia")
    if subtitle:
        text(s, 75, 252, 1450, 58, subtitle, 27, C["lime"] if s["dark"] else C["muted"])


def footer(s, number, source_label):
    color = "#46605A" if s["dark"] else C["line"]
    line(s, 72, 833, 1456, color)
    text(s, 72, 852, 1310, 28, source_label, 18, "#C3D1C3" if s["dark"] else C["muted"])
    text(s, 1430, 848, 98, 32, f"{number:02d} / 10" if number <= 10 else f"B{number-10:02d}",
         20, C["lime"] if s["dark"] else C["green"], True, align="right")


def table(s, x, y, widths, rows, row_h=62, highlight=(), fs=28):
    for ri, row in enumerate(rows):
        if ri == 0:
            rect(s, x, y + ri*row_h, sum(widths), row_h, C["ink"])
        elif ri in highlight:
            rect(s, x, y + ri*row_h, sum(widths), row_h, C["lime"])
        else:
            rect(s, x, y + ri*row_h, sum(widths), row_h, C["white"])
        xx = x
        for ci, value in enumerate(row):
            text(s, xx+18, y+ri*row_h+16, widths[ci]-32, row_h-18, str(value), fs,
                 C["white"] if ri == 0 else C["ink"], ri == 0 or ri in highlight)
            xx += widths[ci]


def build_content():
    ref = crop("sphere-test-0", "data/justChromeBall/test/r_0.png", (160, 160, 640, 640), True)
    main = "docs/reports/assets/2026-11-07-report-analysis/test-crops-views-0-73.png"
    refs = [crop(f"test-{v}-{model}", main, (x, y, x+300, y+300))
            for v,y in [(0,68),(73,410)] for model,x in [("target",12),("sh3",324),("hdr",636)]]
    target, sh3, hdr, target73, sh373, hdr73 = refs
    normal = crop("normal-view-0-learned", "docs/reports/assets/2026-11-07-report-analysis/normal-example-view-0.png",
                  (374, 68, 724, 418))
    env = crop("environment-learned", "docs/reports/assets/2026-11-03-environment-check/comparison.png",
               (512,42,1024,298))
    swap_source = "docs/reports/assets/2026-11-03-reference-normal-swap/normal-swap-highlight.png"
    swap_images = [crop(f"normal-swap-{label}", swap_source, (x,68,x+300,368))
                   for label,x in [("target",12),("learned",324),("reference",636)]]
    loss = crop("loss-comparison", "docs/reports/assets/2026-11-07-refinement-loss/reflection-highlight-view-0.png")
    views = crop("dataset-views", "docs/reports/assets/2026-11-02-sphere/dataset-views.png")

    s = new("Reconstructing reflections", "Opening", 1,
        "OPEN: A chrome sphere is a small object that contains an image of a much larger world. "
        "The task is to reconstruct it from training images and render unseen camera views. "
        "The question is whether explicitly modelling reflections helps more than SH3 directional colour. "
        "Preview the answer: the reflection model improves the baseline, but thin reflected branches remain distorted. "
        "This is a preliminary CP4282 talk based on the reviewed 7 November report. Add your name/student number before presenting. "
        "VISUAL: actual Blender target, sphere test view 0; display crop with alpha composited on white.", dark=True)
    text(s,72,55,780,35,"CP4282  /  PRELIMINARY RESEARCH TALK",20,C["lime"],True)
    text(s,72,163,825,250,"Reconstructing\nreflections.",104,font="Georgia")
    text(s,77,445,730,108,"Deferred Gaussian surfels\nfor a chrome sphere",38)
    rect(s,76,599,695,93,C["green"])
    text(s,98,618,650,66,"Better images.\nWhat still prevents sharp reflections?",28,C["white"],True)
    rect(s,935,136,590,590,C["white"])
    picture(s,ref,945,146,570,570)
    text(s,938,748,590,42,"Actual Blender target · sphere · test view 0",21,"#C3D1C3")
    footer(s,1,"Source: report §§1–2  •  18-minute talk + 2-minute buffer  •  10 November 2026")

    s = new("Why reflections need more than directional colour", "Motivation", 1.5,
        "DEFINE SH3: a set of low-order directional functions predicts colour for each primitive as the camera direction changes. "
        "It is useful for view-dependent appearance, but its angular detail is limited. Here it blurs the tree structure. "
        "The reflection model uses a surface normal to compute where the camera ray reflects, then samples shared learned surroundings. "
        "A normal describes which way a surface faces; it is not a colour. "
        "Do not claim SH cannot represent any reflection, or that our comparison changes only deferred evaluation. "
        "TRANSITION: how do we build that reflected-direction model? Source: report §§2–3, Figure 3.")
    header(s,"01 / Motivation","A reflected branch needs the right direction.")
    for x,p,label in [(72,target,"TARGET"),(555,sh3,"SH3: DIRECTIONAL COLOUR"),(1038,hdr,"EXPLICIT REFLECTION MODEL")]:
        text(s,x,245,460,38,label,22,C["green"],True)
        picture(s,p,x,296,460,360)
    line(s,72,697,1456)
    text(s,72,723,685,75,"SH3\nView direction → primitive colour",30)
    text(s,830,723,698,75,"Reflection model\nView + normal → reflected surroundings",30,C["green"])
    footer(s,2,"Sphere · test view 0 · identical central crops · target / SH3 / HDR-1.0  |  report §§2–3, Fig. 3")

    s = new("Blend the surface, then shade the reflection", "Method", 2.5,
        "A surfel is a flat Gaussian surface patch, with position, orientation, two tangent scales, colour and reflection strength. "
        "Pass one projects the patches and blends surface attributes using visibility weights. "
        "Pass two computes the reflected direction from the combined normal, samples the environment and blends base/reflected light. "
        "Deferred means that reflection is computed after surface attributes are combined at the pixel. "
        "The environment map, reflection strengths and shared angular response are learned; the reference forest is not supplied. "
        "Point out the normal image and learned map are actual saved evidence. The arrows are a conceptual method diagram. "
        "Credit the course Warp trainer, Gaussian Surfels and 3DGS-DR. Our contribution is the implementation and controlled investigation, not a new rendering algorithm. "
        "Source: report §3 and Appendix A.")
    header(s,"02 / Method","Blend the surface. Then shade its reflection.","A two-pass renderer, implemented in the course’s Warp framework.")
    boxes=[(72,"01","Surface patches","Position, orientation,\ncolour, reflection strength"),
           (447,"02","Blend attributes","Normal, depth,\ncolour and coverage"),
           (822,"03","Reflect the ray","Camera ray +\ncombined surface normal"),
           (1197,"04","Sample & shade","Learned HDR map\n+ base colour")]
    for x,n,title,body in boxes:
        rect(s,x,345,330,220,C["white"])
        text(s,x+22,362,280,40,n,25,C["green"],True)
        text(s,x+22,418,288,42,title,30,bold=True)
        text(s,x+22,478,290,76,body,25,C["muted"])
        if x<1197: text(s,x+335,425,38,45,"→",36,C["green"])
    picture(s,normal,73,604,168,168)
    text(s,262,619,320,103,"Learned shading normal\nSphere · validation 0",24,C["muted"])
    picture(s,env,646,604,337,168)
    text(s,1009,619,492,115,"Learned surroundings\n512 × 256 HDR environment\nNo reference map used for fitting",25,C["muted"])
    footer(s,3,"Diagram: implemented pipeline  |  actual maps: HDR-0.75  |  course Units 8–11; Gaussian Surfels; 3DGS-DR")

    s = new("Fit appearance first, then refine geometry", "Training", 2,
        "Explain the two-stage policy as stabilising appearance on a common starting surface before allowing geometry to move. "
        "Stage one runs 10,000 steps with geometry and opacity frozen. Stage two runs 5,000 steps updating geometry, opacity and appearance. "
        "Both methods share this policy; their final surfaces can still differ. Adam state restarts at the transition. "
        "RGB MSE asks the render to match target pixel colours. The mask objective protects foreground coverage independently of RGB. "
        "Normal-depth consistency asks normals to agree with normals estimated from depth. It supplies internal agreement, not reference geometry. "
        "Mask weight is 0.1; the normal weight ramps to 0.01 over the first 1,000 refinement steps. Geometry losses are disabled in stage one. "
        "TRANSITION: what is held constant in the comparison? Source: report §§3.3–4.")
    header(s,"03 / Training","Fit appearance first. Then refine the surface.")
    rect(s,72,275,939,138,C["ink"])
    text(s,98,292,875,54,"10,000 steps",44,C["lime"],True)
    text(s,100,354,870,42,"Appearance only · geometry and opacity fixed",30,C["white"])
    rect(s,1033,275,495,138,C["lime"])
    text(s,1058,292,440,54,"5,000 steps",44,C["ink"],True)
    text(s,1058,354,440,42,"Joint refinement",30)
    for x,title,body in [(72,"RGB matching","Match target colours\nwith mean squared error."),
                         (565,"Mask matching","Keep object coverage\naligned with the silhouette."),
                         (1058,"Normal–depth agreement","Make two descriptions\nof the surface agree.")]:
        line(s,x,471,470)
        text(s,x,497,470,45,title,31,bold=True)
        text(s,x,559,470,100,body,29,C["muted"])
    rect(s,72,707,1456,91,C["soft"])
    text(s,96,732,1406,46,"Shared refinement objective:  MSE + 0.1 × mask + λ(t) × normal–depth",31,C["green"],True)
    footer(s,4,"λ(t) ramps to 0.01 over 1k refinement steps · geometry losses off during appearance fitting  |  report §§3.3–4")

    s = new("A controlled appearance comparison", "Experimental setup", 1.5,
        "The main scene is a chrome sphere under forest illumination. Materials originally mixed recess geometry, nearby reflections and appearance, "
        "so the sphere makes the central question easier to study. It also limits generalisation. "
        "Use 100 training, 100 validation and 200 test views, all 800 by 800. "
        "The SH3 and HDR-1.0 pair share training-mask initialisation, 100k surfels, seed 7 and the 10k+5k budget. Both use the same surfel representation. "
        "Training uses RGB and alpha; reference normals and lighting are only diagnostic. "
        "The comparison tests complete appearance models, not an isolated deferred-shading switch. "
        "Define PSNR as pixel colour accuracy, SSIM as local structural similarity and LPIPS as feature-based perceptual distance. "
        "Source: report §5. Reference photos below are validation views 0, 13, 73, 88.")
    header(s,"04 / Experimental setup","Same starting surface. Same training budget.","Primary comparison: SH3 versus HDR-1.0, both using Gaussian surfels.")
    for x,value,label in [(72,"100k","starting surface patches"),(565,"15k","steps per fitted model"),(1058,"1","scene · seed 7")]:
        text(s,x,337,470,104,value,86,C["green"],True)
        text(s,x,450,470,40,label,28,C["muted"])
    picture(s,views,72,522,738,222)
    text(s,72,756,738,30,"Actual targets · sphere · validation views 0 / 13 / 73 / 88",20,C["muted"])
    rect(s,860,522,668,277,C["white"])
    text(s,886,541,616,52,"100 train / 100 validation / 200 test",31,bold=True)
    text(s,886,609,616,54,"800 × 800 pixels · disjoint camera poses",26,C["muted"])
    text(s,886,686,616,87,"PSNR ↑  pixel accuracy\nSSIM ↑  structure     LPIPS ↓  perceptual distance",27,C["green"])
    footer(s,5,"Training: RGB + alpha only · reference environment and mesh normals reserved for diagnostics  |  report §5")

    s = new("Explicit reflections improve every tested view", "Main result", 2.5,
        "Lead with the actual crops, then the numbers. At equal starting size, HDR-1.0 raises mean test PSNR from 20.2655 to 24.5732 dB "
        "and lowers LPIPS from 0.128760 to 0.083517. That is +4.31 dB and a 35.1 percent reduction. "
        "All 200 test cameras improve in PSNR, SSIM, LPIPS and foreground PSNR. "
        "These are many views of one fitted model pair, not 200 independent training repetitions. "
        "Point at the sky gap and tree structure: more is recognisable, but branches remain warped. "
        "The additional HDR-0.75 size variant provides only a modest further gain; keep the main claim on the matched-size comparison. "
        "Source: report §6.1; saved 200-view test metrics. Backup B1 includes all models and metrics.")
    header(s,"05 / Main result · 200 test views","The reflection model improves every tested view.")
    for x,p,label in [(72,target,"TARGET"),(448,sh3,"SH3"),(824,hdr,"HDR-1.0")]:
        text(s,x,253,350,34,label,22,C["green"],True)
        picture(s,p,x,298,350,350)
    text(s,1230,298,298,96,"+4.31",77,C["green"],True)
    text(s,1234,402,280,42,"dB PSNR",31,C["muted"])
    text(s,1230,485,298,96,"−35.1%",65,C["green"],True)
    text(s,1234,583,280,42,"LPIPS",31,C["muted"])
    table(s,72,694,[510,470,476],[["200-view test mean","PSNR ↑","LPIPS ↓"],
                                 ["SH3 → HDR-1.0","20.27 → 24.57","0.1288 → 0.0835"]],row_h=54,highlight=(1,),fs=27)
    footer(s,6,"Crops: sphere · test view 0 · target / SH3 / HDR-1.0  |  metrics: all 200 test views  |  report §6.1")

    s = new("Higher scores still leave distorted reflections", "Remaining failure", 1,
        "Slow down and acknowledge the failure plainly. Use test view 73 and point to the fence in the lower left, "
        "then a thin branch. The HDR render is recognisably better than SH3, but fine structures remain smeared or bent. "
        "This is why the talk does not stop at the headline score. "
        "Ask: is the model failing to learn the environment, the surface direction, or their combination? "
        "The next diagnostic changes saved rendering inputs; it is not extra successful training. Source: report §§6.1–6.3.",dark=True)
    header(s,"06 / Remaining failure","Better scores. Still distorted branches.","The improvement does not recover the fine reflected structure.")
    for x,p,label in [(72,target73,"TARGET"),(820,hdr73,"HDR-1.0")]:
        text(s,x,324,708,34,label,23,C["lime"],True)
        picture(s,p,x,376,708,425)
    footer(s,7,"Sphere · test view 73 · identical target / HDR-1.0 reflection crops · inspect the fence and thin branches  |  Fig. 3")

    s = new("Normals and lighting must be checked together", "Diagnostic", 3,
        "Explain the intervention before interpreting it. Take the same saved HDR-0.75 model and render four combinations of learned/reference lighting "
        "and learned/reference shading normals. Ground-truth inputs are diagnostic only. Reference normals are from the actual Blender mesh. "
        "The large images hold reference lighting fixed and replace only the shading-normal direction. Coverage, depth and learned materials remain fixed. "
        "Recognisable branches return. Across five validation views, LPIPS drops from 0.085452 to 0.056796 under reference lighting. "
        "But substituting normals under learned lighting worsens LPIPS from 0.096202 to 0.101984. This conditional reversal suggests normal/light compensation. "
        "The mean normal error is only about 1.47 degrees, yet the reflection can be visibly distorted. "
        "Do not call the reference-input result a trained method or proof of complete physical disentanglement. PSNR does not rank these cases in the same order. "
        "Source: report §6.3. All four cases and metric directions are in backup B2.")
    header(s,"07 / Diagnostic · reference inputs, not training","Normal and lighting errors interact.","Hold reference lighting fixed, then replace the shading normals.")
    for x,p,label in zip([72,565,1058],swap_images,["TARGET","REF LIGHT + LEARNED NORMALS","REF LIGHT + REFERENCE NORMALS"]):
        text(s,x,325,470,39,label,22,C["green"],True)
        picture(s,p,x,373,470,339)
    rect(s,72,735,1456,80,C["lime"])
    text(s,94,746,1410,65,"Reference lighting: correcting normals lowers LPIPS 0.0855 → 0.0568.\nLearned lighting: the same correction worsens LPIPS 0.0962 → 0.1020.",26,C["ink"],True)
    footer(s,8,"Images: sphere · validation 0 · target / reference light + learned normals / reference light + reference normals  |  means: 5 views")

    s = new("Changing the loss changes the trade-off", "Follow-up experiments", 2,
        "Two validation-only follow-ups tested whether a simple loss change resolves the failure. "
        "First, raising normal-depth weight from 0.01 to 0.03 slightly worsens image scores and reference-normal error: "
        "1.5728 to 1.5943 degrees on five diagnostic views. More agreement is not more accuracy. "
        "Second, from the same MSE-trained frozen parent, replace refinement MSE with 0.8 L1 + 0.2 D-SSIM for 5k steps. "
        "Mean validation LPIPS improves 10.15 percent across 100 views, while PSNR drops 0.56 dB. Normal error remains essentially unchanged "
        "at 1.5728 to 1.5759 degrees. Show the actual crop: contrast changes, but thin branches remain distorted. "
        "This changes both the pixel loss and structural term, and their balance against geometry losses; it does not isolate SSIM alone. "
        "These trials do not replace the main test models. Source: report §§6.4–6.5.")
    header(s,"08 / Follow-ups · validation only","Changing the loss changes the trade-off.")
    rect(s,72,262,660,194,C["soft"])
    text(s,96,280,612,45,"3× normal–depth weight",32,bold=True)
    text(s,96,344,612,91,"Normal error: 1.573° → 1.594°\nNo recovery of fine reflection detail.",28,C["muted"])
    rect(s,756,262,772,194,C["lime"])
    text(s,780,280,724,45,"L1 + D-SSIM refinement",32,bold=True)
    text(s,780,344,724,91,"10.15% lower LPIPS · −0.56 dB PSNR\nNormal accuracy essentially unchanged.",28,C["green"])
    picture(s,loss,290,486,1020,310)
    footer(s,9,"Sphere · validation 0 · target / MSE / L1 + D-SSIM  |  image means: 100 views; normal error: 5 views  |  §§6.4–6.5")

    s = new("Reconstruction quality needs more than one score", "Conclusion", 1,
        "Return to the research question. In this controlled sphere study, explicit reflections improve the matched SH3 baseline. "
        "The unresolved details are linked to lighting and normal errors that can compensate for one another. "
        "Neither stronger internal consistency nor lower perceptual distance guarantees correct geometry. "
        "The practical lesson is to examine image metrics, matched reflection crops and independent normal diagnostics together. "
        "State the scope: one synthetic scene, one seed, and complete appearance models. Many cameras do not replace independent training repetitions. "
        "Next: compare MSE + D-SSIM to separate the loss components, then repeat across seeds and another reflective scene. "
        "CLOSE: The reflection model improves the baseline, but recovering sharp reflections requires checking geometry and lighting together. "
        "Pause for questions. The three backup slides contain full results, the diagnostic table and attribution.",dark=True)
    header(s,"09 / Takeaways","Reconstruction quality needs\nmore than one score.")
    for y,n,body in [(320,"01","Explicit reflections improve the SH3 comparison."),
                     (434,"02","Normals and lighting jointly limit fine detail."),
                     (548,"03","Better perceptual scores do not establish correct geometry.")]:
        text(s,74,y,95,66,n,43,C["lime"],True)
        text(s,188,y+4,1320,83,body,37,C["bg"])
    line(s,72,670,1456,"#46605A")
    text(s,74,699,1450,48,"Scope: one synthetic scene, one seed; comparison of complete appearance models.",27,"#C3D1C3")
    text(s,74,758,1450,48,"Next: isolate image-loss components, then repeat across seeds and scenes.",27,C["lime"])
    footer(s,10,"Evidence: image metrics + matched reflection crops + reference-normal checks  |  report §§7–8  |  Questions")

    s = new("Full test results", "Backup / results", 0,
        "Use only for questions. HDR-0.75 changes initial tangent scales to 75 percent; those scales can change during refinement. "
        "It was selected on validation, and is additional to the same-size comparison. Its extra gain over HDR-1.0 is about 0.17 dB. "
        "All models retain 100 percent foreground coverage under the stated threshold. The gains across cameras are descriptive camera variation, "
        "not training-seed uncertainty. Main checkpoint sizes are 23.50 MB SH3 versus 8.54 MB HDR in this implementation; "
        "that is stored model data, not GPU memory. Source: report §6.1 and Appendix C.",backup=True)
    header(s,"Backup / Results","The size variant adds only a modest gain.","200 sphere test views · 100k patches · 15k steps · one training seed")
    table(s,72,342,[472,240,240,240,264],[["Model","PSNR ↑","SSIM ↑","LPIPS ↓","FG PSNR ↑"],
          ["SH3","20.2655","0.8214","0.12876","13.6520"],
          ["HDR-1.0 · matched","24.5732","0.8809","0.08352","18.2494"],
          ["HDR-0.75 · additional","24.7453","0.8840","0.08099","18.4427"]],row_h=74,highlight=(2,),fs=27)
    text(s,73,689,1450,88,"Matched HDR-1.0 improves PSNR, SSIM, LPIPS and foreground PSNR on 200 / 200 views.\nCamera variation is not uncertainty across independent training seeds.",29,C["green"])
    footer(s,11,"Source: saved test/metrics.csv; report Table 1 · HDR-0.75 denotes initial tangent-scale multiplier")

    s = new("Four-way lighting and normal diagnostic", "Backup / diagnostic", 0,
        "Read rows conditionally. Under reference lighting, reference normals improve LPIPS by 0.028657. "
        "Under learned lighting, they worsen it by 0.005782. The reference/reference pair has best SSIM and LPIPS, "
        "but lower PSNR than the learned/learned pair. Brightness, learned materials and shader differences remain. "
        "All rows use the same HDR-0.75 model and five validation views 0, 13, 73, 88, 51. "
        "The diagnostic substitutes shading normal direction after rasterisation, preserving depth, coverage and material buffers. "
        "Source: report §6.3 and normal-swap metrics.csv. Not an achieved training result or full relighting claim.",backup=True)
    header(s,"Backup / Diagnostic","Correcting one input alone can make it worse.","Same saved HDR-0.75 model · five validation cameras · reference inputs used only here")
    table(s,72,334,[360,360,240,240,256],[["Lighting","Normals","PSNR ↑","SSIM ↑","LPIPS ↓"],
          ["Learned","Learned","23.8554","0.8719","0.09620"],
          ["Reference","Learned","20.4016","0.8446","0.08545"],
          ["Learned","Reference","23.0536","0.8679","0.10198"],
          ["Reference","Reference","22.7073","0.9436","0.05680"]],row_h=69,highlight=(4,),fs=27)
    text(s,74,715,1450,79,"The jointly corrected case has the best SSIM / LPIPS, but not the best PSNR.\nThis diagnoses compensation; it does not demonstrate a physically disentangled reconstruction.",28,C["green"])
    footer(s,12,"Source: reference-normal-swap/metrics.csv; report Table 2 · views 0 / 13 / 73 / 88 / 51")

    s = new("What comes from prior work, and what this project adds", "Backup / attribution", 0,
        "The course supplies projection, compositing and training infrastructure. Gaussian Surfels motivates the flat representation and normal-depth consistency. "
        "3DGS-DR motivates computing reflection after blending surface attributes. RGS-DR is closely related work, not reproduced in full. "
        "The project implements the selected model in Warp, adds the learned HDR/angular-response configuration and conducts the staged controlled comparisons and diagnostics. "
        "No new rendering algorithm is claimed. The report's file-level attribution and personal AI-use declaration still need author confirmation. "
        "Codex assisted implementation/debugging, experiment tooling, literature lookup and drafting; confirm the exact account personally before presenting. "
        "All experimental images in this deck come from tracked actual experiments. No experiments were rerun and no experimental image was AI-generated.",backup=True)
    header(s,"Backup / Attribution","Implemented and investigated in Warp.")
    for y,title,body in [(275,"Course framework","Projection, compositing and optimisation infrastructure · Units 8–11"),
                         (406,"Gaussian Surfels / 3DGS-DR","Flat patches and consistency / reflection after blending surface attributes"),
                         (537,"Project additions","HDR/angular-response implementation, staged comparisons and error diagnostics")]:
        text(s,74,y,1450,52,title,35,bold=True)
        text(s,74,y+61,1450,53,body,28,C["muted"])
    text(s,74,705,1450,82,"Related work: RGS-DR · full system not reproduced.\nAI assistance: Codex supported implementation, tooling and drafting; author confirmation pending.",26,C["green"])
    footer(s,13,"References: Gaussian Surfels (arXiv:2404.17774); 3DGS-DR (2404.18454); RGS-DR (2504.18468); report Appendix A")


def export_pptx():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W/120), Inches(H/120)
    prs.core_properties.title = "Reconstructing reflections — preliminary research talk"
    prs.core_properties.subject = "CP4282 · 10 main slides + 3 backup slides · 18 minutes"
    prs.core_properties.author = "CP4282 project — author details pending"
    prs.core_properties.keywords = "Gaussian surfels, reflection, SH3, CP4282, preliminary"
    for spec in SLIDES:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb(spec["bg"])
        for ob in spec["objects"]:
            x,y,w,h = [Inches(ob[k]/120) for k in ("x","y","w","h")]
            if ob["kind"] == "rect":
                shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,x,y,w,h)
                shape.fill.solid(); shape.fill.fore_color.rgb = rgb(ob["color"])
                shape.line.fill.background()
            elif ob["kind"] == "image":
                slide.shapes.add_picture(str(ob["path"]),x,y,width=w,height=h)
            else:
                shape = slide.shapes.add_textbox(x,y,w,h)
                tf = shape.text_frame
                tf.word_wrap = True
                tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                for i, value in enumerate(ob["value"].split("\n")):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    p.text = value
                    p.font.name = ob["font"]; p.font.size=Pt(ob["size"]*0.6)
                    p.font.bold=ob["bold"]; p.font.color.rgb=rgb(ob["color"])
                    p.space_before=Pt(0); p.space_after=Pt(0); p.line_spacing=1.12
                    if ob["align"] == "right":
                        from pptx.enum.text import PP_ALIGN
                        p.alignment = PP_ALIGN.RIGHT
        note = f"{spec['section']} | Target time: {spec['minutes']:g} minutes\n\n{spec['notes']}"
        slide.notes_slide.notes_text_frame.text = note
    path=OUT/"reflections-preliminary.pptx"
    prs.save(path)
    return path


def export_html():
    slides=[]
    for i,s in enumerate(SLIDES):
        objects=[]
        for ob in s["objects"]:
            style=f"left:{ob['x']}px;top:{ob['y']}px;width:{ob['w']}px;height:{ob['h']}px;"
            if ob["kind"] == "rect":
                objects.append(f'<div class="ob" style="{style}background:{ob["color"]}"></div>')
            elif ob["kind"] == "image":
                b64=base64.b64encode(ob["path"].read_bytes()).decode()
                objects.append(f'<img class="ob" alt="Actual experiment asset: {ob["path"].stem}" style="{style}" src="data:image/png;base64,{b64}">')
            else:
                style+=f"font-size:{ob['size']}px;color:{ob['color']};font-family:{ob['font']};font-weight:{700 if ob['bold'] else 400};text-align:{ob['align']};"
                objects.append(f'<div class="ob txt" style="{style}">{html.escape(ob["value"])}</div>')
        slides.append(f'<section class="slide" data-index="{i}" aria-label="{html.escape(s["title"])}" style="background:{s["bg"]}">{"".join(objects)}</section>')
    notes=json.dumps([s["notes"] for s in SLIDES]).replace("</","<\\/")
    doc='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reconstructing reflections — preliminary deck</title><style>
*{box-sizing:border-box}body{margin:0;background:#101B19;color:#F3F2EC;font-family:Arial}#stage{position:relative;margin:auto}
.slide{width:1600px;height:900px;position:relative;overflow:hidden;display:none;transform-origin:top left}.slide.active{display:block}
.ob{position:absolute}.txt{line-height:1.12;white-space:pre-line;overflow:visible}.controls{padding:14px 24px;display:flex;align-items:center;gap:18px;font-size:14px}
button{background:#D6E8AF;color:#172D29;border:0;padding:9px 15px;border-radius:4px;cursor:pointer}button:focus-visible{outline:3px solid white}
#notes{max-width:1100px;margin:0 auto 30px;padding:20px 30px;background:#253C35;line-height:1.6;display:none}#counter{margin-left:auto}
@page{size:1600px 900px;margin:0}@media print{body{background:white}.slide,.slide.active{display:block!important;transform:none!important;break-after:page;margin:0}.controls,#notes{display:none!important}#stage{width:1600px!important;height:auto!important}}
</style></head><body><div class="controls"><button onclick="go(-1)">← Previous</button><button onclick="go(1)">Next →</button><button onclick="toggleNotes()">Speaker notes (N)</button><button onclick="document.documentElement.requestFullscreen()">Fullscreen (F)</button><span>Arrow keys / Space to advance · 1–10 main talk · 11–13 backup</span><span id="counter"></span></div><main id="stage">'''+"".join(slides)+'''</main><aside id="notes"></aside><script>
const notes='''+notes+''';let index=Math.max(0,Math.min(12,Number(location.hash.slice(1)||1)-1));const ss=[...document.querySelectorAll('.slide')];
function fit(){let scale=Math.min(innerWidth/1600,(innerHeight-70)/900);let stage=document.getElementById('stage');stage.style.width=1600*scale+'px';stage.style.height=900*scale+'px';ss.forEach(s=>s.style.transform='scale('+scale+')')}
function show(){ss.forEach((s,i)=>s.classList.toggle('active',i===index));document.getElementById('counter').textContent=(index+1)+' / '+ss.length;document.getElementById('notes').textContent=notes[index];history.replaceState(null,'','#'+(index+1));fit()}
function go(delta){index=Math.max(0,Math.min(ss.length-1,index+delta));show()}function toggleNotes(){let n=document.getElementById('notes');n.style.display=n.style.display==='block'?'none':'block'}
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(-1)}if(e.key.toLowerCase()==='n')toggleNotes();if(e.key.toLowerCase()==='f')document.documentElement.requestFullscreen();if(e.key==='Home'){index=0;show()}if(e.key==='End'){index=ss.length-1;show()}});addEventListener('resize',fit);show();
</script></body></html>'''
    target=OUT/"reflections-preliminary.html"
    target.write_text(doc,encoding="utf-8")
    return target


def render_and_check(path):
    browsers=[Path(os.environ.get(v,"C:/"))/p for v in ("PROGRAMFILES","PROGRAMFILES(X86)","LOCALAPPDATA")
              for p in ("Google/Chrome/Application/chrome.exe","Microsoft/Edge/Application/msedge.exe")]
    executable=next(p for p in browsers if p.exists())
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=str(executable),headless=True)
        page=browser.new_page(viewport={"width":1600,"height":970},device_scale_factor=1)
        page.goto(path.as_uri(),wait_until="load")
        page.evaluate("async()=>{await document.fonts.ready;await Promise.all([...document.images].map(i=>i.decode()))}")
        page.add_style_tag(content=".controls,#notes{display:none!important}#stage{width:1600px!important;height:900px!important}.slide{transform:none!important}")
        qa=[]
        for i in range(len(SLIDES)):
            page.evaluate("i=>{index=i;show()}",i)
            active=page.locator(".slide.active")
            issues=active.evaluate("s=>[...s.querySelectorAll('.txt')].filter(e=>e.scrollHeight>e.clientHeight+3||e.scrollWidth>e.clientWidth+3).map(e=>({text:e.textContent,height:e.clientHeight,required:e.scrollHeight,width:e.clientWidth,requiredWidth:e.scrollWidth}))")
            if issues:
                raise RuntimeError(f"Text overflow on slide {i+1}: {issues}")
            active.screenshot(path=str(OUT/f"slide-{i+1:02d}.png"))
            qa.append({"slide":i+1,"title":SLIDES[i]["title"],"text_overflow":issues})
        page.emulate_media(media="print")
        page.pdf(path=str(OUT/"reflections-preliminary.pdf"),print_background=True,prefer_css_page_size=True)
        browser.close()
    import pymupdf
    doc=pymupdf.open(OUT/"reflections-preliminary.pdf")
    assert len(doc)==len(SLIDES),(len(doc),len(SLIDES))
    assert all(p.get_text().strip() for p in doc),"Blank PDF page"
    qa.append({"pdf_pages":len(doc),"format":"16:9", "speaker_minutes":sum(s["minutes"] for s in SLIDES)})
    (OUT/"validation.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    thumbs=[]
    for i in range(len(SLIDES)):
        im=Image.open(OUT/f"slide-{i+1:02d}.png").convert("RGB")
        im.thumbnail((480,270))
        thumbs.append(im)
    board=Image.new("RGB",(3*504+24,5*310+24),"#DDE3DA")
    draw=ImageDraw.Draw(board)
    for i,im in enumerate(thumbs):
        x=24+(i%3)*504;y=24+(i//3)*310
        board.paste(im,(x,y));draw.text((x,y+278),f"{i+1:02d}  {SLIDES[i]['section']}",fill="#172D29")
    board.save(OUT/"overview.png")


def verify_numbers():
    # Read the archived means rather than rely only on rounded prose.
    study=source("docs/reports/assets/2026-11-03-sphere-study/test/metrics.csv")
    rows=list(csv.DictReader(study.open(encoding="utf-8-sig")))
    assert len(rows)==600, len(rows)
    means={}
    for model in ("SH3","HDR-1.0","HDR-0.75"):
        arm=[r for r in rows if r["model"]==model]
        assert len(arm)==200
        means[model]={metric:sum(float(r[metric]) for r in arm)/200
                      for metric in ("psnr_db","ssim","lpips_alex","foreground_psnr_db")}
    assert round(means["SH3"]["psnr_db"],4)==20.2655
    assert round(means["HDR-1.0"]["psnr_db"],4)==24.5732
    assert round(means["HDR-1.0"]["psnr_db"]-means["SH3"]["psnr_db"],2)==4.31
    assert round((1-means["HDR-1.0"]["lpips_alex"]/means["SH3"]["lpips_alex"])*100,1)==35.1
    for metric in ("psnr_db","ssim","lpips_alex","foreground_psnr_db"):
        a={r["view"]:float(r[metric]) for r in rows if r["model"]=="SH3"}
        b={r["view"]:float(r[metric]) for r in rows if r["model"]=="HDR-1.0"}
        assert all((b[v]<a[v]) if metric=="lpips_alex" else (b[v]>a[v]) for v in a)
    loss=source("docs/reports/assets/2026-11-07-refinement-loss/validation-summary.csv")
    lr=list(csv.DictReader(loss.open()))
    drop=(1-float(lr[1]["lpips_alex"])/float(lr[0]["lpips_alex"]))*100
    assert round(drop,2)==10.15
    assert round(float(lr[1]["psnr_db"])-float(lr[0]["psnr_db"]),2)==-0.56
    source("docs/final-report/report.md")


def main():
    (OUT/"assets").mkdir(parents=True,exist_ok=True)
    verify_numbers()
    build_content()
    export_pptx()
    target=export_html()
    notes=[]
    for i,s in enumerate(SLIDES,1):
        notes.append(f"{i:02d}. {s['title']}\nTarget: {s['minutes']:g} minutes\n\n{s['notes']}\n")
    (OUT/"speaker-notes.txt").write_text("\n\n".join(notes),encoding="utf-8")
    (OUT/"asset-manifest.json").write_text(json.dumps(list(PROVENANCE.values()),indent=2),encoding="utf-8")
    render_and_check(target)
    deck=Presentation(OUT/"reflections-preliminary.pptx")
    assert len(deck.slides)==13
    assert all(s.notes_slide.notes_text_frame.text.strip() for s in deck.slides)
    # Geometry checks apply to native PowerPoint objects too.
    for i,s in enumerate(deck.slides,1):
        for shape in s.shapes:
            assert shape.left>=0 and shape.top>=0,(i,shape.name)
            assert shape.left+shape.width<=deck.slide_width+10,(i,shape.name)
            assert shape.top+shape.height<=deck.slide_height+10,(i,shape.name)
    print(json.dumps({"output":str(OUT),"slides":len(SLIDES),"talk_minutes":sum(s['minutes'] for s in SLIDES),
                      "artifacts":[p.name for p in OUT.glob('reflections-*')]}))


if __name__=="__main__":
    main()
