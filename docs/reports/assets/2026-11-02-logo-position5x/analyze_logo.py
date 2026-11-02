"""Float-render metrics and matched scientific figures for fixed validation ROI."""
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import warp as wp

FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'shared'),str(ROOT/'src')]
from evaluate_surfels import build_trainer
from gaussian_set import GaussianSet
from cp4282_gs.surfels.geometry import base
from image_metrics import ssim_rgb

baseline=ROOT/'outputs/Oct30-1733-materials_chrome-staged-hdr-refine-5k'
trial=ROOT/'outputs/Nov02-1651-materials_chrome-staged-hdr-refine-position5x-5k'
parent=ROOT/'outputs/Oct30-1718-materials_chrome-frozen-hdr-10k'
record=json.loads((trial/'run.json').read_text())
parent_record=json.loads((parent/'run.json').read_text())
assert record['status']=='completed'
assert record['training_result']['completed_iterations']==5000
history={'parent_run':str(parent),'current_run':str(trial),
         'parent_run_sha256':hashlib.sha256((parent/'run.json').read_bytes()).hexdigest(),
         'parent_iterations':10000,'current_stage_iterations':5000,'cumulative_iterations':15000,
         'parent_observed_seconds':parent_record['training_result']['trainer_seconds'],
         'current_observed_seconds':record['training_result']['trainer_seconds'],
         'cumulative_observed_seconds':parent_record['training_result']['trainer_seconds']+record['training_result']['trainer_seconds'],
         'scope':'Parent plus continuation; Adam restarts. Historical baseline training overlapped other work; not a controlled speed comparison.'}
with (trial/'stage_history.json').open('x',encoding='utf-8') as handle:
    json.dump(history,handle,indent=2)

wp.init()
targets,cameras,focal,_,frames=base.load_views(ROOT/'data/materials_chrome',800,800,np.ones(3,np.float32),manifest_name='transforms_val.json')
targets=targets[18:19].copy()
cameras=cameras[18:19].copy()
refpath=ROOT/'outputs/Oct30-1435-materials-chrome-audit/Oct30-1440-geometry-all-val-800/r_18.npz'
truth=dict(np.load(refpath))
roi=np.zeros((800,800),bool)
roi[365:575,215:455]=True
foreground=truth['object_id']>0
rows=[]
normal_images=[truth['normal']]
rgb_images=[targets[0]]
for label,run in [('Position LR 1x',baseline),('Position LR 5x',trial)]:
    args=SimpleNamespace(ply=run/'model.ply',background='white',tile_pair_capacity=None,sh_degree=None)
    scene=GaussianSet.from_ply(args.ply)
    trainer=build_trainer(args,scene,targets,cameras,focal,'cuda:0')
    image,_=trainer.render_single_view_image(0)
    image=np.clip(image,0,1)
    buffers=trainer.buffers.numpy().reshape(800,800,9)
    length=np.linalg.norm(buffers[:,:,3:6],axis=-1)
    normal=buffers[:,:,3:6]/np.maximum(length[:,:,None],1e-8)
    coverage=buffers[:,:,7]
    depth=buffers[:,:,6]/np.maximum(coverage,1e-8)
    valid=foreground&(coverage>=.5)&(length>1e-6)
    angle=np.degrees(np.arccos(np.clip(np.sum(normal*truth['normal'],axis=-1),-1,1)))
    for region,mask in [('logo_rectangle',roi),('foreground',foreground),('foreground_outside_logo_rectangle',foreground&~roi)]:
        chosen=valid&mask
        photo_mask=mask&foreground
        mse=float(((image-targets[0])**2)[photo_mask].mean())
        rows.append({'model':label,'region':region,'pixels':int(chosen.sum()),
                     'normal_valid_fraction':float(chosen.sum()/max((foreground&mask).sum(),1)),
                     'normal_mae_deg':float(angle[chosen].mean()),
                     'depth_mae':float(np.abs(depth[chosen]-truth['depth'][chosen]).mean()),
                     'psnr_db':float(-10*np.log10(max(mse,1e-12)))})
    rows[-3]['crop_ssim']=ssim_rgb(image[365:575,215:455],targets[0,365:575,215:455])
    normal_images.append(normal.copy())
    rgb_images.append(image.copy())
    del trainer,scene
    gc.collect()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
for col,label in enumerate(['Reference','Position LR 1x','Position LR 5x']):
    axes[0,col].imshow(rgb_images[col])
    axes[1,col].imshow(np.where(foreground[:,:,None],normal_images[col]*.5+.5,1))
    for row in range(2):
        axes[row,col].set_xlim(215,455)
        axes[row,col].set_ylim(575,365)
        axes[row,col].axis('off')
    axes[0,col].set_title(label+' / RGB')
    axes[1,col].set_title(label+' / normals')
fig.savefig(FOLDER/'logo-comparison.png',dpi=160)
plt.close(fig)
result={'scope':'Supplemental user-selected validation frame18; fixed rectangle from reference extent, not exact logo segmentation. RGB metrics use float renders; reference geometry is evaluation-only.',
        'view':18,'roi_xyxy':[215,365,455,575],'rows':rows,'runs':[str(baseline),str(trial)],
        'reference_sha256':hashlib.sha256(refpath.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'created_utc':datetime.now(timezone.utc).isoformat()}
(FOLDER/'logo-summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
