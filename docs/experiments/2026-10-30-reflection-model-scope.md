# What the environment-only reflection pass leaves out

Original Materials contains reflections of nearby shader balls, the pedestal
and the chrome mesh itself. A single environment `E(direction)` has no surface
position or secondary visibility. It cannot directly trace the corresponding
parallax, occlusion and multiple-bounce paths. Base SH supplies an approximation
to some remaining view dependence; increasing SH or training time does not turn
that environment lookup into local ray transport.

To quantify the issue, `scripts/audit_reflection_scope.py` reconstructs visible
Chrome surface points from independent Blender depth and camera poses, reflects
each camera ray using the interpolated mesh normal, and traces one outgoing ray
against glossy-visible geometry. Original Chrome has zero roughness and no shader
bump. Mesh/reference data is used **only for this diagnostic**, never as training
input. Four cameras were already fixed for earlier comparisons.

| Validation view | Chrome pixels traced | Escape to environment | Hit same Chrome mesh | Hit other geometry |
| --- | ---: | ---: | ---: | ---: |
| 0 | 20,172 | 68.8% | 18.8% | 12.3% |
| 13 | 3,664 | 65.0% | 12.0% | 23.0% |
| 73 | 8,502 | 76.6% | 3.0% | 20.4% |
| 88 | 9,779 | 76.2% | 9.1% | 14.6% |

All visible material pixels were traced in these views; the 50k sampling limit
was not reached. Pixel counts exactly match the independent evaluation tool's
Chrome masks, verifying image orientation and alpha alignment. Other-geometry
hit histograms include multiple separate `SolidModel` objects, as well as the
chrome assembly's core. Median other-hit distances are 0.28–0.35 scene units.

[Reference crops and ray-category figure](../../outputs/Oct30-1557-materials-reflection-scope/reflection-scope.png) ·
[Per-view counts](../../outputs/Oct30-1557-materials-reflection-scope/scope.csv) ·
[Scene/source fingerprints and object histograms](../../outputs/Oct30-1557-materials-reflection-scope/scope.json).

These percentages count ideal-mirror first bounces; they are not energy-weighted
path tracing or an image-error lower bound. The hybrid SH component can absorb
some missing effects. Shader bump, roughness integration and later bounces are
excluded. This demonstrates a representation limitation alongside the measured
normal/optimization problems; it does not attribute the entire quality gap to
local reflection or prove that surfels/deferred rendering are infeasible.

The isolated enlarged chrome experiment removes the other shader balls and tests
far-field reflection more directly. Its own core and self-reflections remain.
Keep original Materials as the primary benchmark, report both separately, and
prioritize correct surface normals before adding more shading parameters. Exact
local effects would require position-dependent residual appearance or secondary
visibility/ray tracing; GGX roughness alone does not supply those paths.

Run the audit with `blender --background SCENE --python-exit-code 1 --python
scripts/audit_reflection_scope.py -- --manifest MANIFEST --reference REFERENCE
--output outputs/MonDD-HHMM-reflection-scope`. The first attempt imported Pillow,
which Blender's bundled Python lacks, and Blender returned zero after its Python
exception. That attempt/state/source are preserved and marked failed after the
artifact check. The working version uses Blender's image API; the explicit
`--python-exit-code 1` makes future script failures visible to the controller.
