# Research presentation

The current deck has **12 main slides for a 15-minute talk and 7 Q&A slides**, in 16:9 format. It uses British English, the agreed speaking script, native editable PowerPoint text/shapes, and actual saved experiment images.

- [PowerPoint deck](../../outputs/presentation-15min/reconstructing-reflections-15min.pptx): editable slides with full speaker notes.
- [Offline slideshow](../../outputs/presentation-15min/reconstructing-reflections-15min.html): self-contained HTML, including notes, appendix navigation and image enlargement.
- [Main-slide overview](../../outputs/presentation-15min/overview-main.png) and [Q&A overview](../../outputs/presentation-15min/overview-qna.png).
- [Speaking script](speaker-script-15min.md) and [visual Q&A appendix](qna-visual-appendix.md): narrative, timing, explanations and source figures.

The main talk covers motivation, Materials exploration, SH3 versus reflection shading, the deferred method, shared training, full-view test results, refinement, all four lighting/normal combinations, the two loss trials, scope and conclusions. Main speaker notes reproduce the spoken paragraphs from the script.

Q&A slides provide more full-view comparisons, RGB differences, normal and depth checks, additional lighting substitutions, loss-trial normal maps and earlier scene failures. They follow the conclusion and sit outside the 15-minute timing.

## Build

Use the isolated presentation environment:

```powershell
py -3.12 -m venv outputs/presentation-tools
./outputs/presentation-tools/Scripts/python.exe -m pip install -r docs/presentation/requirements.txt
./outputs/presentation-tools/Scripts/python.exe docs/presentation/build_deck_15min.py
```

An installed Chrome or Edge is required for preview checks. Outputs stay under ignored `outputs/presentation-15min/` and include the PPTX, HTML, nineteen slide PNGs, two overview sheets, speaker notes, asset/crop provenance and validation/build manifests. No PDF, model rendering or training is performed.

The [new builder](build_deck_15min.py) reuses drawing primitives from [the preliminary builder](build_deck.py), while replacing the content, timing, author metadata, notes and exports. The earlier 20-minute deck and builder remain available separately under `outputs/presentation-preliminary/` and `build_deck.py`.

## Present

PowerPoint contains clickable Q&A topic links on the conclusion slide. In the offline slideshow, Arrow keys or Space advance, N toggles speaker notes, F enters fullscreen, and Q jumps to the first Q&A slide. The dropdown selects any slide; click an image to enlarge it and press Escape to close it. All images are embedded, so no network connection is needed.

Timings are rehearsal targets. The script contains approximately 1,600 spoken words, leaving time for pointing at figures, pauses and transitions. The planned main-slide durations sum to 15 minutes; Q&A has no allotted speaking time.

## Evidence and validation

![Main-slide overview](../../outputs/presentation-15min/overview-main.png)

*Twelve main slides. Actual sphere and Materials images are drawn from the saved experiment evidence; the method pipeline uses editable explanatory shapes. The 200-view test comparison remains separate from five-view normal/lighting diagnostics and validation-only loss trials.*

![Q&A-slide overview](../../outputs/presentation-15min/overview-qna.png)

*Seven Q&A slides, including additional actual RGB comparisons, normal/depth/error maps, environment substitutions, loss-trial normals and earlier failures. Every scientific display retains its scene/view/model labels and relevant scale information.*

Checks verify saved metric claims, all embedded images loading, text overflow on nineteen HTML slide previews, native PowerPoint object bounds, speaker-note inclusion, 15-minute timing, appendix navigation and image enlargement. The main and Q&A preview sheets were visually reviewed. PNG/HTML previews share the slide coordinates with PowerPoint; review in the presenting application before delivery.

[make_qna_figures.py](make_qna_figures.py) rebuilds the three diagnostic comparison sheets from archived saved arrays. The [Q&A appendix](qna-visual-appendix.md) explains the RGB/normal/depth differences and the source of each figure. [The report](../final-report/report.md) retains the detailed method and full evidence.

**Good commit point:** the script, Q&A appendix, deck builder, README and portable Q&A data/images. Copy the generated PowerPoint for presentation; generated outputs are ignored by Git and can be rebuilt.
