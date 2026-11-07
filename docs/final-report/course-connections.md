# Course references used in the report

The readability revision links course ideas to the project's actual choices. It does not copy the course's benchmark results into the project's own result tables. The source pages were fetched directly on 7 November 2026; web search returned no results, but the pages were accessible without another link from the user.

| Course material | Connection in the report | Difference or boundary |
| --- | --- | --- |
| [Unit 5: differentiable rendering](https://weitsang.github.io/gaussian-splatting-notes/unit-5/#loss-measure-image-error) | Section 4 separates the training objective from evaluation | Additional image and geometry losses use the project's implemented gradient paths |
| [Unit 6: metrics](https://weitsang.github.io/gaussian-splatting-notes/unit-6/#reading-the-three-together) | Section 5 defines the measures; Section 6 interprets disagreements using actual crops | Per-view results are camera variation, not repeated-seed uncertainty |
| [Unit 8: synthetic Warp training](https://weitsang.github.io/gaussian-splatting-notes/unit-8/), [Unit 9: full trainer](https://weitsang.github.io/gaussian-splatting-notes/unit-9/) | Section 2 names the implementation starting point | Surfel geometry, deferred HDR shading and diagnostic tooling are project extensions |
| [Unit 10: training loss](https://weitsang.github.io/gaussian-splatting-notes/unit-10/#why-mse-is-a-useful-but-limited-starting-point) | Dedicated Section 4 explains what image and geometry losses ask training to improve | Main study retains MSE. Additional L1/D-SSIM continuation differs from the course's MSE/D-SSIM curriculum; it does not isolate the structural term |
| [Unit 11: appearance beyond SH](https://weitsang.github.io/gaussian-splatting-notes/unit-11/#project-ideas-appearance-models-beyond-spherical-harmonics) | Section 2 motivates reflection direction and normals; Section 3 explains the chosen pipeline | Local SH3 surfel control, simple sphere and fixed budget; no claim to reproduce the course scene scores or establish the best possible SH degree |
| [Unit 12: initialization](https://weitsang.github.io/gaussian-splatting-notes/unit-12/#why-the-renderer-makes-optimization-local) | Section 3 explains the training-mask starting surface | Silhouettes cannot recover the recessed logo; reference mesh normals are reserved for diagnosis |

Section 4 also explains the two distinct meanings of training stages: the course's loss curriculum changes the image objective, while this study's main two stages freeze then release geometry. They should not be conflated.

The [source manifest](course-references.json) records page hashes, titles and exact referenced headings. Complete HTML/text snapshots stay in ignored `outputs/Nov07-report-course-notes/`. Run `python docs/final-report/verify_course_references.py` to check links offline; use `--capture` only after downloading the source snapshots when intentionally refreshing the manifest.

![Actual reflection-normal diagnostic motivating the connection to Unit 11](../reports/assets/2026-11-03-reference-normal-swap/normal-swap-highlight.png)

*Actual `justChromeBall` validation view 0: target / reference lighting with learned normals / reference lighting with reference normals. The image connects the course's normal/reflection discussion to a measured limitation of this implementation; it is not a course result or a new training result.*

No new training, test evaluation or experimental-image generation was performed for this rewrite.
