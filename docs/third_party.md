# Third-party projects and licenses

GSSceneRefine is intended for research and reproducible open-source workflows.
It integrates the following software through its package dependencies / APIs:

| Project | Source | License | Use in GSSceneRefine |
| --- | --- | --- | --- |
| PlayCanvas Engine | https://github.com/playcanvas/engine | MIT | Browser-side Gaussian display and orthographic cameras |
| SuperSplat Editor | https://github.com/playcanvas/supersplat | MIT | Interaction and performance design reference; no bundled editor source at present |
| SplatTransform | https://github.com/playcanvas/splat-transform | MIT | Required local PLY-to-SOG display-cache conversion (Node.js 22) |

Third-party code is not relicensed by this document. Keep each dependency's
license text and copyright notice when distributing modified/copied source,
and distribute any required notices with resulting public releases. The Node
package manager handles `node_modules` dependencies locally; `node_modules`
should not be committed to the repository. Academic/noncommercial use does
not by itself eliminate license obligations.

**Data provenance:** Gaussian PLY and SfM/Poisson input data remain under
their original licenses and ownership. GSSceneRefine does not upload model
files outside localhost. SOG files are visualization caches, and do not replace
full-precision inputs for measurements, validation, or 3D Gaussian masking.
