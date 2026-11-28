# tactile-object-classification

A blind robot (UR10e arm + Sharpa Wave tactile hand) works out by touch alone which object on a table is the target,
then lifts it. Simulation is MuJoCo; physics, observations and learning run batched on one GPU per experiment, so
many experiments can run in parallel.

Report page (results and interactive architecture): https://claude.ai/artifact/EpoKr3C6nwHUPQXc3pWKA3

## Current state

_Updated after every change or finished experiment. Last update: 2026-11-29._

### Previous work done

- **What:** An "Objects" tab on the report page: one card per YCB object (training split first, then held out) with
  four CPU renders (three-quarter, orthographic side and top, the flat-open Sharpa hand beside it from above), mass,
  extents along the PCA axes of the target cloud and along the scan's own axes, height at rest, and warnings when
  the object was still moving after settling or the scan dips below the table. Scale in every image: 1 cm grid paper
  (bold every 5 cm) on the table and on a backdrop, plus a ruler of 1 cm segments. Renders are cached in
  `runs/dev/objects/` by `tacfind/report/gallery.py` (`python scripts/report.py --objects`).
- **Why:** The user wanted to check every object of the dataset by eye, with its size next to the hand.
- **Results:** Flat-open hand on the model: thumb-to-index pad distance 12.4 cm, 23.4 cm long, 17.8 cm wide. The
  convex collision hulls of the round objects sit inside the scans: at rest the scans of the baseball,
  racquetball, pear, tennis ball and golf ball dip 4-5 mm below the table top (they rest that much lower than
  they look). PCA extents overstate the size of objects without a dominant axis (Rubik's cube 8.2-9.4 cm by PCA,
  5.8 cm along its scan axes; mug). Page size 250 KiB -> 2.0 MiB (84 embedded WebP images, 1.6 MB). Report tests
  22/22 and Node tests 15/15 pass.

### Ongoing

- **What:** Waiting for a free GPU on the shared server. The isolated workspace there is ready: code, Python
  environment and assets in one private directory, 82 unit tests passing on its CPU.
- **Why:** No smoke test or training can start before a GPU is free and the run is approved.
- **Predicted outcomes:** A preflight check, then a smoke test of every stage, which is expected to surface GPU-only
  problems (memory per world, throughput, headless rendering).

### Future works

| Step | What | Waits for |
|---|---|---|
| GPU smoke test | Every stage for a few iterations; measure throughput, memory per world and the cost of the library size | GPU access and approval |
| Baseline E001, stages 1-2 | Contrastive and confidence stages: tactile-point-cloud matcher, 1-D CNN encoder, H = 30, contact-ratio label | smoke test |
| Stages 3-4 | PPO with the sparse reward; which modules train in each stage is decided after stages 1-2 work | stages 1-2 |
| Experiment axes | Matcher input (points / latent / latent header), training scheme, encoder type, H, label type, network scale, attention width | baseline |
| Ideas to test later | Reward shaping if the sparse reward does not converge; task-space arm control; pretraining the matcher on synthetic touch patches; rotation-invariant point features; a tactile deformation map instead of contact points | baseline results |
