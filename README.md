# tactile-object-classification

A blind robot (UR10e arm + Sharpa Wave tactile hand) works out by touch alone which object on a table is the target,
then lifts it. Simulation is MuJoCo; physics, observations and learning run batched on one GPU per experiment, so
many experiments can run in parallel.

Report page (results and interactive architecture): https://claude.ai/artifact/EpoKr3C6nwHUPQXc3pWKA3

## Current state

_Updated after every change or finished experiment. Last update: 2026-11-28._

### Previous work done

- **What:** The scene was rebuilt: no bin, an 80 x 180 cm table with the robot at the middle of a long edge, 3 to 8
  objects scattered in the reachable 45 x 60 cm zone, and real YCB object meshes (21 objects, 16 for training and
  5 held out) with measured masses, friction per material and per-world randomisation of mass, friction and centre of
  mass. Success is lifting the target 10 cm and holding it 1 s; the reward is sparse by default. Merged in the same
  step: the staged trainer, the multi-GPU launcher, the server preflight check, the live browser viewer and the
  report page source.
- **Why:** The task is searching a table, not a bin, and realistic object masses and contact parameters narrow the
  gap to the real robot. The trainer and launcher are what the planned experiments run on.
- **Results:** 100 unit tests pass on CPU. A scripted lift check on each object alone succeeded in 23 of 63 trials
  with the official Sharpa finger gains and 15 of 63 with torque-limit gains; fingertip press force reaches 14 N at
  moderate flexion, and the heaviest object (171 g) was held, so failures come from the fixed scripted grasp, not
  from weak fingers. The scripted explorer has fingertip contact in 17 % of steps and almost never lifts an object
  in clutter. Nothing has run on a GPU yet.

### Ongoing

- **What:** Bringing the report page in line with the table scene, and getting access to the GPU servers.
- **Why:** The page still described the bin. No training can start before a GPU is reachable and free.
- **Predicted outcomes:** The page shows the new scene and still no results. Once a GPU is available: a preflight
  check, then a smoke test of every stage, which is expected to surface GPU-only problems.

### Future works

| Step | What | Waits for |
|---|---|---|
| GPU smoke test | Every stage for a few iterations; measure throughput, memory per world and the cost of the library size | GPU access and approval |
| Baseline E001, stages 1-2 | Contrastive and confidence stages: tactile-point-cloud matcher, 1-D CNN encoder, H = 30, contact-ratio label | smoke test |
| Stages 3-4 | PPO with the sparse reward; which modules train in each stage is decided after stages 1-2 work | stages 1-2 |
| Experiment axes | Matcher input (points / latent / latent header), training scheme, encoder type, H, label type, network scale, attention width | baseline |
| Ideas to test later | Reward shaping if the sparse reward does not converge; task-space arm control; pretraining the matcher on synthetic touch patches; rotation-invariant point features; a tactile deformation map instead of contact points | baseline results |
