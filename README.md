# tactile-object-classification

A blind robot (UR10e arm + Sharpa Wave tactile hand) works out by touch alone which object on a table is the target,
then lifts it. Simulation is MuJoCo; physics, observations and learning run batched on one GPU per experiment, so
many experiments can run in parallel.

Report page (results and interactive architecture): https://claude.ai/artifact/EpoKr3C6nwHUPQXc3pWKA3

## Current state

_Updated after every change or finished experiment. Last update: 2026-11-28._

### Previous work done

- **What:** The hand now faces a long edge of the table: at home the fingers point down and the palm faces -x, away
  from the robot (before: +y, along the table). Only wrist 3 turns (about -pi/2; the other joints move a little to
  keep the tcp at the same point) and the hand mount stays as designed. The explorer and the lift check express
  their hand-relative offsets and the palm-down tilt axis in a palm frame taken from the home pose, so `arm_home`
  alone sets the palm direction. `arm_range` grew for shoulder lift, elbow and wrist 1 (1.0, 1.4, 1.4 -> 1.4, 2.0,
  2.0 rad) so the tilted grasp pose reaches the zone.
- **Why:** The user asked for the palm to face a long edge, if the real hardware allows it. It does: the Sharpa
  flange (ISO 9409-1-50-4-M6 pattern with one locating-pin hole) fixes the hand's yaw on the UR tool flange, and
  wrist 3 (+-360 deg) turns the whole hand. Palm towards the robot was tried first and rejected: the palm-down pull
  pose is then out of reach for most of the zone (91 % of the tilted IK lattice with the old range).
- **Results:** Scripted single-object lift check with the official Sharpa finger gains: 51 of 63 trials (before:
  23 of 63; the larger range alone, palm along the table, still gives 23 of 63); palm towards the robot: 35 of 63.
  Scripted explorer, 8 worlds x 300 steps on CPU: fingertip contact in 14.8 % of world-steps (palm along the table,
  same measurement: 18.9 %), 1 wrong pick, no success. Unit tests: 101 of 102 pass on CPU; the preflight test
  fails only because the disk has 3.6 GB free (its runs-directory check). Nothing has run on a GPU yet.

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
