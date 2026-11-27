# tactile-object-classification

A blind robot (UR10e arm + Sharpa Wave tactile hand) reaches into an occluded bin, works out by touch alone which
object is the target, and pulls it out. Simulation is MuJoCo; physics, observations and learning run batched on one
GPU per experiment, so many experiments can run in parallel.

## Current state

_Updated after every change or finished experiment. Last update: 2026-11-27._

### Previous work done

- **What:** The GPU-batched environment (`tacfind/envs/`): MuJoCo Warp physics with every world on one device and a
  plain-MuJoCo reference backend for local tests; one model that holds the whole object library, with the objects
  not in play parked out of the way; resets from a bank of settled bin states; batched fingertip tactile readings;
  the confidence labels (contact ratio and binary); a scripted touch explorer; rendering of one world. In the same
  step the networks got two fixes: a memory-lean attention with a cap on touch tokens, and a uniformity loss.
- **Why:** Experiments have to run in parallel on NVIDIA GPUs, one experiment per GPU, so physics, observations and
  labels must all be tensors on that device. The first network version needed too much memory for thousands of
  worlds, and its object embeddings collapsed under plain contrastive training.
- **Results:** 16 environment tests and 44 network tests pass on CPU. MuJoCo Warp (on its CPU device) matches CPU
  MuJoCo to about 4e-6 rad over a short rollout. The observation has 192 values per step. With the scripted
  explorer, fingertips are in contact in 12.6 % of steps and touch 2.25 distinct objects per episode; the
  contact-ratio label averages 0.21 while in contact and peaks at 0.50. The explorer never managed to lift an object
  out of the bin. Matcher memory in the worst case fell from 52.9 MB to 2.0 MB per sample. On a synthetic check
  (3 seeds) the uniformity loss keeps the object embeddings apart and gives 0.59 top-1 accuracy against 0.36 without
  it (chance 0.25); the lean attention learns more slowly than the original (0.59 against 0.75). Nothing has run on
  a GPU yet.

### Ongoing

- **What:** (1) The trainer: configurable stages (contrastive, confidence, PPO on the encoder, PPO on the policy),
  PPO, run directories with intermediate videos and curves, optional Weights & Biases, and `scripts/launch.py`,
  which fills the available GPUs with queued runs. (2) The live viewer: MuJoCo Studio in the browser, served by the
  simulation process and reached through an SSH tunnel. (3) A grasp feasibility check: can the simulated hand lift
  each object family at all, with the official finger gains and with torque-limit gains.
- **Why:** The trainer and launcher are what the planned experiments run on. The viewer lets a person watch and
  drive the simulation on the GPU server. The feasibility check comes before any GPU time is spent on
  reinforcement learning, because the explorer could not hold a single object.
- **Predicted outcomes:** Every stage runs for a few iterations in unit tests on CPU, and a sweep file starts one
  run per GPU. The viewer's server side works here; its browser side still has to be checked by a person. The
  feasibility check either shows that lifting works with stronger finger gains or names what has to change
  (friction, mass, size range, posture).

### Future works

| Step | What | Waits for |
|---|---|---|
| Report: results tab | One page collecting every run: comment, rollout video, metric curves | trainer |
| Report: architecture tab | The architecture drawn in the style of the original figure, updated with the decisions made; a dropdown picks the experiment; hyperparameters are editable inside the blocks and export as a config YAML; clicking a module shows its input shapes and its computation step by step | trainer (experiment presets) |
| GPU smoke test | Every stage for a few iterations; measure throughput, memory per world and the cost of the library size | GPU access and approval |
| Baseline E001 | Staged training, tactile-point-cloud matcher, 1-D CNN encoder, H = 30, contact-ratio label | smoke test |
| Experiment axes | Matcher input (points / latent / latent header), training scheme, encoder type, H, label type, network scale, attention width | baseline |
| Ideas to test later | Pretraining the matcher on synthetic touch patches without physics; rotation-invariant point features; a tactile deformation map instead of contact points | baseline results |
