# tactile-object-classification

A blind robot (UR10e arm + Sharpa Wave tactile hand) reaches into an occluded bin, works out by touch alone which
object is the target, and pulls it out. Simulation is MuJoCo; physics, observations and learning run batched on one
GPU per experiment, so many experiments can run in parallel.

## Current state

_Updated after every change or finished experiment. Last update: 2026-11-27._

### Previous work done

- **What:** First pass of the bin scene on CPU MuJoCo: the UR10e with the official Sharpa Wave model mounted on its
  flange, a table and an open-top bin, a seeded library of 8 object shape families with surface point clouds, and a
  per-step observation of 192 values.
- **Why:** To check that the two robot models combine into one working scene, and to measure how fast CPU physics is
  before deciding how to scale.
- **Results:** The scene builds and renders. At the home pose the fingers point straight down with the fingertips
  about 8 cm above the bin rim. CPU physics reaches about 110 control steps/s per environment and about 600
  environment steps/s on 8 threads of the development machine. That is too slow for parallel experiments and settled
  the move to GPU physics. The stepping code from this pass is being replaced and was not committed.

### Ongoing

- **What:** (1) Rewriting the environment as batched tensor code: MuJoCo Warp physics with all worlds on one GPU, a
  CPU reference backend for local unit tests, tactile readings, labels, resets from a bank of settled states, and a
  scripted touch explorer. (2) The network modules: three spatial-temporal encoders, the Point Transformer target
  encoder, the cross-modal matcher with three touch-input modes, the actor-critic, and the losses.
- **Why:** Experiments have to run in parallel on NVIDIA GPUs (A100, RTX A6000, H100), one experiment per GPU, so
  nothing in the step path may run on the CPU. The network variants are the axes of the planned experiments.
- **Predicted outcomes:** One experiment runs end to end on a single GPU with thousands of simulated worlds, and the
  unit tests pass locally on CPU. Throughput and GPU memory per world are unknown until the GPU smoke test.

### Future works

| Step | What | Waits for |
|---|---|---|
| Trainer | Staged training (contrastive, then confidence, then PPO on the encoder, then PPO on the policy) with intermediate videos and curves, optional Weights & Biases | environment, networks |
| Launcher | `scripts/launch.py` fills all GPUs with queued runs | trainer |
| Report | One page collecting every run: comment, rollout video, metric curves | trainer |
| GPU smoke test | Every stage for a few iterations; measure throughput and memory | GPU access and approval |
| Baseline E001 | Staged training, tactile-point-cloud matcher, 1-D CNN encoder, H = 30, contact-ratio label | smoke test |
| Experiment axes | Matcher input (points / latent / latent header), training scheme, encoder type, H, label type, network scale | baseline |
