# tactile-object-classification

A blind robot (UR10e arm + Sharpa Wave tactile hand) reaches into an occluded bin, works out by touch alone which
object is the target, and pulls it out. Simulation is MuJoCo; physics, observations and learning run batched on one
GPU per experiment, so many experiments can run in parallel.

## Current state

_Updated after every change or finished experiment. Last update: 2026-11-27._

### Previous work done

- **What:** The network modules, one file each in `tacfind/networks/`: three interchangeable spatial-temporal
  encoders (MLP, 1-D CNN, Transformer), the Point Transformer target encoder, the cross-modal matcher with three
  touch-input modes (tactile points, latent, latent with a positional header), the actor-critic, and `model.py`,
  the single aggregate that wires them. Losses (contrastive, confidence, Chamfer) are in `tacfind/train/losses.py`.
- **Why:** These are the parts the planned experiments swap, so each must be selectable by a config group and
  testable without a simulator.
- **Results:** 41 unit tests pass on CPU in about 25 s. Parameter counts are 0.5 M (small), 1.7 M (base) and 7.5 M
  (large). A learnability check on synthetic surface patches under random rotation lifts contrastive top-1 accuracy
  from 0.17 to 0.76 (chance is 0.25). Two problems surfaced. First, when the target encoder and the touch side learn
  at the same rate, the object embeddings collapse onto one vector and accuracy stays at chance; the check passes
  only with a 10x lower target-encoder learning rate. Second, the matcher's cross-attention needs about 5 MB per
  sample per intermediate tensor in the worst case, too much for thousands of simulated worlds on one GPU.

### Ongoing

- **What:** (1) Rewriting the environment as batched tensor code: MuJoCo Warp physics with all worlds on one GPU, a
  CPU reference backend for local unit tests, tactile readings, labels, resets from a bank of settled states, and a
  scripted touch explorer. (2) Fixing the two network problems: capping the touch tokens and making the
  cross-attention memory-lean, and comparing four remedies for the embedding collapse.
- **Why:** Experiments have to run in parallel on NVIDIA GPUs (A100, RTX A6000, H100), one experiment per GPU, so
  nothing in the step path may run on the CPU and the networks must fit thousands of worlds in GPU memory.
- **Predicted outcomes:** One experiment runs end to end on a single GPU with thousands of simulated worlds, the
  matcher needs about 2 MB per sample or less, and one collapse remedy is chosen as the trainer default with
  evidence. Throughput and GPU memory per world are unknown until the GPU smoke test.

### Future works

| Step | What | Waits for |
|---|---|---|
| Trainer | Staged training (contrastive, then confidence, then PPO on the encoder, then PPO on the policy) with intermediate videos and curves, optional Weights & Biases | environment, networks |
| Launcher | `scripts/launch.py` fills all GPUs with queued runs | trainer |
| Report | One page collecting every run: comment, rollout video, metric curves | trainer |
| Live viewer | MuJoCo Studio in the browser, served by the process on the GPU server and reached through an SSH tunnel: pick a world, pause / step / reset, perturb bodies, drive the hand, see contacts and confidence | environment |
| GPU smoke test | Every stage for a few iterations; measure throughput and memory | GPU access and approval |
| Baseline E001 | Staged training, tactile-point-cloud matcher, 1-D CNN encoder, H = 30, contact-ratio label | smoke test |
| Experiment axes | Matcher input (points / latent / latent header), training scheme, encoder type, H, label type, network scale | baseline |
