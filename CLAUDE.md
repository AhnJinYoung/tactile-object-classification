# CLAUDE.md

Project: **tactile target finding**. A blind UR10e arm with a Sharpa Wave tactile hand finds a target object among
distractors in an occluded bin, by touch only, and pulls it out. `README.md` has the architecture, the config system
and the experiment plan.

## Rules from the user (must follow)

1. **English only**: code, comments, configs, docs, commit messages, reports.
2. **Ask before running experiments.** On a machine without a GPU only unit tests run. Smoke tests and trainings run
   only on GPU resources the user provides, and only after explicit approval.
3. **If anything is unclear or needs a design choice, ask the user.** Never pick silently.
4. **Roles.** The main session orchestrates: planning, ideation, analysis of results. Subagents on Opus 5.5 at medium
   effort do the subtasks: coding, writing artifacts, search.
5. **Every training emits intermediate visualisations** (rollout videos, curves) so progress can be watched while it
   runs; analytics go to the run directory, and to Weights & Biases when enabled.
6. **Experiments run in parallel on NVIDIA GPUs** (A100, RTX A6000, H100). One experiment lives entirely on one GPU:
   batched physics (MuJoCo Warp), observations, labels, rewards and learning are all tensors on that device, with no
   per-environment Python loops and no CPU round trips in the step path. `scripts/launch.py` fills the available GPUs
   with runs. Never assume a GPU model or memory size; sizes come from the config.

## Code conventions

- Compact and readable: if 100 lines can be 50 clear lines, write the 50. No speculative abstraction, no dead options.
- One file = one concern, under ~300 lines; functions under ~50 lines.
- `tacfind/networks/` holds one file per network module; `model.py` is the single aggregate that wires them together.
- Experiments are configs (`configs/`, Hydra), never new scripts: an architecture choice is a config group, a
  hyperparameter is a command-line override.
- Every module starts with a short docstring: what it is for and where it sits in the pipeline. Public functions and
  classes have Google-style docstrings (`Args:` / `Returns:`) that give tensor shapes and SI units.
- Comments explain why, not what. Frames are named in variable names (`pos_base`, `normal_tip`); tensor shapes are
  noted where they matter: `# (B, H, D)`.
- PEP 8, type hints on signatures, f-strings, `pathlib.Path`, isort import order, `if __name__ == "__main__":` in
  scripts.

## Runs

Every run writes only to `runs/<YYYYMMDD-HHMMSS>_<name>/` (resolved config, git commit, metrics, videos, figures,
checkpoints, a short note) and appends one row to `runs/index.csv`. The report in `docs/report/` is generated from
those files; nothing in it is typed by hand except the comments.

## Git

- Remote: `github.com/AhnJinYoung/tactile-object-classification` (public). Commit and push every change.
- `main` holds the framework and the baseline. An architectural experiment (a module changes, not just a
  hyperparameter) gets its own branch `exp/<slug>`; tuning, minor changes and fixes are linear commits on that branch.
- Never commit secrets (tokens live in the git-ignored `.env`), data, runs, checkpoints or third-party assets.
