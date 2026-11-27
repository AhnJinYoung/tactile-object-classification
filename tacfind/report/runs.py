"""Run directories -> the report's results data: summary, downsampled metrics, latest video per stage, chips, notes.

Reads `runs/<timestamp>_<name>/` defensively: a run may still be writing (partial metrics line, no summary yet) or
come from an older trainer with fewer fields. Nothing here invents a value: a missing field stays missing.
"""

import json
import re
from pathlib import Path

import yaml

MAX_POINTS = 240  # rows kept per run after downsampling (all metrics share the selected rows)
KEY_METRICS = ("rollout/success_rate", "rollout/wrong_pick_rate", "acc/contrastive_top1", "conf/mae",
               "rollout/return")
ABS_PATH = re.compile(r"^(/|~|[A-Za-z]:[\\/])")


def public(value):
    """Strip machine-specific absolute paths from a string value (keeps the last path component)."""
    if isinstance(value, str) and ABS_PATH.match(value):
        return f".../{Path(value).name}"
    return value


def read_metrics(path: Path) -> list[dict]:
    """Rows of metrics.jsonl; a partly written last line (run in progress) or a bad line is skipped."""
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [r for r in rows if isinstance(r, dict)]


def downsample(rows: list[dict], n: int = MAX_POINTS) -> dict:
    """Shared rows for every metric: evenly spaced, plus the first and last row of every stage.

    Returns:
        {"step": [...], "iter": [...], "stage": [...], "metrics": {name: [value | None, ...]}} (4 significant digits).
    """
    if not rows:
        return {"step": [], "iter": [], "stage": [], "metrics": {}}
    keep = set(range(0, len(rows), -(-len(rows) // n))) | {len(rows) - 1}
    for i in range(1, len(rows)):
        if rows[i].get("stage") != rows[i - 1].get("stage"):
            keep |= {i - 1, i}
    picked = [rows[i] for i in sorted(keep)]
    names = sorted({k for r in rows for k, v in r.items() if "/" in k and (v is None or isinstance(v, (int, float)))})
    rnd = lambda v: None if v is None else float(f"{v:.4g}")  # noqa: E731
    return {"step": [r.get("step") for r in picked], "iter": [r.get("iter") for r in picked],
            "stage": [r.get("stage") for r in picked],
            "metrics": {k: [rnd(r.get(k)) if isinstance(r.get(k), (int, float)) else None for r in picked]
                        for k in names}}


def latest_videos(run_dir: Path) -> list[dict]:
    """The newest video of every stage: videos/<stage>_<iter>.mp4 (by iteration number)."""
    best = {}
    for path in sorted((run_dir / "videos").glob("*.mp4")):
        match = re.match(r"^(.*)_(\d+)$", path.stem)
        stage, it = (match.group(1), int(match.group(2))) if match else (path.stem, -1)
        if stage not in best or it > best[stage]["iter"]:
            best[stage] = {"stage": stage, "iter": it, "file": path.name}
    return list(best.values())


def parse_overrides(overrides: list[str], groups: set[str]) -> tuple[str | None, dict, dict]:
    """Command-line overrides -> (experiment preset, {group: option}, {dotted key: value}), parsed as Hydra does."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    preset, chosen, values = None, {}, {}
    for item in overrides or []:
        try:
            o = OverridesParser.create().parse_override(str(item))
        except Exception:  # noqa: BLE001 - an unparsable override is shown nowhere rather than guessed
            continue
        key, value = o.key_or_group, o.value()
        if key.startswith("hydra") or o.is_delete():
            continue
        if key == "experiment":
            preset = value
        elif key in groups:
            chosen[key] = value
        else:
            values[key] = public(value)
    return preset, chosen, values


def changes(choices: dict, values: dict, defaults: list[list[str]]) -> list[str]:
    """Chips: group options that differ from the root defaults, then changed keys (name/description excluded)."""
    base = dict(defaults)
    chips = [f"{g}={o}" for g, o in choices.items() if g in base and o != base[g]]
    return chips + [f"{k}={json.dumps(v) if isinstance(v, (list, dict)) else v}" for k, v in values.items()
                    if k not in ("name", "description")]


def collect_run(run_dir: Path, sources: dict, notes: dict) -> dict:
    """One run's page data (also used for the architecture dropdown via `arch`)."""
    summary = {}
    if (run_dir / "summary.json").is_file():
        try:
            summary = json.loads((run_dir / "summary.json").read_text())
        except json.JSONDecodeError:
            summary = {}
    rows = read_metrics(run_dir / "metrics.jsonl")
    preset, chosen, values = parse_overrides(summary.get("overrides", []), set(sources["groups"]))
    choices = {g: o for g, o in (summary.get("choices") or {}).items() if g in sources["groups"] and o}
    preset = preset or (summary.get("choices") or {}).get("experiment")
    preset_values = sources["presets"].get(preset, {}).get("values", {})
    name = summary.get("name") or run_dir.name.partition("_")[2] or run_dir.name
    latest = {}
    for r in rows:
        latest.update({k: v for k, v in r.items() if k in KEY_METRICS and v is not None})
    return {
        "id": run_dir.name, "name": name, "status": summary.get("status", "unknown"),
        "description": summary.get("description", ""), "preset": preset,
        "start_time": summary.get("start_time"), "end_time": summary.get("end_time"),
        "gpu": summary.get("gpu_name") or summary.get("device"), "n_envs": summary.get("n_envs"),
        "params": summary.get("params"), "env_steps": summary.get("env_steps", rows[-1].get("step") if rows else None),
        "peak_gpu_mem_gb": summary.get("peak_gpu_mem_gb"),
        "key_metrics": latest, "curves": downsample(rows), "videos": latest_videos(run_dir),
        "chips": changes(choices, preset_values | values, sources["defaults"]),
        "note": notes.get(run_dir.name) or notes.get(name) or "",
        # a path stripped by `public` cannot be reproduced elsewhere: left out of the architecture tab's export
        "arch": {"choices": choices, "values": {k: v for k, v in (preset_values | values).items()
                                                 if not (isinstance(v, str) and v.startswith(".../"))}},
    }


def collect_runs(runs_dir: Path, sources: dict, notes: dict) -> list[dict]:
    """Every run directory under `runs_dir` (those with a summary or metrics), oldest first; `dev` is skipped."""
    dirs = sorted(p for p in runs_dir.glob("*") if p.is_dir() and p.name != "dev"
                  and ((p / "summary.json").is_file() or (p / "metrics.jsonl").is_file()))
    return [collect_run(p, sources, notes) for p in dirs]


def planned(sources: dict, runs: list[dict], notes: dict) -> list[dict]:
    """Presets without any run yet."""
    done = {r["preset"] for r in runs} | {r["name"] for r in runs}
    out = []
    for key, p in sources["presets"].items():
        if key not in done and p["name"] not in done:
            out.append({"id": key, "name": p["name"], "status": "planned", "description": p["description"],
                        "chips": changes(p["groups"], p["values"], sources["defaults"]),
                        "note": notes.get(key, "")})
    return out


def load_notes(path: Path) -> dict:
    """notes.yaml: {experiment or run name: comment}; empty when missing."""
    if not path.is_file():
        return {}
    return {str(k): str(v) for k, v in (yaml.safe_load(path.read_text()) or {}).items() if v}

