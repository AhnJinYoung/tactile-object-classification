"""Config sources for the report page: the root config, every group option and every experiment preset, unresolved.

The page re-implements Hydra's composition for this repository in JavaScript (docs/report/src/compose.js) from this
data; `compose_preset` is the real Hydra composition the tests compare it with.
"""

from pathlib import Path

from omegaconf import OmegaConf

from tacfind.report.specs import lookup

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
SKIP_GROUPS = {"experiment", "sweep"}  # presets are read separately; sweeps are not part of a run's config


def is_global(path: Path) -> bool:
    """True when the file declares `# @package _global_` (merged at the root instead of under its group)."""
    return any(line.strip().replace(" ", "") == "#@package_global_" for line in path.read_text().splitlines()[:5])


def load_yaml(path: Path) -> dict:
    """A YAML file as plain containers, interpolations kept as strings (OmegaConf's loader, as Hydra uses)."""
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False) or {}


def parse_defaults(entries: list) -> tuple[list[list[str]], dict[str, str]]:
    """Split a defaults list into ([group, option] in order) and {group: option} of `override /group: option`."""
    order, overrides = [], {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue  # _self_
        (key, value), = entry.items()
        if key.startswith("override /"):
            overrides[key.removeprefix("override /")] = value
        elif not key.startswith("override"):
            order.append([key, value])
    return order, overrides


def flatten(tree: dict, prefix: str = "") -> dict:
    """Leaf paths of nested dicts: {"env": {"label": "binary"}} -> {"env.label": "binary"}; lists are leaves."""
    out = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out |= flatten(value, f"{path}.")
        else:
            out[path] = value
    return out


def config_sources(configs: Path = CONFIGS) -> dict:
    """Everything the page needs to compose any experiment.

    Returns:
        dict with root (root config without defaults and hydra), defaults ([group, option] in merge order),
        groups ({group: {global, options: {option: content}}}) and presets ({name: {description, groups, values}};
        values are flattened leaf paths).
    """
    root = load_yaml(configs / "config.yaml")
    defaults, _ = parse_defaults(root.pop("defaults", []))
    root.pop("hydra", None)
    groups = {}
    for folder in sorted(p for p in configs.iterdir() if p.is_dir() and p.name not in SKIP_GROUPS):
        files = sorted(folder.glob("*.yaml"))
        groups[folder.name] = {"global": any(is_global(f) for f in files),
                               "options": {f.stem: load_yaml(f) for f in files}}
    presets = {}
    for path in sorted((configs / "experiment").glob("*.yaml")):
        content = load_yaml(path)
        _, chosen = parse_defaults(content.pop("defaults", []))
        presets[path.stem] = {"name": content.pop("name", path.stem), "description": content.pop("description", ""),
                              "groups": chosen, "values": flatten(content)}
    return {"root": root, "defaults": defaults, "groups": groups, "presets": presets}


def compose_preset(preset: str | None, configs: Path = CONFIGS, overrides: list[str] | None = None) -> tuple:
    """Compose with Hydra itself: root config + `+experiment=<preset>` + overrides.

    Returns:
        (resolved config as plain dicts without the hydra node, {group: selected option}).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    args = ([f"+experiment={preset}"] if preset else []) + list(overrides or [])
    with initialize_config_dir(config_dir=str(configs.resolve()), version_base="1.3"):
        cfg = compose("config", overrides=args, return_hydra_config=True)
    choices = {k: v for k, v in OmegaConf.to_container(cfg.hydra.runtime.choices).items()
               if not k.startswith("hydra")}
    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.pop("hydra")
    return resolved, choices


def get(cfg: dict, path: str, default=None):
    """`lookup` with a default."""
    try:
        return lookup(cfg, path)
    except (KeyError, TypeError):
        return default
