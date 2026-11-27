"""Reference data for the JavaScript tests: `python -m tacfind.report [configs_dir]` prints JSON with the config
sources, the module specs, and for every preset Hydra's own composition, its group choices and the symbol values."""

import json
import sys
from pathlib import Path

from tacfind.report.configs import CONFIGS, compose_preset, config_sources
from tacfind.report.specs import load_specs, symbol_values

if __name__ == "__main__":
    configs = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIGS
    specs, sources, presets = load_specs(), config_sources(configs), {}
    for name in sources["presets"]:
        cfg, choices = compose_preset(name, configs)
        presets[name] = {"cfg": cfg, "choices": choices, "symbols": symbol_values(cfg, choices, specs)}
    json.dump({"sources": sources, "specs": specs, "presets": presets}, sys.stdout)
