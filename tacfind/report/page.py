"""Builds the report page: collects runs, configs and module specs into one JSON blob and inlines it, the stylesheet
and the scripts of `docs/report/src/` into a single HTML file (opens offline; videos sit next to it)."""

import ast
import base64
import json
import shutil
from datetime import date
from pathlib import Path

from tacfind.report.configs import CONFIGS, REPO, config_sources
from tacfind.report.runs import collect_runs, load_notes, planned
from tacfind.report.specs import load_specs

SRC = REPO / "docs" / "report" / "src"
SCRIPTS = ("symbols.js", "compose.js", "charts.js", "diagram.js", "scene.js", "results.js", "arch.js", "app.js")  # load order
ROBOT_FRAME = "table_sanity_150.png"  # a rendered frame of our scene: arm, hand and objects on the table
ENV_VIDEO = "table_sanity.mp4"  # scripted explorer, no learning (environment check)
YCB_FILE = REPO / "tacfind" / "envs" / "ycb.py"


def page_data(runs_dir: Path, notes_path: Path, configs: Path = CONFIGS) -> dict:
    """Everything the page shows. Video paths are relative to the page (`videos/<run>/<file>`)."""
    sources, notes = config_sources(configs), load_notes(notes_path)
    runs = collect_runs(runs_dir, sources, notes)
    dev = runs_dir / "dev"
    frame = dev / ROBOT_FRAME if (dev / ROBOT_FRAME).is_file() else next(iter(sorted(dev.glob("table_sanity_*.png"))), None)
    return {
        "generated": date.today().isoformat(),
        "runs": runs, "planned": planned(sources, runs, notes), "config": sources, "specs": load_specs(),
        "envCheck": {"video": f"videos/dev/{ENV_VIDEO}"} if (dev / ENV_VIDEO).is_file() else None,
        "ycb": ycb_objects(),
        "robotImage": f"data:image/png;base64,{base64.b64encode(frame.read_bytes()).decode()}" if frame else None,
    }


def ycb_objects(path: Path = YCB_FILE) -> dict:
    """{name: [mass kg, material, split]} of the YCB library, read from the source of `tacfind.envs.ycb` without
    importing it (that would load the simulator)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "OBJECTS" for t in node.targets):
            return {k: list(v) for k, v in ast.literal_eval(node.value).items()}
    return {}


def videos_to_copy(data: dict, runs_dir: Path) -> dict[str, Path]:
    """{published path: source file} of every video the page references."""
    out = {f"videos/{r['id']}/{v['file']}": runs_dir / r["id"] / "videos" / v["file"] for r in data["runs"]
           for v in r["videos"]}
    if data["envCheck"]:
        out[data["envCheck"]["video"]] = runs_dir / "dev" / ENV_VIDEO
    return out


def render_html(data: dict, src: Path = SRC) -> str:
    """The single-file page."""
    blob = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    scripts = "\n".join((src / name).read_text() for name in SCRIPTS)
    html = (src / "page.html").read_text()
    for key, value in (("STYLE", (src / "style.css").read_text()), ("DATA", blob), ("SCRIPTS", scripts)):
        html = html.replace(f"{{{{{key}}}}}", value, 1)
    return html


def build(runs_dir: Path, out_dir: Path, notes_path: Path, configs: Path = CONFIGS) -> tuple[Path, dict, dict]:
    """Write out_dir/index.html and copy the videos to out_dir/videos/.

    Returns:
        (page path, page data, {published path: copied file}).
    """
    data = page_data(runs_dir, notes_path, configs)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "videos").is_dir():
        shutil.rmtree(out_dir / "videos")
    copied = {}
    for rel, source in videos_to_copy(data, runs_dir).items():
        if source.is_file():
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, out_dir / rel)
            copied[rel] = out_dir / rel
    page = out_dir / "index.html"
    page.write_text(render_html(data))
    return page, data, copied
