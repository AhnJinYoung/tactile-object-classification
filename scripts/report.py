"""Build the report page: docs/report/build/index.html (+ videos/), from runs/, configs/ and docs/report/notes.yaml.

    python scripts/report.py                 # then publish docs/report/build/index.html with its videos
    python scripts/report.py --objects       # first render the object gallery into runs/dev/objects/ (CPU, minutes)
"""

import argparse
import os
from pathlib import Path

from tacfind.report.page import build

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=REPO / "runs", help="directory of run directories")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "report" / "build", help="output directory")
    parser.add_argument("--notes", type=Path, default=REPO / "docs" / "report" / "notes.yaml", help="comments file")
    parser.add_argument("--objects", action="store_true", help="render the object gallery first (CPU MuJoCo, EGL)")
    args = parser.parse_args()
    if args.objects:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from omegaconf import OmegaConf  # imported here: rendering loads the simulator

        from tacfind.envs.bin_env import plain
        from tacfind.report.gallery import render_gallery
        render_gallery(plain(OmegaConf.load(REPO / "configs" / "env" / "bin.yaml")), args.runs / "dev" / "objects")
    page, data, copied = build(args.runs, args.out, args.notes)
    rel = lambda p: p.resolve().relative_to(REPO) if p.resolve().is_relative_to(REPO) else p  # noqa: E731
    print(f"page: {rel(page)} ({page.stat().st_size / 1024:.0f} KiB)")
    print(f"runs: {len(data['runs'])} ({', '.join(r['id'] for r in data['runs']) or 'none yet: the page shows the empty state'})")
    print(f"planned presets without a run: {len(data['planned'])}")
    print(f"robot frame embedded: {'yes' if data['robotImage'] else 'no'}")
    gallery = data["gallery"]
    print(f"object gallery: {len(gallery['objects'])} objects" if gallery
          else "object gallery: not rendered (run with --objects)")
    print(f"videos ({len(copied)}), publish each next to the page at the path on the left:")
    for published, path in copied.items():
        print(f"  {published} <- {rel(path)}")


if __name__ == "__main__":
    main()
