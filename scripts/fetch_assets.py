"""Download the robot models the MuJoCo scene is built from into the git-ignored `third_party/`.

Run once before using `tacfind.envs`: `python scripts/fetch_assets.py`. Idempotent: a repository that is already at
its pinned commit is left alone. Only the needed files are fetched (shallow, sparse, blob-filtered clones).
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party"

# name: (repository, pinned commit, sparse-checkout patterns, license)
ASSETS = {
    "mujoco_menagerie": (
        "https://github.com/google-deepmind/mujoco_menagerie",
        "eea4c7431b63da8278b0ff31638e0b0e941cec0e",
        ["/universal_robots_ur10e/*.xml", "/universal_robots_ur10e/LICENSE", "/universal_robots_ur10e/assets/"],
        "BSD-3-Clause",
    ),
    "sharpa": (
        "https://github.com/sharpa-robotics/sharpa-urdf-usd-xml",
        "87de65dc893569ea638a69853ef6340008898eb1",
        ["/wave_01/right_sharpa_wave/*.xml", "/wave_01/right_sharpa_wave/meshes/", "/LICENSE*"],  # no USD files
        "Apache-2.0",
    ),
}


def git(*args: str, cwd: Path, check: bool = True) -> str:
    """Run a git command in `cwd` and return its stdout."""
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True).stdout.strip()


def fetch(name: str, url: str, sha: str, patterns: list[str]) -> None:
    """Sparse-checkout `patterns` of `url` at commit `sha` into `third_party/<name>` (shallow, blob-filtered)."""
    dest = ROOT / name
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        git("init", "-q", cwd=dest)
        git("remote", "add", "origin", url, cwd=dest)
    git("sparse-checkout", "set", "--no-cone", *patterns, cwd=dest)
    if git("rev-parse", "--verify", "-q", "HEAD", cwd=dest, check=False) == sha:
        print(f"{name}: up to date at {sha[:10]}")
        return
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha, cwd=dest)
    git("checkout", "-q", "--force", sha, cwd=dest)
    print(f"{name}: checked out at {sha[:10]}")


if __name__ == "__main__":
    for name, (url, sha, patterns, license_) in ASSETS.items():
        fetch(name, url, sha, patterns)
        print(f"  license: {license_} (see {url})")
