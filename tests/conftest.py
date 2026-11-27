"""Shared fixtures: the env config with a small, fast setup, and skips when the robot assets are missing."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: MuJoCo Warp tests on its CPU device (minutes when kernels are uncached)")


@pytest.fixture
def cfg():
    """`env` config on the CPU reference backend with a small settled-state bank."""
    if not all((REPO / "third_party" / d).is_dir() for d in ("sharpa", "mujoco_menagerie", "ycb")):
        pytest.skip("robot / object assets missing: run scripts/fetch_assets.py")
    c = OmegaConf.load(REPO / "configs" / "env" / "bin.yaml")
    c.sim, c.bank_size = "cpu", 8
    return c
