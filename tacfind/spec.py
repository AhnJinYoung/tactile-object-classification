"""Tensor sizes the environment promises to the networks.

This is the only thing `tacfind.networks` needs to know about `tacfind.envs`, so the two never import each other.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSpec:
    """Sizes of one environment step.

    Attributes:
        obs_dim: size D_o of the per-step observation o_t (proprioception + tactile, normalised).
        act_dim: action size (28 = 6 arm + 22 hand joint-target increments).
        priv_dim: size of the privileged simulator state that only the critic sees (0 = none).
        n_tips: number of fingertip tactile sensors (5 on a Sharpa Wave).
        pt_dim: features of one fingertip contact point: position in the robot base frame (3, m), outward normal of
            the touched surface (3, unit), contact force magnitude (1, N).
    """

    obs_dim: int
    act_dim: int = 28
    priv_dim: int = 0
    n_tips: int = 5
    pt_dim: int = 7
