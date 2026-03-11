"""Gymnasium environments for reinforcement learning with ANDES."""

try:
    import gymnasium  # noqa: F401
except ImportError:
    raise ImportError(
        "The andes.rl module requires gymnasium. "
        "Install it with: pip install andes[rl]"
    )

from andes.rl.env import AndesEnv  # noqa: F401

__all__ = ['AndesEnv']
