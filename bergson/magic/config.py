from dataclasses import dataclass
from typing import Literal

from ..config import ValidationConfig

MagicSaveMode = Literal["all", "sqrt", "log"]


@dataclass
class MagicConfig(ValidationConfig):
    """Special config for MAGIC attribution."""

    save_mode: MagicSaveMode = "sqrt"
    """Checkpoint saving mode.

    - 'all' saves every checkpoint. This method uses O(N) space and O(N) time.
    - 'log' saves at a log-spaced interval, more frequently near the end of a training
      segment. Training is recursively divided into segments. This method uses O(log N)
      space and O(N log N) time.
    - 'sqrt' saves at a linearly-spaced interval, every sqrt(N) steps. This method uses
      O(sqrt N) space and O(N) time.

    The original MAGIC paper used 'log', but 'sqrt' is often a better choice when disk
    space is not a concern.
    """

    backward_save_every: int = 0
    """How often (in steps) to save backward state for resume."""

    cleanup_ckpts: bool = True
    """Whether to delete all but the last checkpoint during the backward pass."""

    per_token: bool = False
    """Whether to compute attribution scores per token (instead of per sequence)."""
