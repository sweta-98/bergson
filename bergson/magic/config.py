from dataclasses import dataclass
from typing import Literal

from ..config import ValidationConfig

MagicSaveMode = Literal["all", "sqrt", "log"]


@dataclass
class MagicConfig(ValidationConfig):
    """Special config for MAGIC attribution."""


    """Query/eval dataset for computing attribution target gradients.
    If not specified, defaults to the training dataset."""

    token_batch_size: int = 1024
    
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

    per_token: bool = False
    """Whether to compute attribution scores per token (instead of per sequence)."""
