import torch

from bergson.process_grads import get_trackstar_preconditioner
from bergson.score.score_writer import ScoreWriter


class Scorer:
    """
    Scores training gradients against query gradients.

    Handles all preconditioning internally:
      - Loads preconditioner from disk if ``preconditioner_path`` is given.
      - Applies to query grads once at init time.
      - Applies to index grads per-batch in :meth:`score` (split mode only).

    Accepts a ScoreWriter for saving the scores (disk or in-memory).
    """

    def __init__(
        self,
        query_grads: dict[str, torch.Tensor],
        modules: list[str],
        writer: ScoreWriter,
        device: torch.device,
        dtype: torch.dtype,
        *,
        unit_normalize: bool = False,
        score_mode: str = "individual",
        attribute_tokens: bool = False,
        preconditioner_path: str | None = None,
    ):
        """
        Initialize the scorer.

        Parameters
        ----------
        query_grads : dict[str, torch.Tensor]
            Query gradients keyed by module name.
        modules : list[str]
            List of module names to use for scoring.
        writer : ScoreWriter
            Writer for score output (InMemoryScoreWriter or MemmapScoreWriter).
        device : torch.device
            Device to perform scoring on.
        dtype : torch.dtype
            Dtype for scoring computation.
        unit_normalize : bool
            Whether to unit normalize gradients before scoring.
        score_mode : str
            Scoring mode: "individual" or "nearest".
        attribute_tokens : bool
            Whether gradients are per-token (rows = total_valid tokens).
        preconditioner_path : str | None
            Path to a saved GradientProcessor. When provided:

            * ``unit_normalize=True`` — loads H^(-1/2) and applies to both
              query (here) and index (in :meth:`score`) for split
              (two-sided) preconditioning.
            * ``unit_normalize=False`` — loads H^(-1) and applies to query
              only for one-sided preconditioning.
        """
        self.device = device
        self.dtype = dtype
        self.modules = modules
        self.unit_normalize = unit_normalize
        self.score_mode = score_mode
        self.attribute_tokens = attribute_tokens
        self.writer = writer

        # Load preconditioner: H^(-1/2) for split, H^(-1) for one-sided
        self.preconditioners = get_trackstar_preconditioner(
            preconditioner_path,
            device=device,
            power=-0.5 if unit_normalize else -1,
        )
        if self.preconditioners:
            self.preconditioners = {
                k: v.to(dtype=dtype) for k, v in self.preconditioners.items()
            }

        # Concatenate query grads then apply preconditioning in-place on slices
        self.query_tensor = torch.cat(
            [query_grads[m].to(device=self.device, dtype=self.dtype) for m in modules],
            dim=1,
        )
        if self.preconditioners:
            offset = 0
            for m in modules:
                d = query_grads[m].shape[1]
                if m in self.preconditioners:
                    self.query_tensor[:, offset : offset + d] = (
                        self.query_tensor[:, offset : offset + d].float()
                        @ self.preconditioners[m].float()
                    ).to(self.dtype)
                offset += d

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        """Score a batch of training gradients against all queries."""
        # Convert the gradients to the scoring dtype
        if next(iter(mod_grads.values())).dtype != self.dtype:
            mod_grads = {name: grad.to(self.dtype) for name, grad in mod_grads.items()}

        scores = self.score(mod_grads)
        self.writer(indices, scores)

    @torch.inference_mode()
    def score(self, mod_grads: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute scores for a batch of gradients."""
        grads = torch.cat([mod_grads[m].to(self.device) for m in self.modules], dim=1)

        # Apply H^(-1/2) to index grads for split (two-sided) preconditioning.
        # One-sided mode (unit_normalize=False) only preconditions the query.
        if self.preconditioners and self.unit_normalize:
            offset = 0
            for m in self.modules:
                d = mod_grads[m].shape[1]
                if m in self.preconditioners:
                    grads[:, offset : offset + d] = (
                        grads[:, offset : offset + d] @ self.preconditioners[m]
                    )
                offset += d
        if self.unit_normalize:
            grads = grads / grads.norm(dim=1, keepdim=True)

        if self.score_mode == "nearest":
            all_scores = grads @ self.query_tensor.T
            return all_scores.max(dim=-1).values

        return grads @ self.query_tensor.T
