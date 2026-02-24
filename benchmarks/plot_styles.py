"""Shared color and style definitions for benchmark plots.

This module is intentionally dependency-free (no torch, datasets, etc.)
so that lightweight scripts like ``regenerate_plots.py`` can import it
without pulling in the full benchmark stack.
"""

from __future__ import annotations

# Explicit per-model colors so lines stay distinguishable even with 12+ series.
MODEL_COLORS: dict[str, str] = {
    "pythia-14m": "#1f77b4",  # blue
    "pythia-70m": "#ff7f0e",  # orange
    "pythia-160m": "#2ca02c",  # green
    "pythia-1b": "#d62728",  # red
    "pythia-6.9b": "#9467bd",  # purple
    "pythia-12b": "#8c564b",  # brown
}

# Fallback palette for models not in MODEL_COLORS.
_EXTRA_COLORS = [
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
]


def model_color(model_key: str) -> str:
    """Return a deterministic color for *model_key*."""
    if model_key in MODEL_COLORS:
        return MODEL_COLORS[model_key]
    # Stable fallback: hash to an index in _EXTRA_COLORS.
    return _EXTRA_COLORS[hash(model_key) % len(_EXTRA_COLORS)]
