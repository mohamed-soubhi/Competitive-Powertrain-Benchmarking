"""Single source of truth for the dark "cyber matrix" Plotly styling.

Values are copied verbatim from the original per-script definitions in
``eda_analysis.py`` / ``ml_prediction.py`` / ``build_dashboard_v2.py`` so
regenerated charts are pixel-identical after the S12 refactor.
"""

from __future__ import annotations

from typing import Any

COLORS: dict[str, str] = {
    "bg": "#0a0e14",
    "paper": "#121a24",
    "text": "#f0fdf4",
    "muted": "#94a3b8",
    "grid": "#1e293b",
    "accent": "#10b981",
    "neon": "#00ff88",
    "blue": "#38bdf8",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "purple": "#a855f7",
    "soft": "rgba(16,185,129,0.12)",
}

PALETTE: list[str] = [
    "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7",
    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
]


def plotly_layout(**overrides: Any) -> dict[str, Any]:
    """Return the shared Plotly layout dict, with optional per-figure overrides."""
    layout: dict[str, Any] = {
        "paper_bgcolor": COLORS["paper"],
        "plot_bgcolor": COLORS["bg"],
        "font": {"color": COLORS["text"], "family": "Inter, sans-serif", "size": 13},
        "legend": {
            "bgcolor": COLORS["bg"],
            "bordercolor": COLORS["grid"],
            "font": {"size": 12},
        },
        "margin": {"l": 60, "r": 40, "t": 60, "b": 50},
    }
    layout.update(overrides)
    return layout
