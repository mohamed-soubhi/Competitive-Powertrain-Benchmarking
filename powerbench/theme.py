"""One Plotly styling source: palette, OEM colour map, layout defaults.

Keeps every chart in the app visually consistent. Import ``plotly_layout`` for
the shared layout dict and ``brand_color_map`` so an OEM keeps the same colour
across the trend line, the ranking bar and the mix chart.
"""

from __future__ import annotations

from typing import Any

# neutral, print-safe qualitative palette (not tied to any real brand livery)
PALETTE: list[str] = [
    "#2563eb", "#e11d48", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5", "#94a3b8",
]

COLORS: dict[str, str] = {
    "bg": "#ffffff",
    "paper": "#ffffff",
    "text": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "accent": "#2563eb",
    "good": "#059669",   # lower CO2 = better
    "bad": "#e11d48",
}

# stable OEM order -> colour (majors first, long tail last)
BRAND_ORDER: list[str] = [
    "Scania", "MAN", "DAF", "Volvo Trucks", "Renault Trucks", "Daimler Truck",
    "IVECO", "Ford Trucks", "FUSO", "Isuzu", "Dongfeng", "Unknown",
]


def brand_color_map() -> dict[str, str]:
    return {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(BRAND_ORDER)}


def plotly_layout(**overrides: Any) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "paper_bgcolor": COLORS["paper"],
        "plot_bgcolor": COLORS["bg"],
        "font": {"color": COLORS["text"], "family": "Inter, system-ui, sans-serif", "size": 13},
        "colorway": PALETTE,
        "legend": {"bgcolor": "rgba(255,255,255,0.6)", "bordercolor": COLORS["grid"], "borderwidth": 1},
        "margin": {"l": 64, "r": 32, "t": 56, "b": 48},
        "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
    }
    layout.update(overrides)
    return layout


# diverging scale for correlation heatmaps: blue -1, white 0, red +1
CORR_SCALE = "RdBu_r"
CORR_KW = {"colorscale": CORR_SCALE, "zmid": 0, "zmin": -1, "zmax": 1}
