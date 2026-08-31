"""One Plotly styling source: palette, OEM colour map, layout defaults.

Palette = the validated data-viz reference instance (categorical, light mode).
Run of ``scripts/validate_palette.js``: all hard gates PASS; contrast WARN on
aqua / yellow / magenta means those series must carry relief (visible value
labels or a table view) — every chart in the app does.

The app is light-mode only (Streamlit default), so only the light column is
defined here.
"""

from __future__ import annotations

from typing import Any

# validated categorical order — assign in this order, never cycle past slot 8
PALETTE: list[str] = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

COLORS: dict[str, str] = {
    "bg": "#fcfcfb",        # chart surface (validator surface-1, light)
    "paper": "#fcfcfb",
    "text": "#0b0b0b",      # text-primary
    "muted": "#52514e",     # text-secondary
    "grid": "#e6e5e2",
    "accent": PALETTE[0],
    "good": "#1c5cab",      # CO2 fell — cool pole of the diverging pair
    "bad": "#e34948",       # CO2 rose — warm pole
    "other": "#8a8a83",     # the "Other" OEM bucket — neutral, not a hue
}

# OEMs with enough volume to be their own series (by row count, 2019–2020).
# Everything else folds into "Other" so we never cycle the palette.
MAJOR_BRANDS: list[str] = [
    "Scania", "DAF", "Daimler Truck", "Volvo Trucks", "MAN", "Renault Trucks", "IVECO",
]
OTHER_LABEL = "Other"
BRAND_ORDER: list[str] = [*MAJOR_BRANDS, OTHER_LABEL]


def fold_brand(name: str | None) -> str:
    """Map a raw brand to a major OEM or the 'Other' bucket."""
    return name if name in MAJOR_BRANDS else OTHER_LABEL


def brand_color_map() -> dict[str, str]:
    m = {b: PALETTE[i] for i, b in enumerate(MAJOR_BRANDS)}
    m[OTHER_LABEL] = COLORS["other"]
    return m


def plotly_layout(**overrides: Any) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "paper_bgcolor": COLORS["paper"],
        "plot_bgcolor": COLORS["bg"],
        "font": {"color": COLORS["text"], "family": "Inter, system-ui, sans-serif", "size": 13},
        "colorway": PALETTE,
        "legend": {"bgcolor": "rgba(252,252,251,0.7)", "bordercolor": COLORS["grid"], "borderwidth": 1},
        "margin": {"l": 64, "r": 32, "t": 56, "b": 48},
        "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"], "linecolor": COLORS["grid"]},
        "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"], "linecolor": COLORS["grid"]},
    }
    layout.update(overrides)
    return layout


# Diverging scale for correlation heatmaps: blue (-1) -> neutral gray (0) -> red (+1).
# Two hues + a gray midpoint that reads as "nothing" — never a rainbow.
# Kept deliberately PALE so overlaid r-values stay legible in near-black at every
# cell; polarity comes from hue, magnitude is also printed as text.
CORR_COLORSCALE = [
    [0.0, "#6ba4e4"],
    [0.25, "#bcd6f4"],
    [0.5, "#f0efec"],
    [0.75, "#f4bdbc"],
    [1.0, "#e78b8a"],
]
CORR_KW: dict[str, Any] = {"colorscale": CORR_COLORSCALE, "zmid": 0, "zmin": -1, "zmax": 1}
CORR_TEXTFONT: dict[str, Any] = {"color": COLORS["text"], "size": 12}

# Single-hue sequential ramp (magnitude), light -> dark.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]
