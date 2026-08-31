"""One Plotly + Streamlit styling source: light and dark token sets.

Palettes are the data-viz reference categorical order, stepped per surface and
validated with ``scripts/validate_palette.js`` (all hard gates PASS in both
modes; light has a contrast WARN on aqua/yellow/magenta -> those series carry
relief: visible value labels + legend + table view, which every chart provides).

The app is light by default; a sidebar toggle switches ``dark`` and every chart
colour + a small CSS shell override follow it.
"""

from __future__ import annotations

from typing import Any

# validated categorical order — assign in this order, never cycle past slot 8
_PALETTE_LIGHT: list[str] = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
_PALETTE_DARK: list[str] = [
    "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767",
]

MAJOR_BRANDS: list[str] = [
    "Scania", "DAF", "Daimler Truck", "Volvo Trucks", "MAN", "Renault Trucks", "IVECO",
]
OTHER_LABEL = "Other"
BRAND_ORDER: list[str] = [*MAJOR_BRANDS, OTHER_LABEL]

_LIGHT: dict[str, Any] = {
    "dark": False,
    "palette": _PALETTE_LIGHT,
    "surface": "#fcfcfb",   # plot area
    "paper": "#ffffff",     # around the plot
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e6e5e2",
    "accent": _PALETTE_LIGHT[0],
    "good": "#1c5cab",      # CO2 fell — cool pole
    "bad": "#e34948",       # CO2 rose — warm pole
    "other": "#8a8a83",
    "app_bg": "#f5f4f1",
    "card_bg": "#ffffff",
    "corr_scale": [
        [0.0, "#6ba4e4"], [0.25, "#bcd6f4"], [0.5, "#f0efec"],
        [0.75, "#f4bdbc"], [1.0, "#e78b8a"],
    ],
    "corr_text": "#0b0b0b",
}

_DARK: dict[str, Any] = {
    "dark": True,
    "palette": _PALETTE_DARK,
    "surface": "#1a1a19",
    "paper": "#111110",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#3a3a37",
    "accent": _PALETTE_DARK[0],
    "good": "#5b9bd5",
    "bad": "#e66767",
    "other": "#8f8e84",
    "app_bg": "#111110",
    "card_bg": "#1a1a19",
    "corr_scale": [
        [0.0, "#5b9bd5"], [0.25, "#34506d"], [0.5, "#383835"],
        [0.75, "#7a4746"], [1.0, "#e07b76"],
    ],
    "corr_text": "#ffffff",
}


def tokens(dark: bool = False) -> dict[str, Any]:
    return _DARK if dark else _LIGHT


def fold_brand(name: str | None) -> str:
    """Map a raw brand to a major OEM or the 'Other' bucket."""
    return name if name in MAJOR_BRANDS else OTHER_LABEL


def brand_color_map(dark: bool = False) -> dict[str, str]:
    pal = tokens(dark)["palette"]
    m = {b: pal[i] for i, b in enumerate(MAJOR_BRANDS)}
    m[OTHER_LABEL] = tokens(dark)["other"]
    return m


def plotly_layout(dark: bool = False, **overrides: Any) -> dict[str, Any]:
    t = tokens(dark)
    layout: dict[str, Any] = {
        "paper_bgcolor": t["paper"],
        "plot_bgcolor": t["surface"],
        "font": {"color": t["text"], "family": "Inter, system-ui, sans-serif", "size": 14},
        "colorway": t["palette"],
        "title_font_size": 17,
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": t["grid"],
            "borderwidth": 1,
            "font": {"color": t["text"], "size": 12},
        },
        "margin": {"l": 64, "r": 32, "t": 56, "b": 48},
        "xaxis": {"gridcolor": t["grid"], "zerolinecolor": t["grid"], "linecolor": t["grid"],
                  "color": t["text"], "title_font_size": 13, "tickfont_size": 12},
        "yaxis": {"gridcolor": t["grid"], "zerolinecolor": t["grid"], "linecolor": t["grid"],
                  "color": t["text"], "title_font_size": 13, "tickfont_size": 12},
        "bargap": 0.28,
        "bargroupgap": 0.12,
    }
    layout.update(overrides)
    return layout


def corr_kw(dark: bool = False) -> dict[str, Any]:
    t = tokens(dark)
    return {"colorscale": t["corr_scale"], "zmid": 0, "zmin": -1, "zmax": 1}


def corr_textfont(dark: bool = False) -> dict[str, Any]:
    return {"color": tokens(dark)["corr_text"], "size": 12}


def app_css(dark: bool) -> str:
    """Minimal shell override so the Streamlit chrome matches the chart mode.

    Streamlit's own light/dark (Settings menu) still works; this just keeps the
    in-app toggle self-contained for the demo.
    """
    if not dark:
        return ""
    t = _DARK
    return f"""
<style>
  .stApp {{ background: {t['app_bg']}; color: {t['text']}; }}
  [data-testid="stSidebar"] {{ background: {t['card_bg']}; }}
  [data-testid="stHeader"] {{ background: {t['app_bg']}; }}
  .stApp, .stApp p, .stApp label, .stApp span, .stApp li,
  .stMarkdown, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {t['text']}; }}
  [data-testid="stDataFrame"], [data-testid="stJson"], [data-testid="stCodeBlock"] {{ background: {t['card_bg']}; }}
  pre, code, .stCodeBlock, [data-testid="stJson"] * {{ background: {t['card_bg']} !important; color: {t['text']} !important; }}
  [data-testid="stJson"] {{ border: 1px solid {t['grid']}; border-radius: 6px; }}
</style>
"""
