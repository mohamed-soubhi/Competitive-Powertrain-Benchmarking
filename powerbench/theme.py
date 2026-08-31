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
    "card_bg2": "#f0efec",   # secondary surface (inputs, hover targets)
    "hover": "#eae8e2",
    "border": "#e6e5e2",
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
    "card_bg2": "#222220",   # secondary surface — matches the docs site
    "hover": "#252524",
    "border": "#3a3a37",
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
    """Full dark-mode shell override for the Streamlit chrome.

    The palette mirrors the project's HTML pages (``docs/documentation.html`` /
    the GitHub Pages build): app ``#111110``, card ``#1a1a19``, secondary
    ``#222220``, border ``#3a3a37``, text ``#ffffff`` / muted ``#c3c2b7``,
    accent ``#3987e5``. Every interactive widget (tabs, inputs, dropdown menus,
    sliders, expanders, alerts, buttons, tables, code, JSON) is covered so
    nothing renders in Streamlit's default light styling on a dark page.
    Streamlit's own Settings → Theme still works independently.
    """
    if not dark:
        return ""
    t = _DARK
    bg, card, card2, hover = t["app_bg"], t["card_bg"], t["card_bg2"], t["hover"]
    line, text, muted, accent = t["border"], t["text"], t["muted"], t["accent"]
    return f"""
<style>
  /* ---- roots & scrollbars: native controls follow the mode ---- */
  .stApp {{ background: {bg}; color: {text}; color-scheme: dark; }}
  .stApp, .stApp [class*="st-"] {{
    --background-color: {bg};
    --secondary-background-color: {card};
    --text-color: {text};
    --primary-color: {accent};
  }}
  [data-testid="stHeader"] {{ background: {bg}; }}
  [data-testid="stSidebar"] {{ background: {card}; border-right: 1px solid {line}; }}
  [data-testid="stSidebar"] * {{ color: {text}; }}

  /* ---- typography ---- */
  .stApp p, .stApp li, .stApp label, .stMarkdown,
  [data-testid="stMarkdownContainer"], h1, h2, h3, h4, h5, h6,
  [data-testid="stHeadingWithActionElements"] {{ color: {text}; }}
  [data-testid="stCaptionContainer"], .stApp small, [data-testid="stWidgetLabel"] p {{ color: {muted}; }}
  [data-testid="stMetricValue"] {{ color: {text}; }}
  [data-testid="stMetricLabel"] p {{ color: {muted}; }}
  hr, [data-testid="stDivider"] hr {{ border-color: {line}; }}
  a {{ color: {accent}; }}

  /* ---- tabs ---- */
  [data-baseweb="tab-list"] {{ background: transparent; border-bottom: 1px solid {line}; }}
  [data-baseweb="tab"] {{ color: {muted}; }}
  [data-baseweb="tab"][aria-selected="true"] {{ color: {text}; }}
  [data-baseweb="tab-highlight"] {{ background: {accent}; }}
  [data-baseweb="tab-border"] {{ background: {line}; }}

  /* ---- inputs, selects, dropdown menus ---- */
  [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
  [data-baseweb="textarea"], .stTextInput input, .stNumberInput input,
  [data-testid="stNumberInputContainer"] {{
    background: {card2} !important; color: {text} !important; border-color: {line} !important;
  }}
  [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], [data-baseweb="popover"] ul {{
    background: {card} !important; border: 1px solid {line} !important;
  }}
  [data-baseweb="menu"] li, [role="option"] {{ color: {text} !important; }}
  [role="option"]:hover, [data-baseweb="menu"] li:hover {{ background: {hover} !important; }}
  [data-baseweb="tag"] {{ background: {accent} !important; color: #fff !important; }}

  /* ---- sliders ---- */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background: {accent}; }}
  [data-testid="stSliderTickBar"] {{ background: {line}; }}
  [data-testid="stSlider"] [data-testid="stThumbValue"] {{ color: {text}; }}

  /* ---- checkbox / radio / toggle: tint from color-scheme + accent ---- */
  [data-testid="stCheckbox"], [data-testid="stRadio"], [data-testid="stToggle"] {{ accent-color: {accent}; }}
  [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {{ color: {text}; }}

  /* ---- buttons ---- */
  [data-testid="stBaseButton-secondary"], button[kind="secondary"], [data-testid="baseButton-secondary"] {{
    background: {card2}; color: {text}; border: 1px solid {line};
  }}
  [data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {{ background: {hover}; border-color: {accent}; }}
  [data-testid="stBaseButton-primary"], button[kind="primary"] {{ background: {accent}; color: #fff; border: 0; }}
  [data-testid="stDownloadButton"] button {{ background: {card2}; color: {text}; border: 1px solid {line}; }}

  /* ---- expander, status, alerts ---- */
  [data-testid="stExpander"] {{ background: {card}; border: 1px solid {line}; border-radius: 8px; }}
  [data-testid="stExpander"] summary, [data-testid="stExpander"] p {{ color: {text}; }}
  [data-testid="stStatusWidget"], [data-testid="stStatus"] {{ background: {card}; border: 1px solid {line}; color: {text}; }}
  [data-testid="stAlert"], [data-testid="stNotification"] {{
    background: {card2}; color: {text}; border: 1px solid {line}; border-left: 4px solid {accent};
  }}
  [data-testid="stAlert"] * {{ color: {text}; }}

  /* ---- data display: dataframe, table, code, json ---- */
  [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stJson"],
  [data-testid="stCodeBlock"], [data-testid="stExpanderDetails"] {{ background: {card}; }}
  [data-testid="stTable"] th {{ background: {card2}; color: {text}; }}
  [data-testid="stTable"] td {{ color: {text}; border-color: {line}; }}
  pre, [data-testid="stCodeBlock"] pre {{ background: #0f1720 !important; color: #e7edf3 !important; border: 1px solid {line}; }}
  code:not(pre code) {{ background: {card2}; color: {text}; padding: 0.1em 0.35em; border-radius: 4px; }}
  [data-testid="stJson"] {{ border: 1px solid {line}; border-radius: 6px; }}
  [data-testid="stJson"] * {{ background: {card} !important; color: {text} !important; }}
</style>
"""
