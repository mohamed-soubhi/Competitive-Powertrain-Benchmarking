"""Light/dark token sets and brand folding."""

import pytest

from powerbench.theme import (
    MAJOR_BRANDS,
    OTHER_LABEL,
    app_css,
    brand_color_map,
    corr_kw,
    corr_textfont,
    fold_brand,
    plotly_layout,
    tokens,
)


@pytest.mark.parametrize("dark", [False, True])
def test_tokens_have_all_roles(dark):
    t = tokens(dark)
    for key in ("palette", "surface", "paper", "text", "muted", "grid", "accent",
                "good", "bad", "other", "corr_scale", "corr_text"):
        assert key in t
    assert len(t["palette"]) == 8
    assert t["dark"] is dark


@pytest.mark.parametrize("dark", [False, True])
def test_plotly_layout_uses_mode_surface(dark):
    lay = plotly_layout(dark)
    assert lay["plot_bgcolor"] == tokens(dark)["surface"]
    assert lay["font"]["color"] == tokens(dark)["text"]


@pytest.mark.parametrize("dark", [False, True])
def test_brand_map_covers_all_and_unique(dark):
    m = brand_color_map(dark)
    assert set(m) == {*MAJOR_BRANDS, OTHER_LABEL}
    assert len(set(m.values())) == len(m)


def test_fold_brand():
    assert fold_brand("Scania") == "Scania"
    assert fold_brand("Dongfeng") == OTHER_LABEL
    assert fold_brand(None) == OTHER_LABEL


def test_corr_helpers_mode_specific():
    assert corr_kw(True)["colorscale"] != corr_kw(False)["colorscale"]
    assert corr_textfont(True)["color"] != corr_textfont(False)["color"]


def test_app_css_empty_for_light_nonempty_for_dark():
    assert app_css(False) == ""
    assert "<style>" in app_css(True)
