"""Consolidated figure styling for fraudbench paper.

This module is the single source of truth for matplotlib styling and the
paper color palette. All plotting code in :mod:`fraudbench.analysis.plot`
should call :func:`setup_plot_style` once before drawing.

Tables that report empirical percentages share a single heatmap-shading
convention through :func:`latex_cell_color`: cells linearly interpolate
from white at 0\\% to the named palette color at 100\\%, capped at
:data:`HEATMAP_MAX_INTENSITY` of the ``xcolor`` ``color!N`` mix scale so
text remains legible at the high end.

The matplotlib rcParams themselves live in ``fraudbench.mplstyle`` next
to this file (matplotlib stylesheet syntax). The file ships with the
package so the style is available from any checkout and survives
installation. This module simply locates that file and applies it via
:func:`matplotlib.pyplot.style.use`.

Color palette design
---------------------
The figure programme distinguishes several orthogonal axes of comparison
(metric, AI/human, contextual condition, evaluation setting, model
family). Reusing the same color for two different axes would let a
reader confuse, e.g., a "Human" bar with a "Fraud" bar. Each axis
therefore gets a dedicated palette:

* :data:`METRIC_COLORS` -- five-metric primary palette (Fraud / Refusal /
  Options / Legal Awareness / Fraud Awareness). The rest of the figure
  programme treats these hexes as "owned" by the metric axis.
* :data:`AI_HUMAN_COLORS` -- two-class palette (Human, AI) used by the
  human-baseline figures. Picked from Okabe-Ito to be perceptually
  distinct from every metric color.
* :data:`CONTEXTUAL_COLORS` -- five-class palette for the contextual
  conditions ("none" + low/high peer-pressure / negative-incentive).
  Drawn from a ColorBrewer PuOr-style ramp so peer-pressure (orange)
  and negative-incentive (purple) split cleanly along hue, with
  low/high reading as a brightness ramp.
* :data:`SETTING_COLORS` -- three-class palette for the AI evaluation
  settings (Binary/Open/Agentic). Hatching remains the primary
  distinguisher in compare-settings plots; the colors here only feed
  legend swatches and any future setting-colored bars.
* :data:`FAMILY_COLORS` -- nine-family palette for LLM families. We
  deliberately re-use metric hexes here because per-family figures
  never share a legend with metrics.

All palettes are color-blind safe under deuteranopia / protanopia /
tritanopia (verified by spreading luminance and avoiding the red-green
diagonal). For grayscale/print fall-back, the figure programme also
relies on hatch patterns (compare-settings plots) and on the inherent
luminance ladder of each palette.

The :data:`COLORS` dict is the underlying name-to-hex registry; the
semantic palettes above index into it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator


STYLE_PATH = Path(__file__).resolve().parent / "fraudbench.mplstyle"


# ---------------------------------------------------------------------------
# Underlying name-to-hex registry. Kept flat so LaTeX color helpers and
# legacy callers can still reach individual hexes by name. New code should
# prefer the semantic palettes below.
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    # Brand palette (also used by ``paper/icons/colors.txt``).
    "blue":              "#42C0D2",
    "yellow":            "#FCE108",
    "bright_red":        "#FC087D",
    "dark_purple":       "#A73D70",
    "dark_teal":         "#42757D",
    "dark_green_brown": "#464123",
    "alt_light_blue":    "#73C1CF",

    # Color-blind-safe accent colors used outside the metric palette.
    # Picked from the Okabe-Ito qualitative palette and ColorBrewer PuOr
    # so they are perceptually distinct from the brand palette and from
    # each other under all common color-vision deficiencies.
    "vermillion":   "#D55E00",  # Okabe-Ito; AI-vs-Human (Human bar)
    "deep_blue":    "#0072B2",  # Okabe-Ito; AI-vs-Human (AI bar)
    "orange_light": "#FDB863",  # Brewer PuOr-7 [4]; peer-pressure-low
    "orange_dark":  "#E66101",  # Brewer PuOr-7 [5]; peer-pressure-high
    "purple_light": "#B2ABD2",  # Brewer PuOr-7 [3]; negative-incentive-low
    "purple_dark":  "#5E3C99",  # Brewer PuOr-7 [2]; negative-incentive-high
    "neutral_gray": "#7F7F7F",  # universal "baseline / no condition" grey
}


# ---------------------------------------------------------------------------
# Semantic palettes
#
# The keys are the canonical strings used elsewhere in
# :mod:`fraudbench.analysis` so callers can index directly without an
# extra translation step:
#
#   color = METRIC_COLORS["fraud"]
#   color = CONTEXTUAL_COLORS[row["contextual_condition"]]
#   color = AI_HUMAN_COLORS["human" if is_human else "ai"]
# ---------------------------------------------------------------------------


METRIC_COLORS: dict[str, str] = {
    "fraud":            COLORS["bright_red"],
    "refusal":          COLORS["blue"],
    "options":          COLORS["yellow"],
    "legal_awareness":  COLORS["dark_purple"],
    "fraud_awareness": COLORS["dark_teal"],
}
"""Canonical color per outcome metric (used by every metric-colored bar)."""


AI_HUMAN_COLORS: dict[str, str] = {
    "human": COLORS["vermillion"],
    "ai":    COLORS["deep_blue"],
}
"""Two-class palette for human-baseline vs AI comparisons.

These hexes are chosen so a "Human" bar can never be mistaken for a
"Fraud" bar (which is :data:`COLORS["bright_red"]`) and an "AI" bar
can never be mistaken for a "Refusal" bar (which is
:data:`COLORS["blue"]`).
"""


CONTEXTUAL_COLORS: dict[str, str] = {
    "none":                     COLORS["neutral_gray"],
    "peer_pressure_low":        COLORS["orange_light"],
    "peer_pressure_high":       COLORS["orange_dark"],
    "negative_incentive_low":   COLORS["purple_light"],
    "negative_incentive_high":  COLORS["purple_dark"],
}
"""Five-class palette for contextual conditions.

Disjoint from :data:`METRIC_COLORS` so a contextual-condition bar can
never be confused with a metric bar in adjacent figures. The diverging
PuOr-style structure encodes peer-pressure-vs-incentive as hue and
low-vs-high as luminance, so the palette is meaningful even when only
hue or luminance is preserved (greyscale, partial color vision).
"""

# Aliases for the two human-baseline-only labels: the human plots use
# ``"control"`` instead of ``"none"`` because the human-baseline data
# never sees a contextual "Control" / "no contextual" column. Map both
# spellings to the same neutral-grey swatch so the same physical bar
# gets the same color across the two figure families.
HUMAN_BASELINE_CONDITION_COLORS: dict[str, str] = {
    "control":                  CONTEXTUAL_COLORS["none"],
    "peer_pressure_low":        CONTEXTUAL_COLORS["peer_pressure_low"],
    "peer_pressure_high":       CONTEXTUAL_COLORS["peer_pressure_high"],
    "negative_incentive_low":   CONTEXTUAL_COLORS["negative_incentive_low"],
    "negative_incentive_high":  CONTEXTUAL_COLORS["negative_incentive_high"],
}
"""``CONTEXTUAL_COLORS`` keyed by the labels used in the human-baseline data."""


SETTING_COLORS: dict[str, str] = {
    "closed":  COLORS["neutral_gray"],
    "open":    COLORS["dark_green_brown"],
    "agentic": COLORS["dark_teal"],
}
"""Three-class palette for AI evaluation settings (Binary/Open/Agentic).

The compare-settings plots overlay hatching on top of the metric color,
so this palette is a fall-back used only when a setting needs its own
color swatch (e.g. an outside-the-grid legend).
"""


FAMILY_COLORS: dict[str, str] = {
    "Claude":   COLORS["bright_red"],
    "GPT":      COLORS["blue"],
    "Gemini":   COLORS["yellow"],
    "Grok":     COLORS["dark_purple"],
    "DeepSeek": COLORS["dark_teal"],
    "Qwen":     COLORS["alt_light_blue"],
    "GLM":      COLORS["dark_green_brown"],
    "Kimi":     COLORS["vermillion"],
    "GPT-OSS":  COLORS["purple_dark"],
}
"""Nine-family palette for LLM families.

Reuses :data:`METRIC_COLORS` hexes because per-family figures (the
``average_by_model_*`` plots) never display a metric legend alongside
the family-colored bars. The two families that do not have a metric
analogue (``Kimi`` and ``GPT-OSS``) take accent colors from
:data:`COLORS["vermillion"]` / :data:`COLORS["purple_dark"]` so all
nine families remain pairwise distinguishable.
"""


def setup_plot_style() -> None:
    """Apply the shared paper figure style to matplotlib.

    Loads ``paper/styles/fraudbench.mplstyle`` via :func:`plt.style.use`.
    Idempotent and cheap to call repeatedly.
    """
    if not STYLE_PATH.exists():
        raise FileNotFoundError(
            f"fraudbench style file not found at {STYLE_PATH}. "
            "It should ship next to fraudbench.analysis.style; reinstall the "
            "package or check that fraudbench.mplstyle is present alongside "
            "style.py."
        )
    plt.style.use(str(STYLE_PATH))


_VECTOR_SUFFIXES = {".pdf", ".svg", ".svgz", ".eps", ".ps"}


def _freeze_auto_ticks(fig: plt.Figure, dpi: float) -> None:
    """Pin every automatic major tick locator in ``fig`` to the ticks it
    chooses when the figure is rendered at ``dpi``."""
    original_dpi = fig.dpi
    fig.set_dpi(dpi)
    try:
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                locator = axis.get_major_locator()
                if isinstance(locator, MaxNLocator) and getattr(locator, "_nbins", None) == "auto":
                    axis.set_major_locator(FixedLocator(list(axis.get_majorticklocs())))
    finally:
        fig.set_dpi(original_dpi)


def save_figure(fig: plt.Figure, path: str | Path, **kwargs: Any) -> None:
    """Save ``fig`` to ``path`` so vector output has the same ticks as raster.

    Use this instead of ``fig.savefig`` / ``plt.savefig`` for paper figures.

    matplotlib's ``AutoLocator`` picks the number of major ticks from the
    axis length measured in tick-label heights, floored to an integer, and
    evaluates that ratio at the *renderer's* dpi. Vector backends (PDF, SVG,
    EPS) always render at 72 dpi, so when the ratio sits on an integer
    boundary -- which it does for several paper figures, e.g. a 3.5 in axis
    with 18 pt tick labels gives exactly 7.0 -- floating-point noise makes a
    PDF pick a different tick step (0.25) than the ``savefig.dpi`` PNG the
    figure was designed against (0.2). For vector output this first pins
    every automatic locator to the ticks it chooses at ``savefig.dpi`` (or
    the ``dpi`` keyword), then saves; raster output is saved unchanged.
    """
    path = Path(path)
    if path.suffix.lower() in _VECTOR_SUFFIXES:
        dpi = kwargs.get("dpi", plt.rcParams["savefig.dpi"])
        if dpi == "figure":
            dpi = fig.dpi
        _freeze_auto_ticks(fig, float(dpi))
    fig.savefig(path, **kwargs)


def latex_color_definitions(names: Iterable[tuple[str, str]]) -> str:
    """Return ``\\definecolor`` lines for the given (latex_name, palette_key) pairs.

    Hex values are taken from :data:`COLORS` so LaTeX colors stay in sync
    with matplotlib colors. Example:

    >>> latex_color_definitions([("fraudcolor", "bright_red")])
    '\\\\definecolor{fraudcolor}{HTML}{FC087D}'
    """
    lines: list[str] = []
    for latex_name, palette_key in names:
        hex_value = COLORS[palette_key].lstrip("#").upper()
        lines.append(f"\\definecolor{{{latex_name}}}{{HTML}}{{{hex_value}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared heatmap-shading scale for LaTeX percentage tables
# ---------------------------------------------------------------------------
#
# ``xcolor``'s ``color!N`` syntax mixes ``color`` (N) with white (100 - N),
# so ``!0`` is pure white and ``!100`` is the pure palette color. The
# rendered tables fix the empirical-percentage scale to a single linear
# ramp so a "darker = higher" reading is consistent across every table.

HEATMAP_MAX_INTENSITY: int = 80
"""``xcolor`` mix percentage applied at 100\\% (i.e. ``\\cellcolor{color!80}``).

Caps the high end of the ramp below pure color so dark base hues stay
readable at the maximum (a fully saturated ``dark_teal`` cell at ``!100``
would clash with normal black body text).
"""


def latex_cell_color(value: float, color: str) -> str:
    """Return a ``\\cellcolor{color!N}`` string for *value* in ``[0, 1]``.

    Maps ``0`` to pure white (``!0``) and ``1`` to ``!HEATMAP_MAX_INTENSITY``
    (a deep tint of *color*) using a linear interpolation, so the same
    fraction always produces the same shade across every table that
    consumes :func:`latex_cell_color`. Values outside ``[0, 1]`` are
    clipped; ``None`` and ``NaN`` collapse to an empty string so the
    caller can leave the cell unshaded.

    Args:
        value: Empirical proportion in ``[0, 1]`` (e.g. ``0.42`` for 42\\%).
        color: ``xcolor`` color name already registered via
            :func:`latex_color_definitions` (e.g. ``"fraudcolor"``).
    """
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(v):
        return ""
    clipped = max(0.0, min(1.0, v))
    intensity = int(round(clipped * HEATMAP_MAX_INTENSITY))
    return f"\\cellcolor{{{color}!{intensity}}}"
