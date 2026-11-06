"""Choropleth map of FraudBench scenario counts by U.S. state.

Counts use the top-level ``state`` field embedded in
``data/final/dataset_public.jsonl``. Federal appellate scenarios whose
``state`` is null are reported separately rather than assigned to a state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap, basemap_datadir

from ..scenario_courts import DEFAULT_SCENARIOS, load_scenario_courts
from ..style import COLORS, save_figure, setup_plot_style


DEFAULT_OUTPUT_DIR = Path("paper/figures")
STATE_CODES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]

# Label locations are adjusted from geographic centroids where necessary to
# keep abbreviations readable (notably in the Northeast).
STATE_LABEL_POSITIONS = {
    "AL": (-86.8, 32.8), "AK": (-152.5, 64.0), "AZ": (-111.9, 34.2),
    "AR": (-92.3, 34.9), "CA": (-119.7, 37.2), "CO": (-105.5, 39.0),
    "CT": (-72.7, 41.6), "DE": (-75.5, 39.0), "FL": (-82.2, 28.1),
    "GA": (-83.4, 32.7), "HI": (-157.7, 20.9), "ID": (-114.4, 44.2),
    "IL": (-89.2, 40.0), "IN": (-86.2, 40.0), "IA": (-93.5, 42.1),
    "KS": (-98.4, 38.5), "KY": (-85.3, 37.7), "LA": (-91.9, 31.0),
    "ME": (-69.0, 45.2), "MD": (-76.7, 39.1), "MA": (-71.8, 42.3),
    "MI": (-85.5, 44.4), "MN": (-94.3, 46.0), "MS": (-89.7, 32.7),
    "MO": (-92.5, 38.5), "MT": (-109.6, 47.0), "NE": (-99.8, 41.5),
    "NV": (-116.7, 39.3), "NH": (-71.6, 43.9), "NJ": (-74.6, 40.2),
    "NM": (-106.0, 34.5), "NY": (-75.5, 43.0), "NC": (-79.4, 35.5),
    "ND": (-100.5, 47.5), "OH": (-82.7, 40.3), "OK": (-97.5, 35.6),
    "OR": (-120.5, 44.0), "PA": (-77.7, 40.9), "RI": (-71.5, 41.7),
    "SC": (-80.9, 33.8), "SD": (-100.0, 44.5), "TN": (-86.0, 35.8),
    "TX": (-99.3, 31.3), "UT": (-111.7, 39.3), "VT": (-72.7, 44.2),
    "VA": (-78.7, 37.6), "WA": (-120.5, 47.4), "WV": (-80.6, 38.7),
    "WI": (-89.8, 44.6), "WY": (-107.6, 43.0), "DC": (-77.0, 38.9),
}
STATE_LABEL_OFFSETS = {
    "CT": (12, 7),
    "DC": (22, -22),
    "DE": (18, -10),
    "MA": (20, 13),
    "MD": (8, -18),
    "NH": (8, 9),
    "NJ": (15, 0),
    "RI": (27, 2),
    "VT": (-9, 9),
}


def compute_state_counts(scenarios: pd.DataFrame) -> pd.Series:
    """Count embedded state labels, including explicit zeros for 50 states + DC."""
    unknown_states = sorted(
        set(scenarios["state"].dropna().unique()) - set(STATE_CODES)
    )
    if unknown_states:
        raise ValueError(f"Unknown state labels: {', '.join(unknown_states)}")
    counts = scenarios["state"].value_counts().reindex(STATE_CODES, fill_value=0)
    counts.index.name = "state"
    return counts.astype(int)


def _add_county_choropleth(
    map_: Basemap,
    ax: plt.Axes,
    counts: pd.Series,
    cmap: colors.Colormap,
    norm: colors.Normalize,
    included_states: set[str],
) -> None:
    """Fill bundled county polygons using their parent state's count."""
    shapefile = str(Path(basemap_datadir) / "UScounties")
    map_.readshapefile(
        shapefile,
        "counties",
        drawbounds=False,
        default_encoding="latin-1",
    )
    polygons: list[list[tuple[float, float]]] = []
    facecolors: list[tuple[float, float, float, float]] = []
    for polygon, metadata in zip(map_.counties, map_.counties_info):
        state = metadata["STATE"].strip()
        if state not in included_states:
            continue
        polygons.append(polygon)
        facecolors.append(cmap(norm(int(counts.loc[state]))))

    ax.add_collection(
        PolyCollection(
            polygons,
            facecolors=facecolors,
            edgecolors="none",
            linewidths=0,
            zorder=1,
            # ~3000 county polygons make a vector PDF several MB; rasterize
            # just the fills (at savefig.dpi) and keep lines/text as vectors.
            rasterized=True,
        )
    )
    map_.drawcoastlines(color="#555555", linewidth=0.5, ax=ax, zorder=3)
    map_.drawstates(color="white", linewidth=0.8, ax=ax, zorder=3)
    map_.drawcountries(color="#555555", linewidth=0.5, ax=ax, zorder=3)
    map_.drawmapboundary(fill_color="white", color="none", ax=ax)


def _add_labels(
    map_: Basemap,
    ax: plt.Axes,
    counts: pd.Series,
    states: set[str],
    fontsize: float,
) -> None:
    """Annotate each state with its abbreviation and scenario count."""
    for state in STATE_CODES:
        if state not in states:
            continue
        lon, lat = STATE_LABEL_POSITIONS[state]
        x, y = map_(lon, lat)
        offset = STATE_LABEL_OFFSETS.get(state, (0, 0))
        arrowprops = None
        if offset != (0, 0):
            arrowprops = {
                "arrowstyle": "-",
                "color": "#555555",
                "linewidth": 0.45,
                "shrinkA": 1,
                "shrinkB": 1,
            }
        ax.annotate(
            f"{state}\n{int(counts.loc[state])}",
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#222222",
            linespacing=0.85,
            zorder=4,
            arrowprops=arrowprops,
        )


def plot_state_map(
    counts: pd.Series,
    output_path: Optional[Path] = None,
    *,
    missing_state_count: int = 0,
):
    """Render a state choropleth with Alaska and Hawaii insets."""
    setup_plot_style()
    cmap = colors.LinearSegmentedColormap.from_list(
        "fraudbench_state_counts", ["#FFFFFF", COLORS["blue"], COLORS["dark_teal"]]
    )
    norm = colors.Normalize(vmin=0, vmax=max(int(counts.max()), 1))

    fig = plt.figure(figsize=(12, 7))
    main_ax = fig.add_axes([0.04, 0.10, 0.80, 0.83])
    alaska_ax = fig.add_axes([0.04, 0.06, 0.20, 0.22])
    hawaii_ax = fig.add_axes([0.23, 0.07, 0.14, 0.16])

    lower_48 = set(STATE_CODES) - {"AK", "HI"}
    main_map = Basemap(
        projection="lcc",
        lat_1=33,
        lat_2=45,
        lon_0=-96,
        lat_0=38,
        width=5_500_000,
        height=3_500_000,
        resolution="l",
        ax=main_ax,
    )
    _add_county_choropleth(main_map, main_ax, counts, cmap, norm, lower_48)
    _add_labels(main_map, main_ax, counts, lower_48, fontsize=6.2)

    alaska_map = Basemap(
        projection="aea",
        lat_1=55,
        lat_2=65,
        lon_0=-150,
        lat_0=63,
        width=3_000_000,
        height=1_900_000,
        resolution="l",
        ax=alaska_ax,
    )
    _add_county_choropleth(alaska_map, alaska_ax, counts, cmap, norm, {"AK"})
    _add_labels(alaska_map, alaska_ax, counts, {"AK"}, fontsize=7)

    hawaii_map = Basemap(
        projection="merc",
        llcrnrlon=-161,
        llcrnrlat=18.5,
        urcrnrlon=-154.5,
        urcrnrlat=22.5,
        resolution="l",
        ax=hawaii_ax,
    )
    _add_county_choropleth(hawaii_map, hawaii_ax, counts, cmap, norm, {"HI"})
    _add_labels(hawaii_map, hawaii_ax, counts, {"HI"}, fontsize=7)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(np.asarray(counts))
    colorbar_ax = fig.add_axes([0.88, 0.24, 0.025, 0.52])
    colorbar = fig.colorbar(scalar_mappable, cax=colorbar_ax)
    colorbar.set_label("Number of Scenarios", fontweight="bold")
    colorbar.set_ticks(range(0, int(counts.max()) + 1, 4))

    mapped_total = int(counts.sum())
    fig.text(
        0.46,
        0.045,
        f"Scenarios with a state assignment: n={mapped_total}",
        ha="center",
        fontsize=9,
    )
    if missing_state_count:
        fig.text(
            0.46,
            0.018,
            f"{missing_state_count} federal appellate scenarios span multiple states "
            "and are not mapped.",
            ha="center",
            fontsize=8,
            color="#555555",
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    return fig, (main_ax, alaska_ax, hawaii_ax)


def run(
    input_path: str = DEFAULT_SCENARIOS,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> pd.Series:
    """Generate the choropleth from court fields embedded in scenario data."""
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    scenarios = load_scenario_courts(input_path)
    counts = compute_state_counts(scenarios)
    missing_state_count = int(scenarios["state"].isna().sum())

    print("\n" + "=" * 60)
    print("SCENARIO COUNTS BY STATE")
    print("=" * 60)
    print(counts[counts > 0].to_string())
    print(f"\nMapped scenarios: {int(counts.sum())}")
    print(f"Scenarios without a state assignment: {missing_state_count}")

    plot_state_map(
        counts,
        output_dir / "scenario_counts_by_state.pdf",
        missing_state_count=missing_state_count,
    )
    if show:
        plt.show()
    return counts
