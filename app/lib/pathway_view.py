"""Render a KEGG pathway by overlaying expression data on KEGG's native PNG."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import OUT

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_pathway_index() -> pd.DataFrame:
    return pd.read_parquet(OUT / "pathway_index.parquet")


@st.cache_data(show_spinner=False)
def load_pathway_nodes() -> pd.DataFrame:
    return pd.read_parquet(OUT / "pathway_nodes.parquet")


@st.cache_data(show_spinner=False)
def load_pathway_edges() -> pd.DataFrame:
    return pd.read_parquet(OUT / "pathway_edges.parquet")


@st.cache_data(show_spinner=False)
def png_data_uri(png_path_rel: str) -> str:
    """Encode a pathway PNG as a base64 data URI so it embeds into the figure JSON."""
    b = (ROOT / png_path_rel).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def render_pathway(
    pathway_id: str,
    gene_value: pd.Series,
    *,
    value_label: str,
    value_range: tuple[float, float] | None = None,
    diverging: bool = True,
    fill_opacity: float = 1.0,
) -> go.Figure:
    """Overlay expression on KEGG's native pathway PNG (pathview-style).

    Gene nodes with expression data get a saturated coloured rectangle that
    replaces KEGG's default green fill, with the symbol redrawn on top in
    auto-contrast text. Nodes without data keep a dashed outline so the
    KEGG label shows through. Arrows and compounds come from the PNG.
    """
    info = load_pathway_index().set_index("pathway_id").loc[pathway_id]
    pn = load_pathway_nodes()
    pn = pn[pn["pathway_id"] == pathway_id].copy()
    png_w = int(info["png_w"])
    png_h = int(info["png_h"])

    pn["value"] = pn["ensembl_id"].map(gene_value).astype(float)

    # Colour scale domain
    if value_range is None:
        vals = pn["value"].dropna()
        if diverging:
            v = max(1.0, float(np.nanmax(np.abs(vals))) if len(vals) else 1.0)
            vmin, vmax = -v, v
        else:
            vmin = float(vals.min()) if len(vals) else 0.0
            vmax = float(vals.max()) if len(vals) else 1.0
    else:
        vmin, vmax = value_range
    # Saturated diverging palette — small log2FC values land in clearly blue/red.
    # Midpoint is light gray (#ececec) rather than pure white so near-neutral
    # boxes remain visually distinguishable from blank background.
    SATURATED_DIVERGING = [
        [0.00, "#08306b"],  # deep blue
        [0.25, "#4292c6"],
        [0.45, "#c6dbef"],
        [0.50, "#ececec"],  # light gray neutral
        [0.55, "#fcbba1"],
        [0.75, "#ef3b2c"],
        [1.00, "#67000d"],  # deep red
    ]
    colorscale = SATURATED_DIVERGING if diverging else "Viridis"

    fig = go.Figure()

    # --- base layer: KEGG PNG (always at the bottom) ---
    fig.add_layout_image(dict(
        source=png_data_uri(info["png_path"]),
        xref="x", yref="y",
        x=0, y=0, sizex=png_w, sizey=png_h,
        sizing="stretch",
        opacity=1.0,
        layer="below",
    ))

    # --- overlay: saturated rectangles + auto-contrast labels via annotations ---
    # Use layout.annotations (not a scatter trace) for labels so they always
    # render on top of shapes regardless of layer ordering.
    for _, n in pn.iterrows():
        if n["kind"] != "gene":
            continue  # compounds and pathway-pathway links: untouched
        x0, x1 = n["x"] - n["width"] / 2, n["x"] + n["width"] / 2
        y0, y1 = n["y"] - n["height"] / 2, n["y"] + n["height"] / 2
        v = n["value"]
        if pd.isna(v):
            # Unmapped or filtered: dashed outline only; KEGG's own label shows through
            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                fillcolor="rgba(0,0,0,0)",
                line={"color": "#888", "width": 0.6, "dash": "dot"},
                layer="above",
            )
            continue
        t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        t = float(np.clip(t, 0, 1))
        rgb = _colorscale_sample_rgb(colorscale, t)
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{fill_opacity:.2f})",
            line={"color": "#1a1a1a", "width": 1.0},
            layer="above",
        )
        # Redraw gene symbol as an annotation (always on top of shapes).
        sym = n["primary_symbol"] or ""
        if sym:
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = "#ffffff" if lum < 150 else "#111111"
            fig.add_annotation(
                x=n["x"], y=n["y"],
                xref="x", yref="y",
                text=sym,
                showarrow=False,
                font={"size": 9, "color": text_color,
                      "family": "Helvetica, Arial, sans-serif"},
                align="center",
                xanchor="center", yanchor="middle",
            )

    # --- hover layer: invisible markers at gene-node centres ---
    gene_rows = pn[pn["kind"] == "gene"].copy()
    gene_rows["hover"] = gene_rows.apply(
        lambda r: (
            f"<b>{r['primary_symbol'] or '(unnamed)'}</b><br>"
            f"KEGG entry {r['entry_id']}<br>"
            f"All symbols: {r['symbols'] or '—'}<br>"
            + (f"{value_label}: {r['value']:.2f}" if pd.notna(r['value'])
               else "no expression data")
        ),
        axis=1,
    )
    fig.add_trace(go.Scatter(
        x=gene_rows["x"], y=gene_rows["y"],
        mode="markers",
        marker={"size": 10, "color": "rgba(0,0,0,0)", "line": {"width": 0}},
        hoverinfo="text",
        hovertext=gene_rows["hover"],
        showlegend=False,
    ))

    # --- colourbar: phantom scatter so we get a legend ---
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker={
            "colorscale": colorscale,
            "cmin": vmin, "cmax": vmax,
            "color": [vmin],
            "colorbar": {"title": value_label, "thickness": 12, "len": 0.8},
            "size": 0.1,
        },
        showlegend=False, hoverinfo="skip",
    ))

    # Layout at native PNG dimensions (1:1 pixel mapping preserves KEGG's layout).
    # Cap display height at 900px but allow wider aspect ratios.
    display_h = min(900, png_h)
    display_w = int(png_w * display_h / png_h)
    fig.update_layout(
        width=display_w + 60,       # leave room for colorbar
        height=display_h + 80,      # leave room for title
        margin={"l": 10, "r": 60, "t": 50, "b": 10},
        title=f"{pathway_id} — {info['title']}",
        xaxis={
            "visible": False, "range": [0, png_w],
            "constrain": "domain",
        },
        yaxis={
            "visible": False, "range": [png_h, 0],
            "scaleanchor": "x", "scaleratio": 1,
        },
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


# --- colorscale sampling ---
_SCALES: dict[str, list[tuple[float, tuple[int, int, int]]]] = {}


def _parse_rgb(rgb: str) -> tuple[int, int, int]:
    if rgb.startswith("#"):
        return tuple(int(rgb[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    nums = rgb[rgb.find("(") + 1:rgb.find(")")].split(",")
    return (int(nums[0]), int(nums[1]), int(nums[2]))


def _colorscale_sample_rgb(scale, t: float) -> tuple[int, int, int]:
    """Sample an RGB tuple at position t in [0,1] from either a named Plotly
    colorscale (str) or an explicit list of [position, "#hex"] pairs."""
    if isinstance(scale, str):
        if scale not in _SCALES:
            import plotly.colors as pc
            raw = pc.get_colorscale(scale)
            _SCALES[scale] = [(p, _parse_rgb(c)) for p, c in raw]
        parsed = _SCALES[scale]
    else:
        # list/tuple of [position, color_str]
        key = repr(scale)
        if key not in _SCALES:
            _SCALES[key] = [(p, _parse_rgb(c)) for p, c in scale]
        parsed = _SCALES[key]

    t = float(np.clip(t, 0, 1))
    for i in range(1, len(parsed)):
        p0, c0 = parsed[i - 1]
        p1, c1 = parsed[i]
        if t <= p1:
            f = (t - p0) / (p1 - p0) if p1 > p0 else 0.0
            return (int(c0[0] + f * (c1[0] - c0[0])),
                    int(c0[1] + f * (c1[1] - c0[1])),
                    int(c0[2] + f * (c1[2] - c0[2])))
    return parsed[-1][1]
