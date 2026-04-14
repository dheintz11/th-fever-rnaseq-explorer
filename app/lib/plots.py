"""Plotly helpers for heatmaps and boxplots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import CONDITION_ORDER, SUBSET_ORDER


def condition_heatmap_single(values: pd.Series, title: str, scale: str) -> go.Figure:
    """One-row heatmap across the 8 conditions (subset x temp)."""
    v = values.reindex(CONDITION_ORDER)
    z = np.array([v.values])
    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=list(v.index), y=[title],
            colorscale="Viridis",
            colorbar={"title": scale, "thickness": 12},
            hovertemplate="<b>%{x}</b><br>" + scale + ": %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=140, margin={"l": 140, "r": 20, "t": 30, "b": 40},
        xaxis={"side": "bottom"},
    )
    return fig


def boxplot_by_condition(expr: pd.Series, meta: pd.DataFrame, symbol: str, scale: str) -> go.Figure:
    """Boxplot: one pair of boxes (37/39°C) per subset."""
    df = expr.rename("value").to_frame()
    df["sample_id"] = df.index
    df = df.merge(meta[["sample_id", "subset", "temp_c"]], on="sample_id")
    df["temp"] = df["temp_c"].astype(str) + "°C"
    fig = px.box(
        df, x="subset", y="value", color="temp",
        category_orders={"subset": SUBSET_ORDER, "temp": ["37°C", "39°C"]},
        points="all",
        color_discrete_map={"37°C": "#4C72B0", "39°C": "#C44E52"},
    )
    fig.update_layout(
        title=f"{symbol} expression by subset × temperature",
        yaxis_title=scale, xaxis_title="", boxmode="group",
        height=380, margin={"l": 60, "r": 20, "t": 50, "b": 40},
        legend={"title": ""},
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.7})
    return fig


def module_activity_boxplot(
    scores_by_sample: pd.Series,    # sample_id -> module score
    meta: pd.DataFrame,
    set_name: str,
) -> go.Figure:
    """Boxplot of per-sample module activity score grouped by subset × temperature.

    Score is the mean z-scored VST across the set's genes (per sample), so a
    value of +1 means the set sits ~1 SD above the cross-sample mean on average.
    """
    df = scores_by_sample.rename("score").to_frame()
    df["sample_id"] = df.index
    df = df.merge(meta[["sample_id", "subset", "temp_c"]], on="sample_id")
    df["temp"] = df["temp_c"].astype(str) + "°C"
    fig = px.box(
        df, x="subset", y="score", color="temp",
        category_orders={"subset": SUBSET_ORDER, "temp": ["37°C", "39°C"]},
        points="all",
        color_discrete_map={"37°C": "#4C72B0", "39°C": "#C44E52"},
    )
    fig.update_layout(
        title=f"Module activity: {set_name}",
        yaxis_title="Mean z-score across module genes",
        xaxis_title="", boxmode="group",
        height=360, margin={"l": 60, "r": 20, "t": 50, "b": 40},
        legend={"title": ""},
    )
    fig.update_traces(marker={"size": 5, "opacity": 0.7})
    fig.add_hline(y=0, line={"color": "#888", "width": 1, "dash": "dot"})
    return fig


def gene_set_heatmap(
    values: pd.DataFrame,   # genes x conditions (mean expression)
    row_labels: list[str],
    *,
    z_score_rows: bool,
    scale: str,
    title: str,
) -> go.Figure:
    """Multi-gene heatmap across the 8 conditions.

    Rows are ordered as given (caller chose ordering). z_score_rows centers/scales
    each row to make cross-condition patterns pop even when absolute scales differ.
    """
    m = values.reindex(columns=CONDITION_ORDER)
    z = m.values
    cbar_title = scale
    if z_score_rows:
        mu = np.nanmean(z, axis=1, keepdims=True)
        sd = np.nanstd(z, axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        z = (z - mu) / sd
        cbar_title = f"{scale} (row z-score)"

    # Divergent palette for z-score, viridis for raw scale.
    colorscale = "RdBu_r" if z_score_rows else "Viridis"
    zmid = 0 if z_score_rows else None

    height = max(260, 18 * len(row_labels) + 120)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(m.columns),
            y=row_labels,
            colorscale=colorscale, zmid=zmid,
            colorbar={"title": cbar_title, "thickness": 12},
            hovertemplate="<b>%{y}</b><br>%{x}<br>" + cbar_title + ": %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=min(1400, height),
        margin={"l": 120, "r": 20, "t": 90, "b": 30},
        xaxis={"side": "top"},
        yaxis={"autorange": "reversed"},
    )
    return fig
