"""Cached parquet loaders for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parents[2] / "output"

SUBSET_ORDER = ["Naive", "Th0", "Th17", "iTreg"]
TEMP_ORDER = [37, 39]
CONDITION_ORDER = [f"{s}_{t}C" for s in SUBSET_ORDER for t in TEMP_ORDER]


@st.cache_data(show_spinner=False)
def load_metadata() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "sample_metadata.parquet")
    df["subset"] = df["subset"].str.replace("Naïve", "Naive", regex=False)
    df["condition"] = df["subset"] + "_" + df["temp_c"].astype(str) + "C"
    df["subset"] = pd.Categorical(df["subset"], categories=SUBSET_ORDER, ordered=True)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df.sort_values(["subset", "temp_c", "replicate"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_annotation() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "gene_annotation.parquet")
    df["symbol"] = df["symbol"].fillna(df["ensembl_id"])
    return df


@st.cache_data(show_spinner=False)
def load_vst() -> pd.DataFrame:
    return pd.read_parquet(OUT / "vst.parquet").set_index("ensembl_id")


@st.cache_data(show_spinner=False)
def load_tpm() -> pd.DataFrame:
    return pd.read_parquet(OUT / "tpm.parquet").set_index("ensembl_id")


@st.cache_data(show_spinner=False)
def load_de() -> pd.DataFrame:
    de = pd.read_parquet(OUT / "de_results.parquet")
    return de


@st.cache_data(show_spinner=False)
def load_gene_sets_index() -> pd.DataFrame:
    return pd.read_parquet(OUT / "gene_sets_index.parquet")


@st.cache_data(show_spinner=False)
def load_gene_sets_long() -> pd.DataFrame:
    return pd.read_parquet(OUT / "gene_sets_long.parquet")


def gene_expression(gene_ens: str, scale: str) -> pd.Series:
    """Expression vector (per sample) on the chosen scale: 'VST' or 'log2(TPM+1)'."""
    if scale == "VST":
        vst = load_vst()
        if gene_ens not in vst.index:
            return pd.Series(dtype=float)
        return vst.loc[gene_ens]
    else:
        tpm = load_tpm()
        if gene_ens not in tpm.index:
            return pd.Series(dtype=float)
        return np.log2(tpm.loc[gene_ens] + 1)


def condition_means(expr: pd.Series, meta: pd.DataFrame) -> pd.Series:
    """Collapse per-sample expression to per-condition mean (Naive_37C, ..., iTreg_39C)."""
    s = expr.rename("value").to_frame()
    s["condition"] = meta.set_index("sample_id").loc[s.index, "condition"].values
    return (s.groupby("condition", observed=True)["value"]
             .mean()
             .reindex(CONDITION_ORDER))


def resolve_symbol(query: str, annot: pd.DataFrame) -> pd.DataFrame:
    """Case-insensitive symbol or Ensembl ID match; returns candidate rows."""
    q = query.strip()
    if not q:
        return annot.head(0)
    # Exact symbol or ensembl match first
    exact = annot[(annot["symbol"].str.upper() == q.upper())
                  | (annot["ensembl_id"] == q)]
    if not exact.empty:
        return exact
    # Prefix match on symbol
    pref = annot[annot["symbol"].str.upper().str.startswith(q.upper())]
    return pref.head(20)
