"""Th fever RNA-seq explorer — Streamlit UI.

Data: GSE243323 (4 CD4 T subsets × 37/39°C × 6 reps; no Th1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib.data import (
    CONDITION_ORDER,
    SUBSET_ORDER,
    condition_means,
    gene_expression,
    load_annotation,
    load_de,
    load_gene_sets_index,
    load_gene_sets_long,
    load_metadata,
    resolve_symbol,
)
from lib.plots import (
    boxplot_by_condition,
    condition_heatmap_single,
    gene_set_heatmap,
    module_activity_boxplot,
)
from lib.pathway_view import (
    load_pathway_index,
    load_pathway_nodes,
    render_pathway,
)

st.set_page_config(page_title="Th Fever RNA-seq", layout="wide", page_icon="🧬")

meta = load_metadata()
annot = load_annotation()

st.title("Th-subset fever response — GSE243323")
st.caption(
    "Bulk RNA-seq. 4 subsets × 2 temperatures × 6 replicates = 48 samples. "
    "Expression scale toggles between VST (best for cross-gene comparison) and log2(TPM+1) "
    "(most intuitive biologically)."
)

# ---------------- global controls ---------------------------------------------
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    scale = st.radio(
        "Expression scale", ["VST", "log2(TPM+1)"],
        horizontal=True, index=0,
        help="VST for heatmap comparison across genes; log2(TPM+1) for absolute biology.",
    )

tab_gene, tab_set, tab_path = st.tabs(
    ["🔬 Individual gene", "📚 Gene set / module", "🗺️ Pathway diagram"]
)

# ---------------- tab 1: individual gene --------------------------------------
with tab_gene:
    q = st.text_input(
        "Gene symbol or Ensembl ID", value="Hspa1a",
        help="Case-insensitive. Prefix match shows up to 20 candidates.",
    )
    candidates = resolve_symbol(q, annot)
    if candidates.empty:
        st.warning(f"No genes match '{q}'.")
    else:
        if len(candidates) > 1:
            opt_labels = [
                f"{r.symbol}  ({r.ensembl_id})" + (f"  — {r['name']}" if pd.notna(r['name']) else "")
                for _, r in candidates.iterrows()
            ]
            choice = st.selectbox(f"{len(candidates)} matches — pick one", options=opt_labels)
            ens = candidates.iloc[opt_labels.index(choice)]["ensembl_id"]
        else:
            ens = candidates.iloc[0]["ensembl_id"]
        row = annot.set_index("ensembl_id").loc[ens]
        sym = row["symbol"]
        name = row["name"] if pd.notna(row["name"]) else ""
        biotype = row["biotype"] if pd.notna(row["biotype"]) else ""

        st.markdown(f"### `{sym}`  \n**{name}**  \nEnsembl: `{ens}` · type: `{biotype}`")

        expr = gene_expression(ens, scale)
        if expr.empty:
            st.error(f"`{sym}` not in normalized expression matrix — probably filtered for low counts.")
        else:
            # Per-condition means heatmap
            cm = condition_means(expr, meta)
            st.plotly_chart(
                condition_heatmap_single(cm, title=sym, scale=scale),
                width='stretch',
            )

            # Per-sample boxplot
            st.plotly_chart(
                boxplot_by_condition(expr, meta, symbol=sym, scale=scale),
                width='stretch',
            )

            # DE table
            st.markdown("#### Differential expression across all contrasts")
            de = load_de()
            sub = de[de["ensembl_id"] == ens].copy()
            if sub.empty:
                st.info("No DE stats available for this gene (filtered from DE fit).")
            else:
                sub = sub[["contrast", "numerator", "denominator",
                           "base_mean", "log2fc", "log2fc_shrunk", "padj", "pvalue"]]
                sub = sub.sort_values("padj", na_position="last").reset_index(drop=True)
                st.dataframe(
                    sub,
                    width='stretch', hide_index=True,
                    column_config={
                        "base_mean":    st.column_config.NumberColumn(format="%.1f"),
                        "log2fc":       st.column_config.NumberColumn("log2FC (raw)", format="%.3f"),
                        "log2fc_shrunk":st.column_config.NumberColumn("log2FC (apeglm)", format="%.3f"),
                        "padj":         st.column_config.NumberColumn(format="%.2e"),
                        "pvalue":       st.column_config.NumberColumn(format="%.2e"),
                    },
                )

# ---------------- tab 2: gene set / module ------------------------------------
with tab_set:
    idx = load_gene_sets_index()
    long = load_gene_sets_long()

    sc1, sc2 = st.columns([1, 3])
    with sc1:
        src = st.selectbox(
            "Source",
            options=["Paper", "Hallmark", "KEGG", "Reactome", "GO_BP"],
            index=0,
            help="Paper = hand-curated modules matched to the paper's figures.",
        )
    with sc2:
        search = st.text_input(
            "Filter gene sets", value="",
            help="Case-insensitive substring match on the gene-set name.",
        )

    src_idx = idx[idx["source"] == src].sort_values("n_mapped", ascending=False)
    if search.strip():
        src_idx = src_idx[src_idx["gene_set"].str.contains(search, case=False, regex=False)]
    if src_idx.empty:
        st.warning("No gene sets match that filter.")
        st.stop()

    options = src_idx.apply(
        lambda r: f"{r['gene_set']}  ·  n={r['n_mapped']}",
        axis=1,
    ).tolist()
    pick = st.selectbox(f"Gene set ({len(options):,} available)", options=options)
    set_name = src_idx.iloc[options.index(pick)]["gene_set"]
    set_row = src_idx.iloc[options.index(pick)]

    st.markdown(
        f"**{set_name}** — {set_row['n_mapped']} of {set_row['n_genes']} genes mapped to this dataset."
    )

    members = long[(long["source"] == src) & (long["gene_set"] == set_name)]
    ens_ids = members["ensembl_id"].tolist()

    # Controls for heatmap shaping
    ctl1, ctl2, ctl3 = st.columns(3)
    with ctl1:
        de_contrast = st.selectbox(
            "Order rows by log2FC in contrast",
            options=["(none — alphabetical)"] + sorted(load_de()["contrast"].unique().tolist()),
            index=1 if "fever_Th17" in load_de()["contrast"].unique() else 0,
        )
    with ctl2:
        top_n = st.slider("Max genes shown", min_value=10, max_value=300, value=60, step=10)
    with ctl3:
        z_score = st.checkbox("Z-score rows (center per-gene)", value=True)

    # Build the expression matrix for this gene set.
    from lib.data import load_vst, load_tpm  # local import to respect cache

    expr_mat = load_vst() if scale == "VST" else np.log2(load_tpm() + 1)
    present = [e for e in ens_ids if e in expr_mat.index]
    if not present:
        st.error("None of this set's genes passed the low-count filter.")
        st.stop()
    mat = expr_mat.loc[present]
    # Collapse to per-condition means
    sample_to_cond = meta.set_index("sample_id")["condition"]
    # Collapse per-sample columns to per-condition means via transpose idiom.
    cond_means = (mat.T
                    .groupby(sample_to_cond.reindex(mat.columns).values, observed=True)
                    .mean()
                    .T)
    cond_means = cond_means.reindex(columns=CONDITION_ORDER)

    # Row ordering
    sym_lookup = annot.set_index("ensembl_id")["symbol"]
    if de_contrast != "(none — alphabetical)":
        de = load_de()
        d = (de[(de["contrast"] == de_contrast) & (de["ensembl_id"].isin(cond_means.index))]
             .set_index("ensembl_id"))
        order_key = d["log2fc_shrunk"].reindex(cond_means.index).fillna(d["log2fc"]).fillna(0)
        cond_means = cond_means.assign(_ord=order_key).sort_values("_ord", ascending=False).drop(columns="_ord")
    else:
        row_sym = sym_lookup.reindex(cond_means.index)
        cond_means = cond_means.assign(_sym=row_sym).sort_values("_sym").drop(columns="_sym")

    # Cap to top_n
    cond_means = cond_means.head(top_n)
    row_labels = sym_lookup.reindex(cond_means.index).fillna(cond_means.index.to_series()).tolist()

    st.plotly_chart(
        gene_set_heatmap(
            cond_means, row_labels,
            z_score_rows=z_score, scale=scale,
            title=f"{set_name}  ({len(cond_means)} of {len(present)} genes shown)",
        ),
        width='stretch',
    )

    # ---- module-level quantification ----
    st.markdown("#### Module-level activity")
    module_cols = st.columns([5, 4])

    # Per-contrast summary: mean/median LFC + direction counts
    de_all = load_de()
    de_sub = de_all[de_all["ensembl_id"].isin(present)].copy()
    de_sub["sig_up"] = (de_sub["padj"] < 0.05) & (de_sub["log2fc_shrunk"] > 0)
    de_sub["sig_down"] = (de_sub["padj"] < 0.05) & (de_sub["log2fc_shrunk"] < 0)
    summary = (de_sub
               .groupby("contrast")
               .agg(
                   n_genes=("log2fc_shrunk", lambda s: s.notna().sum()),
                   mean_lfc=("log2fc_shrunk", "mean"),
                   median_lfc=("log2fc_shrunk", "median"),
                   pct_up=("log2fc_shrunk", lambda s: (s > 0).mean() * 100),
                   n_sig_up=("sig_up", "sum"),
                   n_sig_down=("sig_down", "sum"),
               )
               .reset_index())
    # Put fever_* rows first, then lineage_*
    summary["_order"] = summary["contrast"].map(
        lambda c: (0 if c.startswith("fever_") else 1, c)
    )
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    with module_cols[0]:
        st.markdown("**Per-contrast summary (across all set genes):**")
        st.dataframe(
            summary,
            width="stretch", hide_index=True,
            column_config={
                "n_genes":    st.column_config.NumberColumn("n", format="%d"),
                "mean_lfc":   st.column_config.NumberColumn("mean log2FC (shrunk)", format="%.3f"),
                "median_lfc": st.column_config.NumberColumn("median log2FC", format="%.3f"),
                "pct_up":     st.column_config.NumberColumn("% up", format="%.0f%%"),
                "n_sig_up":   st.column_config.NumberColumn("n sig↑", format="%d"),
                "n_sig_down": st.column_config.NumberColumn("n sig↓", format="%d"),
            },
        )
        st.caption(
            "Significance: padj<0.05. Positive mean log2FC means the module as "
            "a whole is up-regulated at the numerator (39°C for fever_*, the "
            "differentiated subset for lineage_*)."
        )

    # Module activity score: per-sample mean of z-scored VST across module genes
    from lib.data import load_vst
    vst_mat = load_vst().loc[present]
    row_means = vst_mat.mean(axis=1)
    row_sds = vst_mat.std(axis=1).replace(0, 1.0)
    z_mat = vst_mat.sub(row_means, axis=0).div(row_sds, axis=0)
    module_scores = z_mat.mean(axis=0)  # sample_id -> score

    with module_cols[1]:
        st.plotly_chart(
            module_activity_boxplot(module_scores, meta, set_name=set_name),
            width="stretch",
        )

    # DE table for the set
    with st.expander("DE stats for all genes in this set (all contrasts)", expanded=False):
        de = load_de()
        sub = de[de["ensembl_id"].isin(present)].merge(
            annot[["ensembl_id", "symbol"]], on="ensembl_id"
        )
        cols = ["symbol", "contrast", "base_mean", "log2fc_shrunk", "padj"]
        st.dataframe(
            sub[cols].sort_values(["contrast", "padj"]).reset_index(drop=True),
            width='stretch', hide_index=True,
            column_config={
                "base_mean": st.column_config.NumberColumn(format="%.1f"),
                "log2fc_shrunk": st.column_config.NumberColumn("log2FC (apeglm)", format="%.3f"),
                "padj": st.column_config.NumberColumn(format="%.2e"),
            },
        )
        csv = sub[cols].to_csv(index=False).encode()
        st.download_button(
            f"Download {set_name} DE table (CSV)",
            data=csv,
            file_name=f"{src}_{set_name.replace(' ','_')}_DE.csv",
            mime="text/csv",
        )

# ---------------- tab 3: KEGG pathway diagram ---------------------------------
with tab_path:
    pidx = load_pathway_index().sort_values("name")
    pc1, pc2, pc3 = st.columns([2, 2, 2])
    with pc1:
        path_labels = [f"{r['pathway_id']} — {r['name']}" for _, r in pidx.iterrows()]
        picked = st.selectbox("KEGG pathway", options=path_labels, index=0)
        picked_id = picked.split(" — ")[0]
    with pc2:
        contrast_options = sorted(load_de()["contrast"].unique().tolist())
        # Pull fever_* to the top
        contrast_options = sorted(contrast_options, key=lambda c: (not c.startswith("fever_"), c))
        chosen_contrast = st.selectbox(
            "Colour nodes by log2FC in contrast",
            options=contrast_options,
            index=0,
        )
    with pc3:
        use_shrunken = st.checkbox("apeglm-shrunken log2FC", value=True,
                                   help="Recommended. Unchecking uses raw Wald log2FC.")

    # Build the per-gene value series for the chosen contrast.
    de = load_de()
    col = "log2fc_shrunk" if use_shrunken else "log2fc"
    gene_value = (de[de["contrast"] == chosen_contrast]
                    .set_index("ensembl_id")[col])

    # Color-range slider so modest effects aren't washed out by a single outlier.
    range_col1, range_col2 = st.columns([2, 3])
    with range_col1:
        color_cap = st.slider(
            "Color range (±log2FC)", min_value=0.5, max_value=5.0, value=2.0, step=0.25,
            help="Tighter range makes small effects more visible; wider range keeps "
                 "extreme genes distinguishable.",
        )

    fig = render_pathway(
        picked_id, gene_value,
        value_label=f"log2FC ({chosen_contrast})",
        value_range=(-color_cap, color_cap),
        diverging=True,
    )
    # width="content" lets the figure render at its native pixel dimensions
    # (KEGG's own pathway image resolution) rather than stretching to column.
    st.plotly_chart(fig, width="content")

    # Diagnostic summary for this pathway × contrast
    nodes = load_pathway_nodes()
    pn = nodes[(nodes["pathway_id"] == picked_id) & (nodes["kind"] == "gene")].copy()
    pn["value"] = pn["ensembl_id"].map(gene_value)
    n_gene = len(pn)
    n_mapped = pn["ensembl_id"].notna().sum()
    n_values = pn["value"].notna().sum()
    sig = de[(de["contrast"] == chosen_contrast) & (de["padj"] < 0.05)
             & (de["ensembl_id"].isin(pn["ensembl_id"].dropna()))]
    st.markdown(
        f"**Pathway coverage:** {n_gene} gene nodes in KGML · "
        f"{n_mapped} mapped to Ensembl · {n_values} with expression data · "
        f"**{len(sig)} significant** (padj<0.05) in this contrast."
    )

    # Legend
    with st.expander("Legend", expanded=False):
        st.markdown(
            "- **Base image** — KEGG's own rendered pathway (arrows, labels, compounds are all KEGG's work).\n"
            "- **Red overlay** — gene up-regulated at numerator vs denominator.\n"
            "- **Blue overlay** — gene down-regulated.\n"
            "- **Overlay intensity** scales with |log2FC| — small changes barely tint the box.\n"
            "- **Dashed outline only** — gene present in pathway but no expression data "
            "(unmapped Ensembl ID or filtered by the low-count cutoff).\n"
            "- Hover any gene box for symbol, value, and entry ID.\n"
            "- Zoom and pan with Plotly controls (top-right)."
        )

# ---------------- footer ------------------------------------------------------
st.divider()
st.caption(
    f"48 samples · {len(annot):,} genes annotated · "
    f"{len(load_de()['contrast'].unique())} contrasts · "
    f"{len(load_gene_sets_index()):,} gene sets (Paper/Hallmark/KEGG/Reactome/GO_BP)."
)
