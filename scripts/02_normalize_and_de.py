"""
Normalize counts (DESeq2 size factors + VST) and run per-contrast DE.

Design:
    ~ condition      where condition = "{subset}_{temp}C"
      (8 levels: Naive/Th0/Th17/iTreg × 37/39)

Contrasts computed:
  Fever response, per subset:    {subset}_39C vs {subset}_37C   (4 contrasts)
  Subset identity at baseline:   Th0/Th17/iTreg_37C vs Naive_37C (3 contrasts)

Outputs under ../output/:
  vst.parquet       gene x sample, variance-stabilized log-like scale
  de_results.parquet   long: (contrast, ensembl_id, log2FC, lfcSE, stat, pvalue, padj, baseMean)
  size_factors.parquet   sample_id -> size_factor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

MIN_COUNT = 10        # keep gene if >= MIN_COUNT in >= MIN_SAMPLES
MIN_SAMPLES = 6       # smallest condition group has 6 replicates


def main() -> None:
    print("→ loading counts + metadata")
    counts = pd.read_parquet(OUT / "counts.parquet").set_index("ensembl_id")
    meta = pd.read_parquet(OUT / "sample_metadata.parquet").set_index("sample_id")
    # Align column order to metadata, and enforce the sample_id order throughout.
    meta = meta.loc[counts.columns]

    # Salmon NumReads are fractional; DESeq2 expects integers.
    counts_int = counts.round().astype("int32")

    n_before = counts_int.shape[0]
    keep = (counts_int >= MIN_COUNT).sum(axis=1) >= MIN_SAMPLES
    counts_int = counts_int.loc[keep]
    print(f"  filtered {n_before:,} → {counts_int.shape[0]:,} genes "
          f"(≥{MIN_COUNT} counts in ≥{MIN_SAMPLES} samples)")

    # Normalize Naïve label for python-friendliness (pydeseq2 tolerates unicode but cleaner to avoid).
    meta = meta.copy()
    meta["subset"] = meta["subset"].str.replace("Naïve", "Naive", regex=False)
    meta["condition"] = meta["subset"] + "_" + meta["temp_c"].astype(str) + "C"

    print("→ fitting DESeq2 (pydeseq2)")
    # DeseqDataSet wants samples x genes.
    dds = DeseqDataSet(
        counts=counts_int.T,
        metadata=meta[["subset", "temp_c", "condition"]],
        design="~condition",
        inference=DefaultInference(n_cpus=4),
        quiet=True,
    )
    dds.deseq2()

    # --- VST (variance-stabilizing transformation) ---
    print("→ VST")
    dds.vst_fit(use_design=False)     # blind VST; standard for exploratory viz
    vst = dds.vst_transform()          # samples x genes, ndarray
    vst_df = pd.DataFrame(vst.T, index=counts_int.index, columns=counts_int.columns)
    vst_df.reset_index().to_parquet(OUT / "vst.parquet", index=False)
    print(f"  vst.parquet: {vst_df.shape[0]:,} genes x {vst_df.shape[1]} samples")

    # --- Size factors ---
    sf = dds.obs["size_factors"].rename("size_factor")
    sf.rename_axis("sample_id").reset_index().to_parquet(OUT / "size_factors.parquet", index=False)
    print(f"  size factors range: {sf.min():.3f} – {sf.max():.3f}")

    # --- Contrasts ---
    subsets = ["Naive", "Th0", "Th17", "iTreg"]
    contrasts: list[tuple[str, str, str]] = []
    # Fever response per subset.
    for s in subsets:
        contrasts.append((f"fever_{s}", f"{s}_39C", f"{s}_37C"))
    # Subset identity at baseline.
    for s in ("Th0", "Th17", "iTreg"):
        contrasts.append((f"lineage_{s}_vs_Naive", f"{s}_37C", "Naive_37C"))

    print(f"→ running {len(contrasts)} contrasts (unshrunken LFCs + Wald stats)")
    frames = []
    for name, a, b in contrasts:
        stats = DeseqStats(dds, contrast=["condition", a, b], quiet=True)
        stats.summary()
        r = stats.results_df.copy()
        r.index.name = "ensembl_id"
        r = r.reset_index()
        r.insert(0, "contrast", name)
        r.insert(1, "numerator", a)
        r.insert(2, "denominator", b)
        n_sig = ((r["padj"] < 0.05) & (r["log2FoldChange"].abs() > 1)).sum()
        print(f"  {name:32s}  n_sig(|lfc|>1, padj<.05) = {n_sig:,}")
        frames.append(r)
    de = pd.concat(frames, ignore_index=True)

    # --- apeglm LFC shrinkage ---
    # apeglm shrinks design-matrix coefficients relative to a reference level.
    # Naive_37C is the default reference in the main fit → free shrinkage for the 4
    # Naive-referenced contrasts. For the 3 within-subset fever contrasts we refit
    # per subset with a minimal ~temp_c design (12 samples each).
    print("→ shrinking LFCs with apeglm")
    shrunk_frames: list[pd.DataFrame] = []

    # Group A: contrasts whose denominator is the main fit's reference.
    print("  main fit covers 4 Naive-referenced contrasts")
    naive_ref_contrasts = [
        ("fever_Naive",            "Naive_39C"),
        ("lineage_Th0_vs_Naive",   "Th0_37C"),
        ("lineage_Th17_vs_Naive",  "Th17_37C"),
        ("lineage_iTreg_vs_Naive", "iTreg_37C"),
    ]
    for name, num in naive_ref_contrasts:
        stats = DeseqStats(dds, contrast=["condition", num, "Naive_37C"], quiet=True)
        stats.summary()
        stats.lfc_shrink(coeff=f"condition[T.{num}]")
        s = stats.results_df[["log2FoldChange"]].rename(
            columns={"log2FoldChange": "log2fc_shrunk"}
        )
        s.index.name = "ensembl_id"
        s = s.reset_index()
        s.insert(0, "contrast", name)
        shrunk_frames.append(s)

    # Group B: per-subset fever contrasts — mini-fit on just that subset's samples.
    for subset in ("Th0", "Th17", "iTreg"):
        print(f"  mini-fit for fever_{subset} ({subset} only, 12 samples)")
        sub_meta = meta[meta["subset"] == subset][["temp_c"]].copy()
        sub_meta["temp_c"] = sub_meta["temp_c"].astype(str) + "C"
        sub_counts = counts_int[sub_meta.index]
        dds_s = DeseqDataSet(
            counts=sub_counts.T,
            metadata=sub_meta,
            design="~temp_c",
            inference=DefaultInference(n_cpus=4),
            quiet=True,
        )
        dds_s.deseq2()
        stats = DeseqStats(dds_s, contrast=["temp_c", "39C", "37C"], quiet=True)
        stats.summary()
        stats.lfc_shrink(coeff="temp_c[T.39C]")
        s = stats.results_df[["log2FoldChange"]].rename(
            columns={"log2FoldChange": "log2fc_shrunk"}
        )
        s.index.name = "ensembl_id"
        s = s.reset_index()
        s.insert(0, "contrast", f"fever_{subset}")
        shrunk_frames.append(s)

    shrunk = pd.concat(shrunk_frames, ignore_index=True)

    # Rename to snake_case and merge shrunken LFCs.
    de = de.rename(columns={
        "baseMean": "base_mean",
        "log2FoldChange": "log2fc",
        "lfcSE": "lfc_se",
        "pvalue": "pvalue",
        "padj": "padj",
    })
    de = de.merge(shrunk, on=["contrast", "ensembl_id"], how="left")
    # Per-subset mini-fits drop genes with 0 counts in that subset (baseMean=0 → LFC undefined).
    # Those genes are also NaN in the unshrunken log2fc column, so no information is lost.
    miss = de.groupby("contrast")["log2fc_shrunk"].apply(lambda s: s.isna().sum())
    if miss.any():
        print("  per-contrast missing shrunken LFCs (expected for 0-count genes):")
        for c, n in miss[miss > 0].items():
            print(f"    {c}: {n}")
    de.to_parquet(OUT / "de_results.parquet", index=False)
    print(f"  de_results.parquet: {len(de):,} rows "
          f"({de['contrast'].nunique()} contrasts × {de['ensembl_id'].nunique():,} genes)")


if __name__ == "__main__":
    main()
