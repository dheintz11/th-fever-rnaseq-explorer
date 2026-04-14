"""
Build the gene-set resolver: long table mapping (source, gene_set) -> ensembl_id.

Sources:
  1. MSigDB Hallmark 2020          (50 curated modules, human symbols)
  2. KEGG 2019 Mouse               (303 pathways)
  3. Reactome Pathways 2024        (~2,100 pathways)
  4. GO Biological Process 2025    (~5,300 terms)
  5. Paper modules                 (hand-curated, matched to paper figures)

All non-"paper" libraries ship with human symbols (upper-case). We match them to
our mouse Ensembl IDs via case-insensitive symbol lookup, which covers the ~95%
of mouse genes that are title-case versions of their human ortholog.

Outputs under ../output/:
  gene_sets_long.parquet      (source, gene_set, ensembl_id, symbol)
  gene_sets_index.parquet     (source, gene_set, n_genes, n_mapped)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import gseapy
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
CACHE = ROOT / "data" / "gene_set_cache"
CACHE.mkdir(parents=True, exist_ok=True)

LIBRARIES = [
    ("Hallmark", "MSigDB_Hallmark_2020"),
    ("KEGG",     "KEGG_2019_Mouse"),
    ("Reactome", "Reactome_Pathways_2024"),
    ("GO_BP",    "GO_Biological_Process_2025"),
]

# Hand-curated modules keyed to the paper's biology. Mouse symbols (title-case).
PAPER_MODULES: dict[str, list[str]] = {
    "Heat_Shock_Response": [
        "Hspa1a", "Hspa1b", "Hspa2", "Hspa4", "Hspa4l", "Hspa5", "Hspa8", "Hspa9",
        "Hspb1", "Hspb8", "Hsph1", "Hsp90aa1", "Hsp90ab1", "Hsp90b1",
        "Dnaja1", "Dnaja4", "Dnajb1", "Dnajb4", "Dnajb9", "Dnajb11",
        "Hspe1", "Hspd1", "Bag3", "Hsf1", "Hsf2",
    ],
    "cGAS_STING_Pathway": [
        "Cgas", "Sting1", "Tmem173", "Tbk1", "Ikbke", "Irf3", "Irf7",
        "Ifnb1", "Ifnar1", "Ifnar2", "Mavs", "Nfkb1", "Rela",
    ],
    "ETC_Complex_I": [
        "Ndufa1","Ndufa2","Ndufa3","Ndufa4","Ndufa5","Ndufa6","Ndufa7","Ndufa8",
        "Ndufa9","Ndufa10","Ndufa11","Ndufa12","Ndufa13",
        "Ndufb1","Ndufb2","Ndufb3","Ndufb4","Ndufb5","Ndufb6","Ndufb7","Ndufb8",
        "Ndufb9","Ndufb10","Ndufb11",
        "Ndufc1","Ndufc2",
        "Ndufs1","Ndufs2","Ndufs3","Ndufs4","Ndufs5","Ndufs6","Ndufs7","Ndufs8",
        "Ndufv1","Ndufv2","Ndufv3",
    ],
    "p53_Apoptosis_Targets": [
        "Trp53", "Mdm2", "Mdm4", "Cdkn1a", "Bax", "Bak1", "Bid", "Bbc3", "Pmaip1",
        "Fas", "Fasl", "Bcl2", "Bcl2l1", "Bcl2l11", "Apaf1", "Casp3", "Casp7",
        "Casp8", "Casp9", "Gadd45a", "Gadd45b", "Gadd45g", "Zmat3", "Tp53i3",
    ],
    "Unfolded_Protein_Response": [
        "Atf4", "Atf6", "Atf3", "Ddit3", "Hspa5", "Xbp1",
        "Eif2ak3", "Ern1", "Eif2s1", "Ppp1r15a", "Ppp1r15b",
        "Herpud1", "Sel1l", "Hyou1", "Derl1", "Edem1",
    ],
    "Mitochondrial_Biogenesis": [
        "Ppargc1a", "Ppargc1b", "Nrf1", "Tfam", "Tfb1m", "Tfb2m",
        "Nfe2l2", "Esrra", "Esrrb", "Polrmt", "Mterf1",
        "Pprc1", "Yy1",
    ],
    "DNA_Damage_Response": [
        "Atm", "Atr", "Chek1", "Chek2", "H2ax", "H2afx", "Mdc1", "Mre11a",
        "Rad50", "Nbn", "Brca1", "Brca2", "Rad51", "Parp1", "Parp2",
        "Tp53bp1", "Xrcc4", "Xrcc6", "Prkdc",
    ],
    "Glycolysis_Core": [
        "Hk1", "Hk2", "Hk3", "Gpi1", "Pfkl", "Pfkm", "Pfkp", "Aldoa", "Aldob",
        "Tpi1", "Gapdh", "Pgk1", "Pgam1", "Eno1", "Eno2", "Eno3",
        "Pkm", "Ldha", "Ldhb", "Slc2a1", "Slc2a3",
    ],
}


def cached_get(library: str) -> dict[str, list[str]]:
    cache = CACHE / f"{library}.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    print(f"  fetching {library} from Enrichr")
    data = gseapy.get_library(library, organism="Mouse")
    with cache.open("wb") as fh:
        pickle.dump(data, fh)
    return data


def main() -> None:
    print("→ loading gene annotation for symbol lookup")
    annot = pd.read_parquet(OUT / "gene_annotation.parquet")
    # Map UPPER(symbol) -> ensembl_id. Keep first if duplicate.
    annot = annot.dropna(subset=["symbol"])
    annot["SYMBOL_UPPER"] = annot["symbol"].str.upper()
    sym2ens = (annot.drop_duplicates("SYMBOL_UPPER")
                    .set_index("SYMBOL_UPPER")[["ensembl_id", "symbol"]])

    def lookup(symbols: list[str]) -> pd.DataFrame:
        hits = sym2ens.reindex([s.upper() for s in symbols]).dropna()
        return hits.reset_index(drop=True)

    long_rows: list[pd.DataFrame] = []
    index_rows: list[dict] = []

    print("→ pulling libraries")
    for source, library in LIBRARIES:
        sets = cached_get(library)
        print(f"  {source}: {len(sets):,} sets")
        for set_name, symbols in sets.items():
            mapped = lookup(symbols)
            if mapped.empty:
                continue
            mapped = mapped.copy()
            mapped.insert(0, "gene_set", set_name)
            mapped.insert(0, "source", source)
            long_rows.append(mapped)
            index_rows.append({
                "source": source,
                "gene_set": set_name,
                "n_genes": len(symbols),
                "n_mapped": len(mapped),
            })

    print(f"  Paper modules: {len(PAPER_MODULES)} hand-curated sets")
    for set_name, symbols in PAPER_MODULES.items():
        mapped = lookup(symbols)
        mapped = mapped.copy()
        mapped.insert(0, "gene_set", set_name)
        mapped.insert(0, "source", "Paper")
        long_rows.append(mapped)
        index_rows.append({
            "source": "Paper",
            "gene_set": set_name,
            "n_genes": len(symbols),
            "n_mapped": len(mapped),
        })

    long_df = pd.concat(long_rows, ignore_index=True)
    idx_df = pd.DataFrame(index_rows)

    # Report mapping coverage per source.
    print("\n→ mapping coverage per source:")
    cov = idx_df.groupby("source").agg(
        n_sets=("gene_set", "size"),
        n_mapped_total=("n_mapped", "sum"),
        n_genes_total=("n_genes", "sum"),
    )
    cov["coverage_pct"] = (cov["n_mapped_total"] / cov["n_genes_total"] * 100).round(1)
    print(cov.to_string())

    long_df.to_parquet(OUT / "gene_sets_long.parquet", index=False)
    idx_df.to_parquet(OUT / "gene_sets_index.parquet", index=False)
    print(f"\n→ wrote:")
    print(f"  gene_sets_long.parquet:  {len(long_df):,} rows, "
          f"{(OUT / 'gene_sets_long.parquet').stat().st_size / 1e6:.2f} MB")
    print(f"  gene_sets_index.parquet: {len(idx_df):,} rows, "
          f"{(OUT / 'gene_sets_index.parquet').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
