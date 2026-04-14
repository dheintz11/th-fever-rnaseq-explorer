"""
Aggregate 48 Salmon gene-level quants from GSE243323 into tidy parquet tables.

Outputs under ../output/:
  sample_metadata.parquet  GSM x (sample_id, subset, temp, replicate, file)
  counts.parquet           gene x sample, Salmon NumReads (integer-ish)
  tpm.parquet              gene x sample, Salmon TPM
  gene_annotation.parquet  gene x (ensembl_id, symbol, name, biotype)

Idempotent. Re-run any time.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
QUANTS = DATA / "quants"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

SERIES_MATRIX = DATA / "GSE243323_series_matrix.txt"


def parse_sample_metadata() -> pd.DataFrame:
    """Parse !Sample_title and !Sample_geo_accession from the series matrix."""
    titles: list[str] = []
    gsms: list[str] = []
    with SERIES_MATRIX.open() as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                gsms = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
    assert len(titles) == len(gsms) == 48, f"expected 48 samples, got {len(titles)}/{len(gsms)}"

    # Title format: "37˚C Th0 Replicate 3"  (note the ˚ char, not °)
    pat = re.compile(r"^(?P<temp>\d+)\S+C\s+(?P<subset>\S+)\s+Replicate\s+(?P<rep>\d+)$")
    rows = []
    for gsm, title in zip(gsms, titles):
        m = pat.match(title)
        if not m:
            raise ValueError(f"unparseable title: {title!r}")
        rows.append(
            {
                "gsm": gsm,
                "title": title,
                "subset": m["subset"],
                "temp_c": int(m["temp"]),
                "replicate": int(m["rep"]),
            }
        )
    meta = pd.DataFrame(rows)
    meta["sample_id"] = (
        meta["subset"] + "_" + meta["temp_c"].astype(str) + "C_r" + meta["replicate"].astype(str)
    )
    meta["condition"] = meta["subset"] + "_" + meta["temp_c"].astype(str) + "C"
    return meta


def find_quant_file(gsm: str) -> Path:
    hits = list(QUANTS.glob(f"{gsm}_*.quant.genes.sf.gz"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{gsm}: found {len(hits)} files, expected 1")
    return hits[0]


def load_quant(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", usecols=["Name", "TPM", "NumReads"])
    return df.rename(columns={"Name": "ensembl_id"}).set_index("ensembl_id")


def build_matrices(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts: dict[str, pd.Series] = {}
    tpm: dict[str, pd.Series] = {}
    index_ref = None
    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="loading quants"):
        q = load_quant(find_quant_file(row.gsm))
        if index_ref is None:
            index_ref = q.index
        elif not q.index.equals(index_ref):
            raise AssertionError(f"{row.gsm}: gene index differs from first sample")
        counts[row.sample_id] = q["NumReads"]
        tpm[row.sample_id] = q["TPM"]
    ordered = meta["sample_id"].tolist()
    return pd.DataFrame(counts)[ordered], pd.DataFrame(tpm)[ordered]


def fetch_gene_annotation(ensembl_ids: list[str]) -> pd.DataFrame:
    """Query mygene.info for symbol/name/biotype. Batched, ~1 min for 55k genes."""
    import mygene

    mg = mygene.MyGeneInfo()
    records = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol,name,type_of_gene",
        species="mouse",
        returnall=False,
        as_dataframe=True,
        df_index=True,
    )
    # Collapse duplicate hits (keep first) and normalize columns.
    records = records[~records.index.duplicated(keep="first")]
    annot = pd.DataFrame(index=pd.Index(ensembl_ids, name="ensembl_id"))
    for col in ("symbol", "name", "type_of_gene"):
        annot[col] = records[col] if col in records.columns else pd.NA
    annot = annot.rename(columns={"type_of_gene": "biotype"})
    annot["symbol"] = annot["symbol"].fillna(annot.index.to_series())
    return annot.reset_index()


def main() -> None:
    print("→ parsing sample metadata")
    meta = parse_sample_metadata()
    print(f"  {len(meta)} samples, subsets={sorted(meta['subset'].unique())}, "
          f"temps={sorted(meta['temp_c'].unique())}")

    print("→ aggregating 48 Salmon quants")
    counts, tpm = build_matrices(meta)
    print(f"  counts: {counts.shape[0]:,} genes x {counts.shape[1]} samples")
    print(f"  tpm:    {tpm.shape[0]:,} genes x {tpm.shape[1]} samples")

    print("→ fetching gene annotation from mygene.info")
    annot = fetch_gene_annotation(counts.index.tolist())
    matched = annot["symbol"].notna().sum()
    print(f"  resolved symbols for {matched:,} / {len(annot):,} genes")

    print("→ writing parquet outputs")
    meta.to_parquet(OUT / "sample_metadata.parquet", index=False)
    counts.reset_index(names="ensembl_id").to_parquet(OUT / "counts.parquet", index=False)
    tpm.reset_index(names="ensembl_id").to_parquet(OUT / "tpm.parquet", index=False)
    annot.to_parquet(OUT / "gene_annotation.parquet", index=False)
    for name in ("sample_metadata", "counts", "tpm", "gene_annotation"):
        p = OUT / f"{name}.parquet"
        print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
