# Th-subset fever response explorer — GSE243323

Interactive reanalysis of bulk RNA-seq from *Subset-specific mitochondrial stress
and DNA damage shape T cell responses to fever and inflammation* (Science
Immunology 2024, DOI `10.1126/sciimmunol.adp3475`).

**Data:** 48 samples = Naive / Th0 / Th17 / iTreg × 37°C / 39°C × 6 replicates.
Deposited Salmon gene-level quants from GEO GSE243323.

## App

`app/Home.py` — Streamlit UI with three tabs:
1. **Individual gene** — symbol/Ensembl lookup → per-sample heatmap + condition boxplot + DE stats across 7 contrasts.
2. **Gene set / module** — 7,783 curated sets (Hallmark, KEGG, Reactome, GO BP, hand-curated "paper modules") → heatmap + per-contrast summary table + module-activity boxplot.
3. **KEGG pathway diagram** — 13 paper-relevant pathways rendered over KEGG's native PNG with expression-colored overlays.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py
```

## Regenerate data (optional)

The repo ships precomputed parquets under `output/` and KEGG PNGs under
`data/kegg_png/`. To rebuild from the raw GEO submission:

```bash
pip install pydeseq2 mygene gseapy Pillow tqdm requests  # dev deps
python scripts/01_build_matrices.py
python scripts/02_normalize_and_de.py
python scripts/03_build_gene_sets.py
python scripts/04_fetch_kegg_pathways.py
```

Step 1 requires the GEO tar (`data/GSE243323_RAW.tar`, ~28 MB) — not committed;
download from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243323.

## Pipeline summary

- **Normalization:** DESeq2 VST (blind) via `pydeseq2`.
- **Differential expression:** Wald tests, 7 contrasts (4 fever: `{subset}_39C` vs `{subset}_37C`; 3 lineage: `{subset}_37C` vs `Naive_37C`).
- **LFC shrinkage:** apeglm. Naive-referenced contrasts shrink from the main fit; within-subset fever contrasts use per-subset mini-fits.
- **Gene sets:** Hallmark/KEGG/Reactome/GO BP pulled via `gseapy` + Enrichr; 8 hand-curated paper modules for the paper's key biology (HSR, cGAS-STING, ETC complex I, p53 apoptosis, UPR, mito biogenesis, DNA damage, glycolysis).
- **Pathway overlay:** KEGG's native PNG as base layer; saturated diverging fills at gene-box coordinates from KGML; gene symbols redrawn in auto-contrast text.
