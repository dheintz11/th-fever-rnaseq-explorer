"""
Download KGML for paper-relevant KEGG pathways; parse nodes, edges, positions.

KEGG KGML fields we care about:
  <entry id N name="mmu:ID [mmu:ID ...]" type="gene">
    <graphics name="Symbol1, Symbol2, ..." x= y= width= height=/>
  </entry>
  <relation entry1=.. entry2=.. type="GErel|PPrel|PCrel|ECrel">
    <subtype name="activation|inhibition|phosphorylation|..."/>
  </relation>

We map entries to our mouse Ensembl IDs via the FIRST symbol in graphics.name
(case-insensitive match against our gene_annotation table). Entries with no
symbol match stay in the diagram but are coloured grey.

Outputs under ../output/:
  pathway_index.parquet    (pathway_id, title, n_entries, n_relations, w, h)
  pathway_nodes.parquet    (pathway_id, entry_id, symbols, primary_symbol,
                            ensembl_id, x, y, width, height, kind)
  pathway_edges.parquet    (pathway_id, src_entry, dst_entry, type, subtypes)
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
KGML_DIR = ROOT / "data" / "kgml"
PNG_DIR = ROOT / "data" / "kegg_png"
KGML_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

# Curated list of paper-relevant KEGG pathways (mouse).
PATHWAYS: dict[str, str] = {
    "mmu04115": "p53 signaling pathway",
    "mmu04210": "Apoptosis",
    "mmu04217": "Necroptosis",
    "mmu04218": "Cellular senescence",
    "mmu04141": "Protein processing in endoplasmic reticulum",
    "mmu04623": "Cytosolic DNA-sensing pathway",
    "mmu04668": "TNF signaling pathway",
    "mmu04064": "NF-kappa B signaling pathway",
    "mmu04630": "JAK-STAT signaling pathway",
    "mmu00010": "Glycolysis / Gluconeogenesis",
    "mmu00020": "Citrate cycle (TCA cycle)",
    "mmu00190": "Oxidative phosphorylation",
    "mmu04066": "HIF-1 signaling pathway",
}


def download_kgml(pathway_id: str) -> Path:
    out = KGML_DIR / f"{pathway_id}.xml"
    if out.exists():
        return out
    url = f"https://rest.kegg.jp/get/{pathway_id}/kgml"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out.write_text(r.text)
    time.sleep(0.3)  # be polite to KEGG
    return out


def download_png(pathway_id: str) -> Path:
    """KEGG's native pathway image. Provides the laid-out arrows and labels."""
    out = PNG_DIR / f"{pathway_id}.png"
    if out.exists():
        return out
    url = f"https://www.kegg.jp/kegg/pathway/mmu/{pathway_id}.png"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out.write_bytes(r.content)
    time.sleep(0.3)
    return out


def parse_kgml(path: Path, pathway_id: str, sym2ens: dict[str, str]) -> tuple[
    dict, pd.DataFrame, pd.DataFrame
]:
    tree = ET.parse(path)
    root = tree.getroot()
    title = root.attrib.get("title", "")

    nodes: list[dict] = []
    max_x = 0.0
    max_y = 0.0
    for entry in root.findall("entry"):
        g = entry.find("graphics")
        if g is None:
            continue
        try:
            x = float(g.attrib.get("x", "nan"))
            y = float(g.attrib.get("y", "nan"))
            w = float(g.attrib.get("width", "46"))
            h = float(g.attrib.get("height", "17"))
        except ValueError:
            continue
        if pd.isna(x) or pd.isna(y):
            continue
        name = g.attrib.get("name", "")
        # graphics.name is "Sym1, Sym2, ..." — strip trailing "...".
        symbols = [s.strip().rstrip(".").strip() for s in name.split(",")] if name else []
        symbols = [s for s in symbols if s and s != "..."]
        primary = symbols[0] if symbols else None
        ens = sym2ens.get(primary.upper()) if primary else None
        nodes.append({
            "pathway_id": pathway_id,
            "entry_id": entry.attrib["id"],
            "kegg_names": entry.attrib.get("name", ""),
            "symbols": ", ".join(symbols),
            "primary_symbol": primary,
            "ensembl_id": ens,
            "kind": entry.attrib.get("type", ""),
            "x": x, "y": y, "width": w, "height": h,
        })
        max_x = max(max_x, x + w / 2)
        max_y = max(max_y, y + h / 2)

    edges: list[dict] = []
    for rel in root.findall("relation"):
        subs = ", ".join(s.attrib.get("name", "") for s in rel.findall("subtype"))
        edges.append({
            "pathway_id": pathway_id,
            "src_entry": rel.attrib["entry1"],
            "dst_entry": rel.attrib["entry2"],
            "type": rel.attrib.get("type", ""),
            "subtypes": subs,
        })

    info = {
        "pathway_id": pathway_id,
        "title": title,
        "n_entries": len(nodes),
        "n_relations": len(edges),
        "w": max_x + 20, "h": max_y + 20,
    }
    return info, pd.DataFrame(nodes), pd.DataFrame(edges)


def main() -> None:
    print(f"→ target: {len(PATHWAYS)} KEGG pathways")

    # Build (upper-case) symbol → ensembl lookup from our annotation.
    annot = pd.read_parquet(OUT / "gene_annotation.parquet").dropna(subset=["symbol"])
    annot["U"] = annot["symbol"].str.upper()
    sym2ens = dict(annot.drop_duplicates("U").set_index("U")["ensembl_id"])
    print(f"  sym→ens lookup has {len(sym2ens):,} entries")

    from PIL import Image

    info_rows, node_frames, edge_frames = [], [], []
    for pid, pname in tqdm(PATHWAYS.items(), desc="fetching KGML + PNG"):
        path = download_kgml(pid)
        png_path = download_png(pid)
        info, nodes, edges = parse_kgml(path, pid, sym2ens)
        # Stamp the PNG dimensions onto the pathway index (coord system = PNG pixels).
        with Image.open(png_path) as im:
            info["png_w"], info["png_h"] = im.size
        info["png_path"] = str(png_path.relative_to(ROOT))
        info_rows.append(info)
        node_frames.append(nodes)
        edge_frames.append(edges)

    pathway_idx = pd.DataFrame(info_rows)
    pathway_idx["name"] = pathway_idx["pathway_id"].map(PATHWAYS)
    nodes_df = pd.concat(node_frames, ignore_index=True)
    edges_df = pd.concat(edge_frames, ignore_index=True)

    print("\n→ coverage per pathway (gene nodes resolved to our Ensembl IDs):")
    gene_only = nodes_df[nodes_df["kind"] == "gene"].copy()
    cov = (gene_only.groupby("pathway_id")
                    .agg(n_gene_nodes=("entry_id", "size"),
                         n_mapped=("ensembl_id", lambda s: s.notna().sum())))
    cov = cov.join(pathway_idx.set_index("pathway_id")["name"]).reset_index()
    cov["pct"] = (cov["n_mapped"] / cov["n_gene_nodes"] * 100).round(1)
    print(cov.to_string(index=False))

    pathway_idx.to_parquet(OUT / "pathway_index.parquet", index=False)
    nodes_df.to_parquet(OUT / "pathway_nodes.parquet", index=False)
    edges_df.to_parquet(OUT / "pathway_edges.parquet", index=False)
    for n in ("pathway_index", "pathway_nodes", "pathway_edges"):
        p = OUT / f"{n}.parquet"
        print(f"  {p.name}: {p.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
