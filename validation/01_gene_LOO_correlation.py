"""
validation/01_gene_LOO_correlation.py
──────────────────────────────────────
LOO Spearman correlation for all 100 CC gene candidates on the public
SciPlex3 dataset. Generates threshold-filtered gene list JSONs used by
02_validate_PI_sciplex.py.

This script is part of the validation workflow for the proliferation_index
package. It is NOT the end-user tool — see ../compute_pi.py for that.

OUTPUT
──────
  ../results/loo_correlation.csv          (wide: one row per gene)
  ../results/loo_correlation_long.csv     (long: one row per gene × dataset)
  ../gene_lists/cc_genes_r02.json         (r_mean ≥ 0.2)
  ../gene_lists/cc_genes_r03.json         (r_mean ≥ 0.3)
  ../gene_lists/cc_genes_r04.json         (r_mean ≥ 0.4)

USAGE
─────
  conda run -n scvi python validation/01_gene_LOO_correlation.py

  # Custom paths
  conda run -n scvi python validation/01_gene_LOO_correlation.py \\
      --sciplex /path/to/sciplex3_atomicPI2.symbolVar.h5ad

MEMORY ESTIMATE
───────────────
  581,777 cells × 100 genes × 4 B ≈ 233 MB
  Recommended: ≥ 4 GB RAM
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pi_functions import compute_pi, load_cc_counts, load_gene_dict

DEFAULT_SCIPLEX = (Path("/mnt/lab-store/projects/ATOMIC/analysis/dong")
                   / "object/SciPlexGxE/sciplex3_atomicPI2.symbolVar.h5ad")
DEFAULT_GENES   = ROOT / "gene_lists/cc_genes_pool.json"
DEFAULT_OUT     = ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sciplex",   default=str(DEFAULT_SCIPLEX), help="Path to SciPlex3 symbolVar h5ad")
    p.add_argument("--gene-list", default=str(DEFAULT_GENES),   help="Path to cc_genes_pool.json")
    p.add_argument("--out",       default=str(DEFAULT_OUT),     help="Output directory")
    return p.parse_args()


def compute_loo_correlations(counts, libsize, present_genes, s_pool, g2m_pool, dataset_name):
    gene_to_col = {g: i for i, g in enumerate(present_genes)}
    s_present   = [g for g in s_pool   if g in gene_to_col]
    g2m_present = [g for g in g2m_pool if g in gene_to_col]

    print(f"  S: {len(s_present)}/{len(s_pool)},  G2M: {len(g2m_present)}/{len(g2m_pool)}")

    records = []
    for phase_genes, other_genes, phase_label in [
        (s_present, g2m_present, "S"),
        (g2m_present, s_present, "G2M"),
    ]:
        for gene in phase_genes:
            loo_same  = [gene_to_col[g] for g in phase_genes if g != gene]
            other_idx = [gene_to_col[g] for g in other_genes]
            s_idx, g2m_idx = (
                (loo_same, other_idx) if phase_label == "S" else (other_idx, loo_same)
            )
            pi_loo    = compute_pi(counts, libsize, s_idx, g2m_idx)
            gene_expr = counts[:, gene_to_col[gene]].ravel()
            r, pval   = spearmanr(gene_expr, pi_loo)
            records.append({
                "gene": gene, "phase": phase_label, "dataset": dataset_name,
                "spearman_r": float(r), "p_value": float(pval),
            })
    return pd.DataFrame(records)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_dict = load_gene_dict(args.gene_list)
    s_pool    = gene_dict["s.genes"]
    g2m_pool  = gene_dict["g2m.genes"]
    all_pool  = s_pool + g2m_pool
    print(f"Gene pool: S={len(s_pool)}, G2M={len(g2m_pool)}, total={len(all_pool)}")

    if not Path(args.sciplex).exists():
        print(f"ERROR: SciPlex3 file not found: {args.sciplex}")
        sys.exit(1)

    print(f"\nDataset: SciPlex3")
    counts, libsize, present, missing = load_cc_counts(args.sciplex, all_pool)
    print(f"  Cells: {counts.shape[0]:,}  |  genes: {len(present)}/{len(all_pool)}")
    if missing:
        print(f"  Absent: {missing}")

    df_long = compute_loo_correlations(counts, libsize, present, s_pool, g2m_pool, "SciPlex3")
    del counts

    # Rename spearman_r to r_SciPlex3 for consistency in wide format
    df_wide = df_long[["gene", "phase", "spearman_r", "p_value"]].copy()
    df_wide = df_wide.rename(columns={"spearman_r": "r_SciPlex3"})
    df_wide["r_mean"] = df_wide["r_SciPlex3"]   # single dataset → r_mean == r_SciPlex3
    df_wide = df_wide.sort_values(["phase", "r_mean"], ascending=[True, False]).reset_index(drop=True)

    # ── Save LOO results ───────────────────────────────────────────────────────
    df_wide.to_csv(out_dir / "loo_correlation.csv", index=False)
    df_long.to_csv(out_dir / "loo_correlation_long.csv", index=False)
    print(f"\nResults → {out_dir / 'loo_correlation.csv'}")

    # ── Save threshold-filtered gene lists ─────────────────────────────────────
    gene_list_dir = ROOT / "gene_lists"
    gene_to_phase = {g: "S" for g in s_pool}
    gene_to_phase.update({g: "G2M" for g in g2m_pool})

    print("\nFiltered gene lists (saved to gene_lists/):")
    for thr in [0.2, 0.3, 0.4]:
        passing = df_wide[df_wide["r_mean"] >= thr]["gene"].tolist()
        filtered = {
            "s.genes":   [g for g in passing if gene_to_phase.get(g) == "S"],
            "g2m.genes": [g for g in passing if gene_to_phase.get(g) == "G2M"],
        }
        thr_tag = f"r{int(thr * 10):02d}"
        out_json = gene_list_dir / f"cc_genes_{thr_tag}.json"
        with open(out_json, "w") as f:
            json.dump(filtered, f, indent=4)
        n_s, n_g2m = len(filtered["s.genes"]), len(filtered["g2m.genes"])
        print(f"  r≥{thr}: S={n_s}, G2M={n_g2m}, total={n_s+n_g2m}  → {out_json.name}")

    # ── Print summary ──────────────────────────────────────────────────────────
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + "═" * 65)
    print("LOO Spearman r on SciPlex3  (higher = stronger independent signal)")
    print("═" * 65)
    print(df_wide.to_string(index=False))

    for thr in [0.2, 0.3, 0.4]:
        low = df_wide[df_wide["r_mean"] < thr]
        print(f"\nBelow r={thr} ({len(low)} genes): {low['gene'].tolist()}")

    print(f"\nNext: python validation/02_validate_PI_sciplex.py")


if __name__ == "__main__":
    main()
