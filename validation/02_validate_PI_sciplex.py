"""
02_validate_PI_sciplex.py
──────────────────────────
Validate our PI against the monocle3-based reference PI (SciPlex paper)
across three LOO-filtered gene lists (r≥0.2, r≥0.3, r≥0.4).

Also validates against MKI67 expression as a canonical proliferation marker
to test the hypothesis that LOO-filtered PI captures proliferation signal
more cleanly than the full 100-gene pool.

PURPOSE
───────
1. Shows our simple PI formula reproduces the published monocle3
   proliferation_index on the public SciPlex3 dataset.
2. Tests whether LOO filtering improves alignment with MKI67 — a canonical
   G2/M marker — even when overall correlation with monocle3 PI slightly
   decreases due to fewer genes.

HYPOTHESIS
──────────
LOO-filtered PI (r≥0.2) should correlate BETTER with MKI67 than the
full 100-gene pool PI, because LOO removes genes with weak proliferation
signal, reducing noise in the final score.

PREREQUISITE
────────────
Run 01_gene_LOO_correlation.py first — it generates:
  gene_lists/cc_genes_r02.json
  gene_lists/cc_genes_r03.json
  gene_lists/cc_genes_r04.json

WHAT IS COMPARED
────────────────
For each threshold (r≥0.2, r≥0.3, r≥0.4) and also the full 100-gene pool:
  PI         = log1p( (Σ_S + Σ_G2M raw counts) / nCount_RNA )
  reference  = monocle3 proliferation_index  (published, in adata.obs)
  MKI67      = log1p( MKI67_raw_count / nCount_RNA )  (canonical marker)

Spearman r and Pearson r are reported both overall and per cell type.

USAGE
─────
  conda run -n scvi python 02_validate_PI_sciplex.py

  # Custom paths
  conda run -n scvi python 02_validate_PI_sciplex.py \\
      --sciplex /path/to/sciplex3_atomicPI2.symbolVar.h5ad \\
      --gene-list-dir gene_lists/ \\
      --out results/

MEMORY ESTIMATE
───────────────
  581,777 cells × 100 CC genes × 4 B ≈ 233 MB
  (same matrix reused across all threshold comparisons)
  Recommended: ≥ 2 GB RAM
"""

import argparse
import json
import sys
from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from proliferation_index.pi_functions import compute_pi, load_cc_counts, load_gene_dict

# ─── Default paths ────────────────────────────────────────────────────────────
_BASE = Path("/mnt/lab-store/projects/ATOMIC/analysis/dong")
DEFAULT_SCIPLEX       = _BASE / "object/SciPlexGxE/sciplex3_atomicPI2.symbolVar.h5ad"
DEFAULT_GENE_LIST_DIR = ROOT / "gene_lists"
DEFAULT_OUT           = ROOT / "results"
MONOCLE3_PI_KEY       = "proliferation_index"
CELLTYPE_KEY          = "cell_type"
MKI67_GENE            = "MKI67"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sciplex",        default=str(DEFAULT_SCIPLEX),       help="Path to SciPlex3 symbolVar h5ad")
    p.add_argument("--gene-list-dir",  default=str(DEFAULT_GENE_LIST_DIR), help="Directory containing gene list JSONs")
    p.add_argument("--out",            default=str(DEFAULT_OUT),           help="Output directory")
    p.add_argument("--no-plot",        action="store_true",                 help="Skip matplotlib figures")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_pi_for_genelist(
    counts: np.ndarray,
    libsize: np.ndarray,
    present_genes: list[str],
    gene_dict: dict,
    exclude: Optional[list] = None,
) -> np.ndarray:
    """Compute PI given a counts matrix (pre-loaded for the 100-gene pool).
    exclude: list of gene names to omit from the calculation (e.g. [MKI67_GENE]).
    """
    excl = set(exclude) if exclude else set()
    gene_to_col = {g: i for i, g in enumerate(present_genes)}
    s_idx   = [gene_to_col[g] for g in gene_dict["s.genes"]   if g in gene_to_col and g not in excl]
    g2m_idx = [gene_to_col[g] for g in gene_dict["g2m.genes"] if g in gene_to_col and g not in excl]
    return compute_pi(counts, libsize, s_idx, g2m_idx)


def correlation_stats(a: np.ndarray, b: np.ndarray) -> dict:
    r_sp, p_sp = spearmanr(a, b)
    r_pe, p_pe = pearsonr(a, b)
    return {"spearman_r": r_sp, "spearman_p": p_sp,
            "pearson_r":  r_pe, "pearson_p":  p_pe,
            "n": len(a)}


def load_mki67_expr(sciplex_path: str, libsize: np.ndarray) -> Optional[np.ndarray]:
    """
    Load MKI67 raw counts and return log1p(count / nCount_RNA).
    Returns None if MKI67 is not found in the dataset.
    """
    import anndata as ad
    import scipy.sparse as sp

    adata = ad.read_h5ad(sciplex_path, backed="r")
    if MKI67_GENE not in adata.var_names:
        print(f"[WARN] {MKI67_GENE} not found in dataset var_names — skipping MKI67 validation.")
        adata.file.close()
        return None

    adata_sub = adata[:, [MKI67_GENE]]
    raw = adata_sub.layers["counts_RNA"]
    mki67_raw = raw.toarray().ravel() if sp.issparse(raw) else np.asarray(raw).ravel()
    adata.file.close()

    safe_libsize = np.where(libsize > 0, libsize, 1.0)
    return np.log1p(mki67_raw / safe_libsize)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_list_dir = Path(args.gene_list_dir)

    if not Path(args.sciplex).exists():
        print(f"ERROR: SciPlex3 file not found: {args.sciplex}")
        sys.exit(1)

    # ── Discover gene list files ──────────────────────────────────────────────
    configs = []  # list of (label, json_path)

    pool_json = gene_list_dir / "cc_genes_pool.json"
    if pool_json.exists():
        configs.append(("pool_100", pool_json))

    for thr_tag, label in [("r02", "r≥0.2"), ("r03", "r≥0.3")]:
        p = gene_list_dir / f"cc_genes_{thr_tag}.json"
        if p.exists():
            configs.append((label, p))
        else:
            print(f"[WARN] {p.name} not found — run 01_gene_LOO_correlation.py first.")

    if not configs:
        print("No gene list files found. Exiting.")
        sys.exit(1)

    # ── Load the union of all CC genes ────────────────────────────────────────
    all_genes_union: list[str] = []
    seen = set()
    for _, jp in configs:
        gd = load_gene_dict(jp)
        for g in gd["s.genes"] + gd["g2m.genes"]:
            if g not in seen:
                all_genes_union.append(g)
                seen.add(g)

    print(f"Loading SciPlex3 counts for {len(all_genes_union)} unique CC genes...")
    counts, libsize, present_genes, missing = load_cc_counts(args.sciplex, all_genes_union)
    print(f"  Cells: {counts.shape[0]:,}  |  genes found: {len(present_genes)}/{len(all_genes_union)}")
    if missing:
        print(f"  Missing from dataset: {missing}")
    print(f"  Counts matrix: {counts.nbytes / 1e6:.0f} MB")

    # ── Load reference PI + cell type + MKI67 ────────────────────────────────
    import anndata as ad
    adata = ad.read_h5ad(args.sciplex, backed="r")
    ref_pi    = adata.obs[MONOCLE3_PI_KEY].values.astype(float)
    celltypes = adata.obs[CELLTYPE_KEY].values if CELLTYPE_KEY in adata.obs.columns else None
    adata.file.close()

    print(f"  Reference PI (monocle3): min={ref_pi.min():.4f}, "
          f"mean={ref_pi.mean():.4f}, max={ref_pi.max():.4f}")

    mki67_expr = load_mki67_expr(args.sciplex, libsize)
    if mki67_expr is not None:
        nonzero_pct = (mki67_expr > 0).mean() * 100
        print(f"  MKI67 (log-normalized): "
              f"mean={mki67_expr.mean():.4f}, nonzero={nonzero_pct:.1f}%")

    # ── Compute PI + correlations for each gene list ──────────────────────────
    print(f"\n{'═'*85}")
    hdr = (f"{'Gene list':<12} {'n_genes':>7} {'n_S':>5} {'n_G2M':>6}  "
           f"{'Spearman r':>10}  {'Pearson r':>9}")
    if mki67_expr is not None:
        hdr += f"  {'r_MKI67':>9}"
    print(hdr)
    print("═"*85)

    summary_records = []
    per_cell_scores = {"monocle3_pi": ref_pi}
    if mki67_expr is not None:
        per_cell_scores["mki67_expr"] = mki67_expr
    if celltypes is not None:
        per_cell_scores["cell_type"] = celltypes

    for label, json_path in configs:
        gd = load_gene_dict(json_path)
        n_s   = len(gd["s.genes"])
        n_g2m = len(gd["g2m.genes"])
        n_tot = n_s + n_g2m

        pi_vals   = compute_pi_for_genelist(counts, libsize, present_genes, gd)
        stats_all = correlation_stats(pi_vals, ref_pi)

        col_key = f"pi_{label.replace('≥','').replace('.','').replace(' ','')}"
        per_cell_scores[col_key] = pi_vals

        # MKI67 correlation: compute PI without MKI67 to avoid circularity
        # (MKI67 is in the gene pool, so including it inflates r artificially)
        mki67_r_sp = float("nan")
        if mki67_expr is not None:
            pi_no_mki67 = compute_pi_for_genelist(
                counts, libsize, present_genes, gd, exclude=[MKI67_GENE]
            )
            mki67_stats = correlation_stats(pi_no_mki67, mki67_expr)
            mki67_r_sp  = mki67_stats["spearman_r"]
            col_key_nomki = f"pi_no_mki67_{label.replace('≥','').replace('.','').replace(' ','')}"
            per_cell_scores[col_key_nomki] = pi_no_mki67

        line = (f"  {label:<10} {n_tot:>7}  {n_s:>5}  {n_g2m:>6}  "
                f"{stats_all['spearman_r']:>10.4f}  {stats_all['pearson_r']:>9.4f}")
        if mki67_expr is not None:
            line += f"  {mki67_r_sp:>9.4f}"
        print(line)

        row = {"gene_list": label, "n_total": n_tot, "n_S": n_s, "n_G2M": n_g2m}
        row.update(stats_all)
        if mki67_expr is not None:
            row["mki67_spearman_r"] = mki67_r_sp

        # Per cell-type breakdown
        if celltypes is not None:
            for ct in sorted(set(celltypes)):
                mask = celltypes == ct
                if mask.sum() < 50:
                    continue
                ct_stats = correlation_stats(pi_vals[mask], ref_pi[mask])
                row[f"r_sp_{ct}"] = ct_stats["spearman_r"]
                row[f"r_pe_{ct}"] = ct_stats["pearson_r"]

        summary_records.append(row)

    print("═"*85)

    # ── MKI67 hypothesis summary ──────────────────────────────────────────────
    if mki67_expr is not None:
        print(f"\nMKI67 hypothesis test (higher r_MKI67 → purer proliferation signal):")
        pool_mki67 = next(r["mki67_spearman_r"] for r in summary_records if r["gene_list"] == "pool_100")
        for rec in summary_records:
            if rec["gene_list"] == "pool_100":
                continue
            delta = rec["mki67_spearman_r"] - pool_mki67
            sign  = "▲" if delta > 0 else "▼"
            print(f"  {rec['gene_list']}: r_MKI67={rec['mki67_spearman_r']:.4f}  "
                  f"({sign}{abs(delta):.4f} vs pool_100)")

    # ── Per cell-type detail ──────────────────────────────────────────────────
    if celltypes is not None:
        unique_cts = sorted(set(celltypes))
        print(f"\nPer cell-type Spearman r (vs monocle3 PI):")
        header = f"  {'Gene list':<12}" + "".join(f"  {ct[:10]:>10}" for ct in unique_cts)
        print(header)
        print("  " + "─" * (len(header) - 2))
        for rec in summary_records:
            row_str = f"  {rec['gene_list']:<12}"
            for ct in unique_cts:
                val = rec.get(f"r_sp_{ct}", float("nan"))
                row_str += f"  {val:>10.4f}"
            print(row_str)

    # ── Save ─────────────────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_records)
    out_summary = out_dir / "validation_summary.csv"
    df_summary.to_csv(out_summary, index=False)
    print(f"\nSummary saved → {out_summary}")

    df_scores = pd.DataFrame(per_cell_scores)
    out_scores = out_dir / "validation_scores.csv"
    df_scores.to_csv(out_scores, index=False)
    print(f"Per-cell scores → {out_scores}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_configs = len(configs)
            rng = np.random.default_rng(42)
            n_plot = min(15000, len(ref_pi))
            idx_plot = rng.choice(len(ref_pi), n_plot, replace=False)

            # ── Figure 1: PI vs monocle3 reference scatter ─────────────────
            fig1, axes1 = plt.subplots(1, n_configs, figsize=(4 * n_configs, 4), squeeze=False)

            for ax, (label, _), rec in zip(axes1[0], configs, summary_records):
                col_key = f"pi_{label.replace('≥','').replace('.','').replace(' ','')}"
                pi_vals = per_cell_scores[col_key]

                if celltypes is not None:
                    cmap = plt.get_cmap("tab10")
                    cts = sorted(set(celltypes))
                    for i, ct in enumerate(cts):
                        m = celltypes[idx_plot] == ct
                        ax.scatter(ref_pi[idx_plot][m], pi_vals[idx_plot][m],
                                   s=1, alpha=0.3, color=cmap(i % 10),
                                   label=ct, rasterized=True)
                    if ax is axes1[0][0]:
                        ax.legend(markerscale=5, fontsize=6, loc="upper left",
                                  title=CELLTYPE_KEY, title_fontsize=6)
                else:
                    ax.scatter(ref_pi[idx_plot], pi_vals[idx_plot],
                               s=1, alpha=0.3, color="steelblue", rasterized=True)

                n_s, n_g2m = rec["n_S"], rec["n_G2M"]
                r_sp = rec["spearman_r"]
                r_pe = rec["pearson_r"]
                ax.set_title(f"{label}  (S={n_s}, G2M={n_g2m})\n"
                             f"Spearman r={r_sp:.3f}  Pearson r={r_pe:.3f}", fontsize=9)
                ax.set_xlabel("monocle3 PI (reference)", fontsize=8)
                ax.set_ylabel("PI", fontsize=8)

            fig1.suptitle(f"SciPlex3 validation — PI vs monocle3 PI\n"
                          f"(n={len(ref_pi):,} cells, plotted: {n_plot:,})",
                          fontsize=10, y=1.02)
            fig1.tight_layout()
            out_fig1 = out_dir / "validation_scatter.png"
            fig1.savefig(out_fig1, dpi=150, bbox_inches="tight")
            plt.close(fig1)
            print(f"Scatter (vs monocle3) → {out_fig1}")

            # ── Figure 2: PI vs MKI67 scatter ─────────────────────────────
            if mki67_expr is not None:
                fig2, axes2 = plt.subplots(1, n_configs, figsize=(4 * n_configs, 4), squeeze=False)

                for ax, (label, _), rec in zip(axes2[0], configs, summary_records):
                    col_key = f"pi_no_mki67_{label.replace('≥','').replace('.','').replace(' ','')}"
                    pi_vals = per_cell_scores[col_key]

                    if celltypes is not None:
                        cmap = plt.get_cmap("tab10")
                        cts = sorted(set(celltypes))
                        for i, ct in enumerate(cts):
                            m = celltypes[idx_plot] == ct
                            ax.scatter(mki67_expr[idx_plot][m], pi_vals[idx_plot][m],
                                       s=1, alpha=0.3, color=cmap(i % 10),
                                       label=ct, rasterized=True)
                        if ax is axes2[0][0]:
                            ax.legend(markerscale=5, fontsize=6, loc="upper left",
                                      title=CELLTYPE_KEY, title_fontsize=6)
                    else:
                        ax.scatter(mki67_expr[idx_plot], pi_vals[idx_plot],
                                   s=1, alpha=0.3, color="darkorange", rasterized=True)

                    n_s, n_g2m = rec["n_S"], rec["n_G2M"]
                    r_mki67 = rec.get("mki67_spearman_r", float("nan"))
                    ax.set_title(f"{label}  (S={n_s}, G2M={n_g2m})\n"
                                 f"Spearman r={r_mki67:.3f} vs MKI67", fontsize=9)
                    ax.set_xlabel("MKI67 expression (log-norm)", fontsize=8)
                    ax.set_ylabel("PI", fontsize=8)

                fig2.suptitle(f"SciPlex3 MKI67 validation — PI (MKI67-excluded) vs MKI67\n"
                              f"(n={len(ref_pi):,} cells, plotted: {n_plot:,})",
                              fontsize=10, y=1.02)
                fig2.tight_layout()
                out_fig2 = out_dir / "validation_mki67_scatter.png"
                fig2.savefig(out_fig2, dpi=150, bbox_inches="tight")
                plt.close(fig2)
                print(f"Scatter (vs MKI67)  → {out_fig2}")

                # ── Figure 3: MKI67 r bar chart ───────────────────────────
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                labels_list   = [r["gene_list"] for r in summary_records]
                mki67_r_list  = [r.get("mki67_spearman_r", float("nan")) for r in summary_records]
                monocle_r_list = [r["spearman_r"] for r in summary_records]

                x = np.arange(len(labels_list))
                width = 0.35
                bars1 = ax3.bar(x - width/2, monocle_r_list, width, label="vs monocle3 PI", color="steelblue", alpha=0.8)
                bars2 = ax3.bar(x + width/2, mki67_r_list,   width, label="vs MKI67",       color="darkorange", alpha=0.8)

                ax3.set_xlabel("Gene list", fontsize=10)
                ax3.set_ylabel("Spearman r", fontsize=10)
                ax3.set_title("PI (MKI67-excluded) correlation: vs monocle3 PI vs MKI67\n"
                              "(MKI67 excluded from PI to avoid circularity)", fontsize=10)
                ax3.set_xticks(x)
                ax3.set_xticklabels(labels_list)
                ax3.legend(fontsize=9)
                ax3.set_ylim(0, 1)

                for bar in bars1:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
                for bar in bars2:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

                fig3.tight_layout()
                out_fig3 = out_dir / "validation_mki67_bar.png"
                fig3.savefig(out_fig3, dpi=150, bbox_inches="tight")
                plt.close(fig3)
                print(f"Bar chart (MKI67)   → {out_fig3}")

        except ImportError:
            print("[WARN] matplotlib not available — skipping plots.")


if __name__ == "__main__":
    main()
