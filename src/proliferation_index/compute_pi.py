"""
compute_pi.py
─────────────
Compute a data-adaptive Proliferation Index (PI) for any scRNA-seq h5ad file.

ALGORITHM
─────────
1. Load the 100-gene candidate pool (cc_genes_pool.json).
2. Run Leave-One-Out (LOO) Spearman correlation on your data:
   for each gene g, compute PI without g, then correlate expr(g) vs LOO-PI.
3. Show how many genes pass each r threshold (0 / 0.2 / 0.3 / 0.4 / 0.5).
4. Ask you to select a threshold  (or pass --r-threshold to skip the prompt).
   Use --r-threshold 0 to skip LOO entirely and use the full 100-gene pool
   (recommended for post-mitotic / low-cycling tissues like brain).
5. Compute PI with the passing genes.
6. Save PI as 'proliferation_index' in adata.obs and write h5ad.

PI FORMULA
──────────
PI = log1p( (Σ_S_raw_counts + Σ_G2M_raw_counts) / nCount_RNA )

REQUIREMENTS
────────────
  h5ad must have:
    • var_names   : gene symbols  (e.g. 'MCM5', not Ensembl IDs)
    • layers      : 'counts_RNA'  (raw UMI counts)
    • obs column  : 'nCount_RNA'  (total UMI per cell)

USAGE
─────
  # Interactive (recommended for first-time users)
  p_index --input mydata.h5ad

  # Non-interactive (for pipelines)
  p_index --input mydata.h5ad --r-threshold 0.3

  # Custom output path
  p_index --input mydata.h5ad --output mydata_with_pi.h5ad

MEMORY ESTIMATE
───────────────
  LOO phase   : n_cells × 100 genes × 4 B  (e.g. 336k cells → ~270 MB)
  Write phase : full h5ad load required — plan for ≥ 2× the file size in RAM
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .pi_functions import (
    HAS_PSUTIL, rss_mb,
    compute_pi, detect_counts_source, load_cc_counts, load_gene_dict,
)

# ─── Default paths ────────────────────────────────────────────────────────────
DEFAULT_GENE_POOL = Path(__file__).parent / "gene_lists/cc_genes_pool.json"
VALID_THRESHOLDS  = [0.0, 0.2, 0.3, 0.4, 0.5]
OBS_KEY           = "proliferation_index"

# Preset → (counts_layer, libsize_obs_key)
PRESETS = {
    "seurat": ("counts_RNA", "nCount_RNA"),
    "scanpy": ("counts",     "total_counts"),
}


# ─── Run logger ───────────────────────────────────────────────────────────────

class RunLogger:
    """Prints to stdout and accumulates lines for a final log file."""

    def __init__(self, log_path: Path):
        self._path = log_path
        self._lines: list[str] = []
        self._start = datetime.datetime.now()
        self._log(f"{'='*60}")
        self._log(f"compute_pi.py  started: {self._start.isoformat(timespec='seconds')}")
        if not HAS_PSUTIL:
            self._log("  (install psutil for RAM tracking: pip install psutil)")

    def _log(self, msg: str):
        print(msg)
        self._lines.append(msg)

    def log(self, msg: str, mem: bool = False):
        if mem:
            ram = f"  [RAM: {rss_mb():.0f} MB]" if HAS_PSUTIL else ""
            self._log(f"{msg}{ram}")
        else:
            self._log(msg)

    def save(self):
        elapsed = datetime.datetime.now() - self._start
        self._log(f"Elapsed: {elapsed}")
        self._log(f"{'='*60}")
        self._path.write_text("\n".join(self._lines) + "\n")
        print(f"Log saved → {self._path}")

    def save_error(self, tb: str):
        """Save error log with full traceback. Called on exception."""
        import traceback as _tb
        err_path = self._path.with_name(self._path.stem.replace("_run", "_error") + ".log")
        elapsed = datetime.datetime.now() - self._start
        lines = self._lines + [
            "",
            f"{'!'*60}",
            f"FAILED after {elapsed}",
            tb,
            f"{'!'*60}",
        ]
        err_path.write_text("\n".join(lines) + "\n")
        print(f"\nError log saved → {err_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",       required=True,               help="Input h5ad file (gene symbol var_names)")
    p.add_argument("--output",      default=None,                help="Output h5ad path (default: <input>_pi.h5ad)")
    p.add_argument("--gene-pool",   default=str(DEFAULT_GENE_POOL), help="Path to cc_genes_pool.json")
    p.add_argument("--r-threshold", type=float, default=None,    help="LOO r threshold (skip interactive prompt)")
    p.add_argument("--preset",      default="auto",              choices=["auto", "seurat", "scanpy"],
                   help="Input format preset: auto (default) | seurat (counts_RNA/nCount_RNA) | scanpy (counts/total_counts)")
    p.add_argument("--counts-layer",default=None,                help="Override layer name for raw counts (overrides --preset)")
    p.add_argument("--libsize-key", default=None,                help="Override obs column for total counts (overrides --preset)")
    return p.parse_args()


# ─── LOO correlation (same core as validation/01) ─────────────────────────────

def run_loo(counts, libsize, present_genes, s_pool, g2m_pool):
    """Return DataFrame: gene, phase, spearman_r."""
    gene_to_col = {g: i for i, g in enumerate(present_genes)}
    s_present   = [g for g in s_pool   if g in gene_to_col]
    g2m_present = [g for g in g2m_pool if g in gene_to_col]

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
            r, _      = spearmanr(gene_expr, pi_loo)
            records.append({"gene": gene, "phase": phase_label, "spearman_r": float(r)})

    return pd.DataFrame(records)


# ─── Interactive threshold selection ──────────────────────────────────────────

def print_threshold_table(df_loo, s_pool, g2m_pool):
    """Print gene-count summary for each threshold and return the table."""
    gene_to_phase = {g: "S" for g in s_pool}
    gene_to_phase.update({g: "G2M" for g in g2m_pool})

    all_genes = df_loo["gene"].tolist()
    n_s_all   = sum(1 for g in all_genes if gene_to_phase.get(g) == "S")
    n_g2m_all = sum(1 for g in all_genes if gene_to_phase.get(g) == "G2M")

    print(f"\n{'─'*60}")
    print(f"  {'Threshold':>12}  {'Total':>6}  {'S genes':>8}  {'G2M genes':>10}")
    print(f"{'─'*60}")
    print(f"  r ≥ 0 (pool)  {len(all_genes):>6}  {n_s_all:>8}  {n_g2m_all:>10}  ← no filter")
    print(f"{'─'*60}")

    rows = [{"threshold": 0.0, "n_total": len(all_genes), "n_S": n_s_all, "n_G2M": n_g2m_all}]
    for thr in [t for t in VALID_THRESHOLDS if t > 0]:
        passing = df_loo[df_loo["spearman_r"] >= thr]["gene"].tolist()
        n_s   = sum(1 for g in passing if gene_to_phase.get(g) == "S")
        n_g2m = sum(1 for g in passing if gene_to_phase.get(g) == "G2M")
        print(f"  r ≥ {thr:.1f}        {len(passing):>6}  {n_s:>8}  {n_g2m:>10}")
        rows.append({"threshold": thr, "n_total": len(passing), "n_S": n_s, "n_G2M": n_g2m})
    print(f"{'─'*60}")
    return pd.DataFrame(rows)


def ask_threshold():
    valid = [str(t) for t in VALID_THRESHOLDS]
    while True:
        raw = input(f"\nSelect r threshold [{'/'.join(valid)}]: ").strip()
        try:
            val = float(raw)
            if val in VALID_THRESHOLDS:
                return val
        except ValueError:
            pass
        print(f"  Please enter one of: {', '.join(valid)}")


# ─── Run logic (called by main, wrapped in try/except) ───────────────────────

def _run(args, input_path: Path, output_path: Path, logger: RunLogger):
    if not input_path.exists():
        logger.log(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    if not Path(args.gene_pool).exists():
        logger.log(f"ERROR: gene pool not found: {args.gene_pool}")
        sys.exit(1)

    # ── Resolve counts layer / libsize key ────────────────────────────────────
    if args.counts_layer is not None or args.libsize_key is not None:
        counts_layer = args.counts_layer
        libsize_key  = args.libsize_key or "nCount_RNA"
        logger.log(f"Counts source: layer='{counts_layer}', libsize='{libsize_key}'  (manual override)")
    elif args.preset == "auto":
        logger.log("\nAuto-detecting counts source...")
        counts_layer, libsize_key = detect_counts_source(input_path)
        layer_label = counts_layer if counts_layer is not None else "adata.X"
        logger.log(f"  Detected: layer='{layer_label}', libsize='{libsize_key}'")
    else:
        counts_layer, libsize_key = PRESETS[args.preset]
        logger.log(f"Counts source: preset='{args.preset}'  →  layer='{counts_layer}', libsize='{libsize_key}'")

    # ── Load gene pool ─────────────────────────────────────────────────────────
    gene_dict = load_gene_dict(args.gene_pool)
    s_pool    = gene_dict["s.genes"]
    g2m_pool  = gene_dict["g2m.genes"]
    all_pool  = s_pool + g2m_pool
    logger.log(f"\nGene pool: {len(s_pool)} S + {len(g2m_pool)} G2M = {len(all_pool)} total")

    # ── Phase 1: Load CC gene counts ──────────────────────────────────────────
    logger.log("\nPhase 1 – Loading CC gene counts (backed, low memory)...", mem=True)
    counts, libsize, present, missing = load_cc_counts(
        input_path, all_pool,
        counts_layer=counts_layer,
        libsize_obs_key=libsize_key,
    )
    logger.log(f"  Cells: {counts.shape[0]:,}  |  CC genes found: {len(present)}/{len(all_pool)}")
    if missing:
        logger.log(f"  Genes absent from dataset: {missing}")
    logger.log(f"  Counts matrix: {counts.nbytes / 1e6:.0f} MB", mem=True)

    # ── r-threshold 0: skip LOO, use full pool ────────────────────────────────
    if args.r_threshold == 0.0:
        logger.log("\nSkipping LOO (--r-threshold 0): using full gene pool.")
        r_thr         = 0.0
        passing_genes = present
    else:
        # ── LOO correlation ───────────────────────────────────────────────────
        logger.log(f"\nRunning LOO Spearman correlation ({len(present)} genes × {counts.shape[0]:,} cells)...")
        df_loo = run_loo(counts, libsize, present, s_pool, g2m_pool)
        logger.log("  LOO done.", mem=True)

        # ── Threshold summary + selection ─────────────────────────────────────
        tbl = print_threshold_table(df_loo, s_pool, g2m_pool)

        # Warn if all non-zero thresholds yield 0 genes (non-cycling tissue)
        if tbl[tbl["threshold"] > 0]["n_total"].max() == 0:
            logger.log(
                "\n[HINT] All LOO thresholds yield 0 genes — this dataset likely has\n"
                "       low cell-cycle activity (e.g. post-mitotic tissue).\n"
                "       Use --r-threshold 0 to compute PI with the full 100-gene pool."
            )

        if args.r_threshold is not None:
            if args.r_threshold not in VALID_THRESHOLDS:
                logger.log(f"ERROR: --r-threshold must be one of {VALID_THRESHOLDS}")
                sys.exit(1)
            r_thr = args.r_threshold
            logger.log(f"Using r threshold: {r_thr}  (--r-threshold)")
        else:
            r_thr = ask_threshold()
            logger.log(f"Using r threshold: {r_thr}  (interactive)")

        passing_genes = df_loo[df_loo["spearman_r"] >= r_thr]["gene"].tolist()

    gene_to_col = {g: i for i, g in enumerate(present)}
    s_idx   = [gene_to_col[g] for g in passing_genes if g in gene_to_col and g in set(s_pool)]
    g2m_idx = [gene_to_col[g] for g in passing_genes if g in gene_to_col and g in set(g2m_pool)]

    logger.log(f"\nPhase 2 – Computing PI ({len(passing_genes)} genes: S={len(s_idx)}, G2M={len(g2m_idx)})...", mem=True)
    pi_values = compute_pi(counts, libsize, s_idx, g2m_idx)
    del counts
    logger.log(f"  PI stats: mean={pi_values.mean():.4f}, std={pi_values.std():.4f}, "
               f"min={pi_values.min():.4f}, max={pi_values.max():.4f}", mem=True)

    # ── Phase 3: Write PI back to h5ad ────────────────────────────────────────
    logger.log(f"\nPhase 3 – Loading full h5ad for write-back (peak RAM phase)...", mem=True)
    import anndata as ad
    adata = ad.read_h5ad(input_path)
    logger.log(f"  Full h5ad loaded.", mem=True)

    adata.obs[OBS_KEY] = pi_values
    adata.uns["proliferation_index_params"] = {
        "r_threshold":  r_thr,
        "counts_layer": counts_layer,
        "libsize_key":  libsize_key,
        "n_genes_S":    len(s_idx),
        "n_genes_G2M":  len(g2m_idx),
        "genes_S":      [g for g in passing_genes if g in set(s_pool)],
        "genes_G2M":    [g for g in passing_genes if g in set(g2m_pool)],
    }
    adata.write_h5ad(output_path)
    logger.log(f"  Writing done.", mem=True)

    logger.log(f"\nSaved → {output_path}")
    logger.log(f"  obs key: '{OBS_KEY}'")
    logger.log(f"  uns key: 'proliferation_index_params'  (records genes + params used)")

    logger.save()


def main():
    import traceback
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_pi.h5ad"
    )
    log_path = output_path.with_name(output_path.stem + "_run.log")
    logger = RunLogger(log_path)

    try:
        _run(args, input_path, output_path, logger)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        logger.save_error(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
