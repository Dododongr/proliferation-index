"""
pi_functions.py
Core functions for Proliferation Index (PI) calculation.

PI formula (SciPlex / monocle3 style):
    PI = log1p( (sum_S_raw_counts + sum_G2M_raw_counts) / total_library_size )

This matches the monocle3-based proliferation_index published in Srivatsan et al. 2020
(SciPlex) and is robust across sparse single-cell RNA-seq datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp

# ─── Memory helper ───────────────────────────────────────────────────────────

try:
    import psutil as _psutil
    import os as _os
    def rss_mb() -> float:
        """Current process RSS in MB."""
        return _psutil.Process(_os.getpid()).memory_info().rss / 1024 ** 2
    HAS_PSUTIL = True
except ImportError:
    def rss_mb() -> float:
        return float("nan")
    HAS_PSUTIL = False


# ─── Gene list helpers ───────────────────────────────────────────────────────

def load_gene_dict(path: str | Path) -> dict[str, list[str]]:
    """Load a JSON gene list with keys 's.genes' and 'g2m.genes'."""
    with open(path) as f:
        return json.load(f)


# ─── Core PI calculation ─────────────────────────────────────────────────────

def compute_pi(
    counts: np.ndarray,
    libsize: np.ndarray,
    s_idx: Sequence[int],
    g2m_idx: Sequence[int],
) -> np.ndarray:
    """
    Compute Proliferation Index per cell.

    Parameters
    ----------
    counts : np.ndarray, shape (n_cells, n_genes)
        Dense raw-count matrix (subset to CC genes).
    libsize : np.ndarray, shape (n_cells,)
        Total raw counts per cell (adata.obs['nCount_RNA']).
    s_idx : sequence of int
        Column indices of S-phase genes in `counts`.
    g2m_idx : sequence of int
        Column indices of G2M-phase genes in `counts`.

    Returns
    -------
    np.ndarray, shape (n_cells,)  –  PI values in [0, ∞).
    """
    s_idx = list(s_idx)
    g2m_idx = list(g2m_idx)

    s_sum = counts[:, s_idx].sum(axis=1).ravel() if s_idx else np.zeros(counts.shape[0])
    g2m_sum = counts[:, g2m_idx].sum(axis=1).ravel() if g2m_idx else np.zeros(counts.shape[0])

    safe_libsize = np.where(libsize > 0, libsize, 1.0)
    return np.log1p((s_sum + g2m_sum) / safe_libsize)


# ─── Dataset loading helper ──────────────────────────────────────────────────

# Ordered candidate lists for auto-detection
_LAYER_CANDIDATES   = ["counts_RNA", "counts"]   # None → fall back to adata.X
_LIBSIZE_CANDIDATES = ["nCount_RNA", "total_counts"]


def detect_counts_source(h5ad_path: str | Path) -> tuple[str | None, str]:
    """
    Auto-detect counts layer and libsize obs key from an h5ad file.

    Tries (in priority order):
      layer    : 'counts_RNA' → 'counts' → None (= adata.X)
      libsize  : 'nCount_RNA' → 'total_counts'

    Returns
    -------
    (counts_layer, libsize_obs_key)
        counts_layer is None when adata.X should be used.

    Raises
    ------
    ValueError if no suitable libsize column is found.
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")

    # Layer
    layer = next((l for l in _LAYER_CANDIDATES if l in adata.layers), None)

    # Libsize
    obs_cols = set(adata.obs.columns)
    libsize_key = next((k for k in _LIBSIZE_CANDIDATES if k in obs_cols), None)
    if libsize_key is None:
        adata.file.close()
        raise ValueError(
            f"Cannot auto-detect libsize column. "
            f"Tried: {_LIBSIZE_CANDIDATES}. Available obs: {sorted(obs_cols)}"
        )

    adata.file.close()
    return layer, libsize_key


def load_cc_counts(
    h5ad_path: str | Path,
    gene_list: list[str],
    counts_layer: str | None = "counts_RNA",
    libsize_obs_key: str = "nCount_RNA",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load raw-count expression for CC-gene subset from an h5ad file.

    Uses backed='r' mode and CSC-efficient column subsetting so that only
    ~100 gene columns are loaded into memory (not the full matrix).

    Parameters
    ----------
    h5ad_path :
        Path to h5ad file. var_names must be gene symbols.
    gene_list :
        Ordered list of all candidate CC genes to extract.
    counts_layer :
        Name of the raw-counts layer. Pass None to use adata.X.
    libsize_obs_key :
        obs column with total raw count per cell.

    Returns
    -------
    counts_dense : np.ndarray, shape (n_cells, n_present_genes)
    libsize      : np.ndarray, shape (n_cells,)
    present_genes: list[str]  – genes from gene_list found in the dataset
    missing_genes: list[str]  – genes from gene_list absent in the dataset
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    var_set = set(adata.var_names)

    present_genes = [g for g in gene_list if g in var_set]
    missing_genes = [g for g in gene_list if g not in var_set]

    # Column-subset view (CSC-efficient: reads only selected columns from disk)
    adata_sub = adata[:, present_genes]
    if counts_layer is None:
        raw = adata_sub.X
    else:
        raw = adata_sub.layers[counts_layer]
    counts_dense = raw.toarray() if sp.issparse(raw) else np.asarray(raw)

    libsize = adata.obs[libsize_obs_key].values.astype(float)

    adata.file.close()
    return counts_dense, libsize, present_genes, missing_genes
