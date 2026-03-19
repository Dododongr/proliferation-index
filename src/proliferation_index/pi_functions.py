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


def _h5ad_keys(h5ad_path: str | Path) -> tuple[list[str], list[str], list[str]]:
    """
    Read layer names, obs column names, and var_names from an h5ad file
    using h5py directly — bypasses anndata version compatibility issues
    (e.g. uns/log1p/base=null encoding errors).

    Returns
    -------
    (layer_keys, obs_keys, var_names)
    """
    import h5py
    import numpy as np

    with h5py.File(h5ad_path, "r") as f:
        layer_keys = list(f.get("layers", {}).keys())

        # obs columns — h5py groups have keys(); datasets are columns
        obs_keys = [k for k in f["obs"].keys() if k != "_index"]

        # var_names — stored in var/_index or var/index
        var_grp = f["var"]
        if "_index" in var_grp:
            var_names = [v.decode() if isinstance(v, bytes) else v
                         for v in var_grp["_index"][:]]
        elif "index" in var_grp:
            var_names = [v.decode() if isinstance(v, bytes) else v
                         for v in var_grp["index"][:]]
        else:
            # fallback: first string dataset in var
            idx_key = next(k for k in var_grp.keys()
                           if var_grp[k].dtype.kind in ("S", "U", "O"))
            var_names = [v.decode() if isinstance(v, bytes) else v
                         for v in var_grp[idx_key][:]]

    return layer_keys, obs_keys, var_names


def detect_counts_source(h5ad_path: str | Path) -> tuple[str | None, str]:
    """
    Auto-detect counts layer and libsize obs key from an h5ad file.

    Uses h5py directly to read only file metadata (layer names, obs column
    names), which avoids anndata version compatibility issues with uns.

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
    layer_keys, obs_keys, _ = _h5ad_keys(h5ad_path)

    layer = next((l for l in _LAYER_CANDIDATES if l in layer_keys), None)

    obs_set = set(obs_keys)
    libsize_key = next((k for k in _LIBSIZE_CANDIDATES if k in obs_set), None)
    if libsize_key is None:
        raise ValueError(
            f"Cannot auto-detect libsize column. "
            f"Tried: {_LIBSIZE_CANDIDATES}. Available obs: {sorted(obs_set)}"
        )

    return layer, libsize_key


def _load_cc_counts_h5py(
    h5ad_path: str | Path,
    gene_list: list[str],
    counts_layer: str | None,
    libsize_obs_key: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    h5py-based fallback for load_cc_counts.

    Reads var_names, obs[libsize_obs_key], and the selected gene columns
    directly from the HDF5 file without using anndata at all, avoiding all
    anndata version compatibility issues.

    For CSC matrices (most common), only the required gene columns are read
    from disk (memory-efficient). For CSR, the full matrix is loaded.
    """
    import h5py

    with h5py.File(h5ad_path, "r") as f:
        # ── var_names ──────────────────────────────────────────────────────
        var_grp = f["var"]
        idx_key = "_index" if "_index" in var_grp else "index"
        var_names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in var_grp[idx_key][:]]

        var_idx      = {g: i for i, g in enumerate(var_names)}
        present_genes = [g for g in gene_list if g in var_idx]
        missing_genes = [g for g in gene_list if g not in var_idx]
        col_indices   = [var_idx[g] for g in present_genes]

        # ── libsize ────────────────────────────────────────────────────────
        libsize = np.array(f["obs"][libsize_obs_key][:], dtype=float)
        n_cells = len(libsize)

        # ── counts matrix ──────────────────────────────────────────────────
        if counts_layer is not None and "layers" in f and counts_layer in f["layers"]:
            mat_grp = f["layers"][counts_layer]
        elif "X" in f:
            mat_grp = f["X"]
        else:
            raise ValueError(
                f"Cannot find counts matrix. "
                f"Tried: layers['{counts_layer}'], X"
            )

        enc = mat_grp.attrs.get("encoding-type", "")
        if isinstance(enc, bytes):
            enc = enc.decode()

        n_present = len(present_genes)
        counts_dense = np.zeros((n_cells, n_present), dtype=np.float32)

        if "csc" in enc:
            # CSC: read only the needed columns — memory efficient
            data    = mat_grp["data"][:]
            indices = mat_grp["indices"][:]
            indptr  = mat_grp["indptr"][:]
            for out_col, col_idx in enumerate(col_indices):
                start, end = int(indptr[col_idx]), int(indptr[col_idx + 1])
                if start < end:
                    counts_dense[indices[start:end], out_col] = data[start:end]

        elif "csr" in enc:
            # CSR: load full sparse matrix then subset columns
            data    = mat_grp["data"][:]
            indices = mat_grp["indices"][:]
            indptr  = mat_grp["indptr"][:]
            shape   = tuple(mat_grp.attrs["shape"])
            mat = sp.csr_matrix((data, indices, indptr), shape=shape)
            counts_dense = mat[:, col_indices].toarray().astype(np.float32)

        else:
            # Dense fallback
            counts_dense = np.array(mat_grp[:, col_indices], dtype=np.float32)

    return counts_dense, libsize, present_genes, missing_genes


def load_cc_counts(
    h5ad_path: str | Path,
    gene_list: list[str],
    counts_layer: str | None = "counts_RNA",
    libsize_obs_key: str = "nCount_RNA",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load raw-count expression for CC-gene subset from an h5ad file.

    Primary path: anndata backed='r' mode with CSC-efficient column
    subsetting (only ~100 gene columns loaded into memory).

    Fallback path: if anndata raises a version-compatibility error on uns
    (common with Seurat-converted h5ad files, e.g. uns/log1p/base=None),
    automatically switches to a pure h5py reader that reads only the
    needed columns directly — no file modification required.

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

    # ── Primary: anndata backed mode ──────────────────────────────────────
    try:
        adata = ad.read_h5ad(h5ad_path, backed="r")
    except Exception as e:
        err_str = str(e) + type(e).__name__
        is_uns_issue = "IORegistry" in err_str or "null" in err_str or "log1p" in err_str
        is_unicode_issue = isinstance(e, UnicodeDecodeError)
        if not is_uns_issue and not is_unicode_issue:
            raise  # unrelated error
        reason = "uns compatibility issue" if is_uns_issue else "UnicodeDecodeError in obs (non-ASCII cell labels)"
        print(
            f"[WARN] anndata backed read failed ({reason}).\n"
            f"       Switching to h5py fallback — no file modification needed.\n"
            f"       (Error: {e})"
        )
        return _load_cc_counts_h5py(h5ad_path, gene_list, counts_layer, libsize_obs_key)

    # ── Normal anndata path ───────────────────────────────────────────────
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


def write_pi_h5py(
    input_path: str | Path,
    output_path: str | Path,
    pi_values: np.ndarray,
    obs_key: str,
    params_dict: dict,
) -> None:
    """Write PI column and params to h5ad via h5py when anndata full load fails.

    Copies input → output, then appends obs/<obs_key> (float64 array) and
    uns/proliferation_index_params (dict group) directly using h5py — no
    anndata read required, so non-ASCII obs string columns are never touched.
    """
    import shutil
    import h5py

    shutil.copy2(str(input_path), str(output_path))

    with h5py.File(str(output_path), "a") as f:
        # ── obs/PI ────────────────────────────────────────────────────────
        obs = f["obs"]
        if obs_key in obs:
            del obs[obs_key]
        ds = obs.create_dataset(obs_key, data=pi_values.astype(np.float64))
        ds.attrs["encoding-type"] = "array"
        ds.attrs["encoding-version"] = "0.2.0"

        # ── uns/proliferation_index_params ────────────────────────────────
        uns = f.require_group("uns")
        pkey = "proliferation_index_params"
        if pkey in uns:
            del uns[pkey]
        pgrp = uns.require_group(pkey)
        pgrp.attrs["encoding-type"] = "dict"
        pgrp.attrs["encoding-version"] = "0.1.0"

        str_dt = h5py.string_dtype(encoding="utf-8")

        # scalar float
        ds = pgrp.create_dataset("r_threshold", data=float(params_dict["r_threshold"]))
        ds.attrs["encoding-type"] = "scalar"
        ds.attrs["encoding-version"] = "0.2.0"

        # scalar ints
        for k in ("n_genes_S", "n_genes_G2M"):
            ds = pgrp.create_dataset(k, data=int(params_dict[k]))
            ds.attrs["encoding-type"] = "scalar"
            ds.attrs["encoding-version"] = "0.2.0"

        # scalar strings
        for k in ("counts_layer", "libsize_key"):
            ds = pgrp.create_dataset(k, data=str(params_dict[k]), dtype=str_dt)
            ds.attrs["encoding-type"] = "scalar"
            ds.attrs["encoding-version"] = "0.2.0"

        # string arrays (gene lists)
        for k in ("genes_S", "genes_G2M"):
            gene_list = params_dict[k]
            ds = pgrp.create_dataset(k, data=np.array(gene_list, dtype=object), dtype=str_dt)
            ds.attrs["encoding-type"] = "array"
            ds.attrs["encoding-version"] = "0.2.0"
