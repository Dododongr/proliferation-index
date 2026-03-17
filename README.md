# Proliferation Index (PI) — adaptive scRNA-seq scoring

Compute a data-adaptive **Proliferation Index** for any scRNA-seq h5ad file.
Implements the SciPlex / monocle3 PI formula with automatic gene selection via Leave-One-Out (LOO) Spearman correlation.

---

## Algorithm

```mermaid
flowchart TD
    A["Input h5ad\n(gene symbol var_names)"] --> B["Load 100-gene candidate pool\n46 S-phase + 54 G2M-phase"]
    B --> C["Auto-detect counts source\ncounts_RNA / counts / adata.X\nnCount_RNA / total_counts"]
    C --> D["LOO Spearman correlation\nbacked mode — only CC genes in RAM"]
    D --> E["Show threshold table\nr ≥ 0.2 / 0.3 / 0.4 / 0.5"]
    E --> F{"Select threshold\n--r-threshold or interactive"}
    F --> G["Filter to passing genes"]
    G --> H["Compute PI\nlog1p((ΣS + ΣG2M) / nCount)"]
    H --> I["Load full h5ad\n⚠ peak RAM"]
    I --> J["Save adata.obs['proliferation_index']\n+ adata.uns['proliferation_index_params']"]
    J --> K["Output\n_pi.h5ad  +  _run.log"]
```

### PI formula
```
PI = log1p( (Σ S-phase raw counts + Σ G2M-phase raw counts) / total UMI )
```
Matches the monocle3-based `proliferation_index` from [Srivatsan et al. 2020 (SciPlex)](https://www.science.org/doi/10.1126/science.aax6234).

---

## Validation

Tested against monocle3 `proliferation_index` on **SciPlex3** (581,777 cells × 58,347 genes):

| Gene list       | N genes | Spearman r | A549   | K562   | MCF7   |
|-----------------|---------|-----------|--------|--------|--------|
| Pool (100)      | 100     | **0.906** | 0.994  | 0.994  | 0.996  |
| LOO r ≥ 0.2     | 30      | 0.776     | 0.905  | 0.887  | 0.894  |
| LOO r ≥ 0.3     | 9       | 0.727     | 0.811  | 0.776  | 0.839  |
| LOO r ≥ 0.4     | 2       | 0.608     | 0.664  | 0.552  | 0.695  |

> **Recommended threshold: r ≥ 0.2** — best balance of gene coverage and correlation.

---

## Installation

```bash
# Clone
git clone <repo-url>
cd proliferation_index

# Dependencies (conda)
conda activate scvi   # anndata, scanpy, scipy, numpy, pandas

# Optional — for RAM tracking
pip install psutil
```

---

## Usage

### Interactive (recommended for first run)
```bash
python compute_pi.py --input data.h5ad
```
Runs LOO, shows gene counts per threshold, asks you to pick.

### Non-interactive (pipeline)
```bash
python compute_pi.py --input data.h5ad --r-threshold 0.2
```

### Custom output path
```bash
python compute_pi.py \
  --input  data.h5ad \
  --output data_with_pi.h5ad \
  --r-threshold 0.2
```

### All options
| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Input h5ad (gene symbol var_names) |
| `--output` | `<input>_pi.h5ad` | Output h5ad path |
| `--r-threshold` | interactive | LOO r threshold: 0.2 / 0.3 / 0.4 / 0.5 |
| `--preset` | `auto` | `auto` · `seurat` · `scanpy` |
| `--counts-layer` | auto-detected | Override raw counts layer name |
| `--libsize-key` | auto-detected | Override total UMI obs column name |
| `--gene-pool` | built-in | Path to custom gene pool JSON |

---

## Input requirements

| Field | Required | Notes |
|---|---|---|
| `var_names` | gene symbols | e.g. `MCM5`, not Ensembl IDs |
| `layers['counts_RNA']` | one of these | Seurat-converted h5ad |
| `layers['counts']` | one of these | Scanpy-native |
| `adata.X` | one of these | fallback |
| `obs['nCount_RNA']` | one of these | Seurat total UMI |
| `obs['total_counts']` | one of these | Scanpy total UMI |

Auto-detected by `--preset auto` (default). Override with `--preset seurat/scanpy` or explicit `--counts-layer` / `--libsize-key`.

---

## Output

**`_pi.h5ad`** — original h5ad with two additions:

```python
adata.obs["proliferation_index"]       # float, log1p scale, one value per cell

adata.uns["proliferation_index_params"] = {
    "r_threshold":  0.2,
    "counts_layer": "counts_RNA",
    "libsize_key":  "nCount_RNA",
    "n_genes_S":    5,
    "n_genes_G2M":  25,
    "genes_S":      [...],
    "genes_G2M":    [...],
}
```

**`_run.log`** — timestamped RAM and runtime log per phase.
**`_error.log`** — saved only on failure, includes full traceback.

---

## Gene lists

| File | N genes | Description |
|---|---|---|
| `cc_genes_pool.json` | 100 | Full candidate pool (default) |
| `atomic_cc_genes.json` | 97 | Pre-LOO reference list |
| `regev_lab.json` | — | Tirosh et al. 2015 |
| `seurat_default.json` | — | Seurat built-in |

---

## Memory estimate

| Phase | RAM |
|---|---|
| LOO (backed mode) | n_cells × 100 genes × 4 B ≈ **270 MB** for 336k cells |
| Write-back (full load) | ≥ 2× h5ad file size |
