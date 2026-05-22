# Phase 8: Single-Cell Validation Using Public Data (GSE174367)

External validation analyses using a public single-nucleus multiome
dataset of human prefrontal cortex (Morabito et al., *Nat Genet* 2021).

## Data Source

NCBI GEO **[GSE174367](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174367)**:
snRNA-seq + snATAC-seq from prefrontal cortex of 12 AD and 8 control donors.

Downloaded via `download_GSE174367.sh` (writes to a local data directory).

## Scripts

| File | Description |
|---|---|
| `download_GSE174367.sh` | Fetch GSE174367 filtered matrices and cell metadata from NCBI FTP |
| `utils_geo.py` | Shared utilities (cell-type mapping, gene-list helpers) |
| `analysis_A_snRNAseq_expression.py` | Per-gene cell-type expression aggregation, concordance, specificity index |
| `analysis_B_snATACseq_accessibility.py` | Variant-to-peak interval mapping, pseudobulk per-peak accessibility, correlation with AlphaGenome scores |
| `analysis_C_bulk_rnaseq.py` | Bulk RNA-seq AD vs control differential expression (Mann–Whitney U + BH FDR) |

## Inputs (not in repo)

- GSE174367 filtered feature/peak matrices (download from GEO)
- `data/Table_S2_variant_data.csv` (provided in this repo)
- `data/Table_S1_gene_list.csv` (provided in this repo)

## Outputs

Written to `results/phase8_geo_validation/{A,B,C}/`. See those folders
for column descriptions.

## Reproducing

```bash
# 1. Download GEO data (~620 MB)
bash download_GSE174367.sh

# 2. Run analyses (requires scanpy, anndata)
python3 analysis_A_snRNAseq_expression.py
python3 analysis_B_snATACseq_accessibility.py
python3 analysis_C_bulk_rnaseq.py
```

## Reference

Morabito S, Miyoshi E, Michael N, et al. Single-nucleus chromatin
accessibility and transcriptomic characterization of Alzheimer's disease.
*Nature Genetics* 2021;53:1143–1155.
