# Integrative Analysis of Rare Variants and AlphaGenome-Predicted Regulatory Impacts within Alzheimer's Disease-Associated Genes

## Overview

This repository contains the analysis code and summary-level results for our study investigating the predicted regulatory impacts of rare variants enriched in Alzheimer's disease (AD) within 85 AD-associated genes. We analyzed 1,801,711 rare variants (MAF < 1%) from 24,595 ADSP WGS participants and used [AlphaGenome](https://deepmind.google/technologies/alphagenome/) to predict variant-level regulatory effects across 11 functional modalities, identifying 9,943 unique variants (AC >= 3) mapped to gene regions.

## Repository Structure

```
rarevariants/
├── code/
│   ├── phase0_data_prep/             # Gene extraction, phenotype processing
│   ├── phase1_rare_variants/         # PLINK2 rare variant extraction & gene mapping
│   ├── phase2_alphgenome/            # AlphaGenome API prediction (11 modalities, developed using alphagenome-mcp)
│   ├── phase3_plink/                 # PLINK frequency analysis (case/control)
│   ├── phase4_analysis/              # Case-control ratio, enrichment, reviewer analyses
│   ├── phase6_r5_replication/        # R5 independent replication (N=11,545)
│   ├── phase7_validation/            # Statistical validation & robustness
│   ├── phase8_geo_validation/        # Single-cell validation using GSE174367
│   └── phase9_chromhmm_8regions/     # ChromHMM enrichment across 8 brain epigenomes
├── data/
│   ├── Table_S1_gene_list.csv        # 85 AD gene list (ADSP GVC)
│   └── Table_S2_variant_data.csv     # 9,943 unique variants with AlphaGenome scores
└── results/
    ├── gene_interaction_ratios.csv         # Gene-level interaction ratios
    ├── cell_type_analysis.csv              # Cell type-specific summary statistics
    ├── IR_comparison_all_populations.csv   # AD vs Non-AD IR comparison
    ├── IR_comparison_pop_stratified.csv    # Population-stratified IR
    ├── sensitivity/                        # Sensitivity & robustness analyses
    ├── phase8_geo_validation/              # snRNA-seq, snATAC-seq, bulk RNA-seq outputs
    └── phase9_chromhmm_8regions/           # Per-state and grouped chromHMM enrichment tables
```

## Requirements

- Python >= 3.8
- pandas, numpy, scipy, matplotlib, seaborn, intervaltree, statsmodels
- PLINK 2.0 (Phase 1 variant extraction)
- PLINK 1.9 (Phase 3 frequency analysis)
- scanpy, anndata (Phase 8 single-cell analyses)

## Data Access

Individual-level genetic data are from the Alzheimer's Disease Sequencing Project (ADSP) WGS Release 4 and are available through [NIAGADS](https://adsp.niagads.org/) upon approval. See `data/README.md` for details.

External public datasets used in Phases 8–9:

- GSE174367 (Morabito et al., *Nat Genet* 2021) — NCBI GEO
- Roadmap Epigenomics 15-state chromHMM tracks (E067–E074, E125)

Gene-level and variant-level summary statistics are included in this repository under `data/` and `results/`.

## Note on Code Paths

All analysis code was executed on the Indiana University high-performance computing (HPC) environment. File paths in the code reference that specific infrastructure. To reproduce the analysis, update paths to match your local data locations.

## Contact

- Taeho Jo - tjo@iu.edu
- Center for Neuroimaging, Department of Radiology and Imaging Sciences
- Indiana University School of Medicine, Indianapolis, IN

## License

MIT License. See individual data use agreements for ADSP data restrictions.
