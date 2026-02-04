# Integrative Analysis of Rare Variants and AlphaGenome-Predicted Regulatory Impacts within Alzheimer's Disease-Associated Genes

## Overview

This repository contains the analysis code and summary-level results for our study investigating the predicted regulatory impacts of rare variants enriched in Alzheimer's disease (AD) within 85 AD-associated genes. We analyzed 1,838,711 rare variants (MAF < 0.1%) from 24,595 ADSP WGS participants and used [AlphaGenome](https://deepmind.google/technologies/alphagenome/) to predict variant-level regulatory effects across multiple functional modalities.

## Repository Structure

```
alphagenome/
├── code/
│   ├── phase0_data_prep/        # Gene extraction, phenotype processing
│   ├── phase1_rare_variants/    # PLINK2 rare variant extraction & gene mapping
│   ├── phase2_alphgenome/       # AlphaGenome API prediction (11 modalities)
│   ├── phase3_plink/            # PLINK frequency analysis (case/control)
│   ├── phase4_analysis/         # Case-control ratio, enrichment, reviewer analyses
│   └── phase5_figures/          # Manuscript figure generation
├── data/
│   ├── README.md                # Data access instructions (ADSP application)
│   └── Supplementary_Table_S1_GeneList.csv  # 85 AD gene list
├── results/
│   ├── gene_interaction_ratios.csv         # Gene-level interaction ratios
│   ├── cell_type_analysis.csv              # Cell type-specific summary statistics
│   ├── IR_comparison_all_populations.csv   # AD vs Non-AD IR comparison
│   ├── IR_comparison_pop_stratified.csv    # Population-stratified IR
│   └── sensitivity/                        # Sensitivity & robustness analyses
```

## Requirements

```
Python >= 3.8
pandas
numpy
scipy
matplotlib
seaborn
intervaltree
```

PLINK 2.0 is required for Phase 1 variant extraction. PLINK 1.9 is required for Phase 3 frequency analysis.

## Data Access

Individual-level genetic data are from the Alzheimer's Disease Sequencing Project (ADSP) WGS Release 4 and are available through [NIAGADS](https://adsp.niagads.org/) upon approval. See `data/README.md` for details.

Gene-level summary statistics are included in this repository under `results/`.

## Note on Code Paths

All analysis code was executed on the Indiana University high-performance computing (HPC) environment. File paths in the code reference that specific infrastructure. To reproduce the analysis, update paths to match your local data locations.

## Contact

- Taeho Jo - tjo@iu.edu
- Center for Neuroimaging, Department of Radiology and Imaging Sciences
- Indiana University School of Medicine, Indianapolis, IN

## License

MIT License. See individual data use agreements for ADSP data restrictions.
