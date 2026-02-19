# Results Directory

Supplementary data files for **"Integrative Analysis of Rare Variants and AlphaGenome-Predicted Regulatory Impacts in Alzheimer's Disease-Associated Genes"** (Jo et al.).

All results derived from the ADSP R4 discovery cohort (N=24,595; 9,943 unique rare variants, AC≥3) and R5 replication cohort (N=11,545).

---

## Root Files — R4 Discovery Core Results

| File | Rows | Description |
|------|------|-------------|
| `cell_type_analysis.csv` | 4 | Cell-type-specific (Neuron, Microglia, Astrocyte, Ubiquitous) mean AlphaGenome scores and CC ratios |
| `gene_interaction_ratios.csv` | 66 | Gene-level interaction ratios (IR) for 66 AD genes with variant counts |
| `IR_comparison_all_populations.csv` | 8 | Modality-specific IR comparing AD vs. Non-AD gene groups (all populations) |
| `IR_comparison_pop_stratified.csv` | 8 | Bootstrap 95% CI for modality IRs, AD vs. Non-AD, population-stratified |

---

## r5_replication/ — R5 Independent Replication (Section 5.1)

R5 cohort (11,545 participants; 1,408 AD, 10,137 CN) used to replicate R4 findings.

| File | Rows | Description |
|------|------|-------------|
| `cohort_demographics.tsv` | 4 | R5 sample sizes by population (NHW, Hispanic, AA) with case:control ratios |
| `filtering_waterfall.tsv` | 3 | QC pipeline: 166,219 → 124,531 (MAF<1%, AC≥3) → 124,396 variants |
| `modality_score_comparison.tsv` | 4 | Mean AlphaGenome scores: case-enriched vs. control-enriched in R5 (Cohen's d, P-values) |
| `IR_replication.tsv` | 4 | Modality-level IR replication: R5 IR vs. R4 IR with Fisher exact P |
| `population_stratified.tsv` | 3 | Population-specific (NHW, Hispanic, AA) IR and gene-level correlation with R4 |
| `direction_concordance.tsv` | 4 | Spearman correlation of R4 vs. R5 CC ratios per population |
| `gene_IR_replication.tsv` | 43 | Gene-level IR in R5 vs. R4 with direction concordance flag |
| `bootstrap_IR_summary.tsv` | 4 | Bootstrap R5 IR (median, 95% CI) and whether R4 IR falls within R5 CI |
| `bootstrap_IR_by_population.tsv` | 300 | 100 bootstrap iterations × 3 populations: per-iteration IR and P-values |
| `downsampling_design.tsv` | 4 | Downsampling parameters to match R4 case:control ratio (~2.9:1) |

---

## gene_analysis/ — Gene-Level Analyses (Section 5.2)

Per-gene IR estimation with bootstrap confidence intervals and cross-dataset comparison.

| File | Rows | Description |
|------|------|-------------|
| `spearman_correlation.csv` | 74 | R4 per-gene Spearman/point-biserial correlation between AlphaGenome scores and case-enrichment (4 modalities) |
| `gene_IR_percentile80.csv` | 74 | Gene-level IR at 80th percentile threshold for RNA-seq, CAGE, ChIP-histone |
| `bootstrap_CI_R4.tsv` | 261 | R4 bootstrap 95% CI for gene × modality IRs (1,000 iterations) |
| `bootstrap_CI_R5.tsv` | 245 | R5 bootstrap 95% CI for gene × modality IRs (1,000 iterations) |
| `shrinkage_IR.tsv` | 261 | Empirical Bayes shrinkage-adjusted IR per gene × modality |
| `R4_vs_R5_gene_IR.tsv` | 244 | R4 vs. R5 gene-level IR comparison: direction concordance, CI overlap |

---

## regression/ — Logistic Regression Validation (Section 5.3)

Firth penalized logistic regression as an alternative to the CC ratio/IR method.

| File | Rows | Description |
|------|------|-------------|
| `modality_standardization.tsv` | 4 | Descriptive statistics (mean, SD, skewness, kurtosis) for z-scored modality scores |
| `logistic_regression.tsv` | 13 | Single- and multi-modality Firth regression ORs for R4 and R5 |
| `modality_correlation.tsv` | 4 | Spearman correlation matrix among 4 standardized modality scores |
| `regression_vs_IR.tsv` | 4 | Comparison: CC ratio IR vs. regression OR per modality (direction agreement) |

---

## sensitivity/ — Sensitivity & Robustness (Section 5.4)

Existing R4 validation (CSV) + additional robustness analyses (TSV).

| File | Rows | Description |
|------|------|-------------|
| `cadd_vs_alphgenome_auc.csv` | 7 | ROC-AUC: AlphaGenome composite (0.554) vs. CADD (0.509) vs. individual modalities |
| `cohens_d.csv` | 5 | Effect sizes (Cohen's d < 0.08) confirming signal from allele frequency, not score magnitude |
| `eqtl_enrichment.csv` | 2 | eQTL enrichment: case variants depleted for blood eQTLs (OR=0.70, P=6.3e-14) |
| `gene_length_summary.csv` | 1 | Gene length not a confounder (regression P=0.71, R² change=0.16%) |
| `AC_threshold.tsv` | 16 | IR robustness across AC thresholds (1, 3, 5, 10) × 4 modalities |
| `percentile_cutoff.tsv` | 40 | IR at varying percentile cutoffs (50th–95th) × 4 modalities |
| `decile_dose_response.tsv` | 40 | Dose-response: case-enrichment % by score decile (D1–D10) × 4 modalities |
| `confounder_correlation.tsv` | 5 | Spearman correlation of gene-level IR with potential confounders (gene length, variant count, density) |

---

## regulatory/ — Regulatory Context (Section 5.5)

ChromHMM-based functional annotation of variant locations.

| File | Rows | Description |
|------|------|-------------|
| `chromhmm_enrichment.tsv` | 30 | Enrichment of high-effect variants in ChromHMM states (Roadmap E073, 15-state) × 4 modalities. Enhancer OR=2.60, Bivalent OR=7.87 |

---

## tables/ — Manuscript Tables

Publication-formatted summary tables referenced in the manuscript.

| File | Description |
|------|-------------|
| `Table6_R5_replication.tsv` | **Table 6.** R5 replication summary: R4 IR, R5 bootstrap IR (median, 95% CI) |
| `Table7_sensitivity.tsv` | **Table 7.** Sensitivity analysis (3 panels): AC threshold, percentile cutoff, confounders |
| `TableS9_R5_demographics.tsv` | **Table S9.** R5 cohort demographics by population |
| `TableS10_R5_per_population.tsv` | **Table S10.** Population-stratified R5 replication (NHW, Hispanic, AA) |
| `TableS11_gene_IR_bootstrap.tsv` | **Table S11.** Gene-level bootstrap IR (R4 vs. R5, shrinkage-adjusted) for 54 genes |
| `TableS12_regression.tsv` | **Table S12.** Logistic regression results (single/multi-modality, R4 and R5) |
| `TableS13_chromhmm.tsv` | **Table S13.** ChromHMM regulatory element enrichment summary |
| `TableS14_decile.tsv` | **Table S14.** Decile dose-response (RNA-seq and ChIP-histone CE%) |

---

## File Inventory

| Subfolder | New | Existing | Total |
|-----------|-----|----------|-------|
| *(root)* | — | 4 | 4 |
| `r5_replication/` | 10 | — | 10 |
| `gene_analysis/` | 6 | — | 6 |
| `regression/` | 4 | — | 4 |
| `sensitivity/` | 4 | 4 | 8 |
| `regulatory/` | 1 | — | 1 |
| `tables/` | 8 | — | 8 |
| **Total** | **33** | **8** | **41** |
