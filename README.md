# Rare variants and AlphaGenome-predicted regulatory impact in Alzheimer's disease genes

Analysis code for Jo et al., *Cell Reports* (CELL-REPORTS-D-26-03243).

Rare variants (MAF < 1%) in 85 AD-associated genes were scored with AlphaGenome
across eight regulatory modalities and compared against case-control allele
frequencies in 24,595 ADSP R4 participants, with an independent assessment in
11,545 ADSP R5 participants.

## Data

No participant-level or variant-level data are distributed here. Obtain them from:

| Source | Where |
|---|---|
| ADSP R4 / R5 whole-genome sequencing | dbGaP `phs000572` (controlled access; apply through NIAGADS) |
| snRNA-seq / snATAC-seq, prefrontal cortex | GEO `GSE174367` |
| ChromHMM 15-state, 8 brain epigenomes | https://egg2.wustl.edu/roadmap/ |
| cis-eQTL, whole blood | https://www.eqtlgen.org |
| cis-eQTL, brain (v10) | https://gtexportal.org |
| GWAS lead variants | Bellenguez et al. 2022, Supplementary Table 5, doi:10.1038/s41588-022-01024-z |
| Gene annotation | GENCODE v38, GRCh38 |

Place downloads under `data/` and pass paths on the command line.

`data/Table_S2_variant_data.csv` holds the 9,943 analysed variants with their
AlphaGenome scores and case-control summary statistics, the same table served by
the browser at https://taehojo.github.io/rarevariants/ . It contains no
individual-level genotypes.

## Pipeline

| Script | Produces |
|---|---|
| `01_extract_rare_variants.py` | Per-ancestry rare-variant extraction (PLINK 2.0) and mapping to the 85 gene regions |
| `02_alphagenome_predict.py` | AlphaGenome scores, 8 modalities, via the API |
| `03_interaction_ratio.py` | Table 2, Supplementary Tables S5, S7 (A-C), S8 |
| `04_clustering_models.py` | Supplementary Tables S19, S23 (GEE, RINT) |
| `05_permutation_specificity.py` | Supplementary Table S27 (1,000 size-matched gene sets) |
| `06_noncoding_strata.py` | Supplementary Table S22 (VEP three strata) |
| `07_r5_assessment.py` | Supplementary Tables S11, S18 (R5 assessment) |
| `08_snatac_concordance.py` | Supplementary Table S13, Figure S6 |
| `09_chromhmm_enrichment.py` | Supplementary Tables S14a/b, Figure S7 |
| `10_bellenguez_concordance.py` | Supplementary Table S20 |

`03_interaction_ratio.py` is the entry point and reproduces the primary result:

```
pip install -r requirements.txt
python code/03_interaction_ratio.py --variants data/variant_table.csv --outdir results
```

Its outputs in `results/paper_tables/` are the values printed in the paper. The analysis set
is variants with allele count >= 3, deduplicated on chromosome:position:REF:ALT,
carrying all four primary AlphaGenome scores and a finite case-control ratio
(n = 9,866); case-only variants are excluded.

## Requirements

Python >= 3.8 with the packages in `requirements.txt`. PLINK 2.0 for script 01,
Ensembl VEP v110 for script 06.

## Related

- AlphaGenome MCP server: https://github.com/taehojo/alphagenome-mcp
- Variant browser: https://taehojo.github.io/rarevariants/
