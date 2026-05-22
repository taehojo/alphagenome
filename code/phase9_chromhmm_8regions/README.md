# Phase 9: ChromHMM Regulatory State Enrichment Across Eight Brain Epigenomes

ChromHMM 15-state regulatory enrichment analysis using Roadmap
Epigenomics annotations for eight brain epigenomes.

## Epigenomes

| ID | Tissue |
|---|---|
| E067 | Angular Gyrus |
| E068 | Anterior Caudate |
| E069 | Cingulate Gyrus |
| E071 | Hippocampus Middle |
| E072 | Inferior Temporal Lobe |
| E073 | Dorsolateral Prefrontal Cortex |
| E074 | Substantia Nigra |
| E125 | NH-A Astrocytes |

ChromHMM 15-state core annotations are obtained from the
[Roadmap Epigenomics Consortium](https://egg2.wustl.edu/roadmap/web_portal/chr_state_learning.html).

## Scripts

| File | Description |
|---|---|
| `step5_chromhmm_enrichment.py` | hg38→hg19 liftOver, chromHMM intersection, Fisher's exact tests per state and grouped category, BH FDR |
| `step6_variant_deepdive.py` | Per-variant profile builder for top variants (HDF5 track data, TF binding, histone marks) |
| `step7_final_report.py` | HTML summary report generator |

## Inputs (not in repo)

- Deduplicated rare variant table (AC ≥ 3) with case/control allele frequencies
- 8 chromHMM 15-state bed.gz files from Roadmap Epigenomics
- UCSC `hg38ToHg19.over.chain.gz`

## Outputs

Written to `results/phase9_chromhmm_8regions/`:

- `chromhmm_enrichment_all.csv` — per-state Fisher tests
- `chromhmm_grouped_enrichment.csv` — grouped-state Fisher tests
- `variants_chromhmm_annotated.csv` — per-variant chromHMM state assignment

## Reproducing

```bash
python3 step5_chromhmm_enrichment.py
```

## Reference

Roadmap Epigenomics Consortium et al. *Nature* 2015;518:317–330.
