# Data

Two files are distributed here. Neither contains individual-level genotypes.

### `Table_S1_gene_list.csv`

The 85 AD-associated genes analysed in the study (ADSP Gene Verification
Committee top-hits list), with GRCh38 coordinates and the cell-type category
used in the cell-type-stratified analyses.

The number in the filename is historical and does not correspond to any table
number in the current manuscript.

`Cell_Type` is derived from `cell_type_dictionary.csv` with each gene counted
once, which is how the manuscript reports it: genes assigned to two categories
(CLU, ANK3) are counted under Neuron, the single oligodendrocyte gene (MAF) is
grouped with Ubiquitous, and SLC2A4RG, which is absent from the dictionary, is
also grouped with Ubiquitous. This gives Neuron 16, Microglia 14, Astrocyte 5
and Ubiquitous 50.

### `cell_type_dictionary.csv`

The full hand-curated dictionary as used in the analysis: 89 entries covering 87
unique genes, in `gene,category` form.

| Category | Entries |
|---|---|
| Neuron | 17 |
| Microglia | 15 |
| Astrocyte | 7 |
| Oligodendrocyte | 2 |
| Ubiquitous | 48 |

CLU appears under both Neuron and Astrocyte, and ANK3 under both Neuron and
Oligodendrocyte. APOE, CD33 and MAPT are in the dictionary but not in the
analysed 85-gene set. These are literature-based priors, not derived from
single-nucleus RNA-seq or any reference atlas.

### Everything else

Obtained from the sources listed in the top-level README. ADSP genotype and
phenotype data are under controlled access (dbGaP phs000572) and cannot be
redistributed.
