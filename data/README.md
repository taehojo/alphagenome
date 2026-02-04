# Data

## Included Data

### Supplementary_Table_S1_GeneList.csv

List of 85 AD-associated genes analyzed in this study, selected based on the ADSP Gene Verification Committee (GVC) top hits list.

| Column | Description |
|--------|-------------|
| `gene_name` | HGNC gene symbol |
| `Priority` | Evidence tier (1–4) |
| `Cell_Type` | Predominant cell type expression category |
| `N_variants` | Number of rare variants mapped to gene region |
| `N_case_enriched` | Number of variants with case-control ratio > 1 |
| `Pct_case_enriched` | Percentage of case-enriched variants |
| `Mean_CC` | Mean case-control allele frequency ratio |
| `Median_CC` | Median case-control allele frequency ratio |
| `RNA_seq` | Mean AlphaGenome RNA-seq effect score |
| `CAGE` | Mean AlphaGenome CAGE effect score |

## Data Not Included

### variant_cc_with_alphgenome.csv (Primary Analysis File)

The main analysis file (18,412 rows, 33 columns) contains individual-level variant data derived from ADSP WGS and cannot be publicly shared due to data use agreements.

**Schema (33 columns):**

| Column | Description |
|--------|-------------|
| `variant_id` | Unique variant identifier (chr:pos:ref:alt) |
| `chr` | Chromosome |
| `pos` | Genomic position (GRCh38) |
| `ref` | Reference allele |
| `alt` | Alternative allele |
| `gene` | Mapped gene symbol |
| `region` | Variant location (promoter/gene_body/utr3) |
| `case_AC` | Case allele count |
| `case_AN` | Case allele number |
| `case_AF` | Case allele frequency |
| `ctrl_AC` | Control allele count |
| `ctrl_AN` | Control allele number |
| `ctrl_AF` | Control allele frequency |
| `cc_ratio` | Case-control allele frequency ratio (case_AF / ctrl_AF) |
| `total_AC` | Total allele count (case_AC + ctrl_AC) |
| `rna_seq` | AlphaGenome RNA-seq effect score |
| `cage` | AlphaGenome CAGE effect score |
| `dnase` | AlphaGenome DNase effect score |
| `chip_histone` | AlphaGenome ChIP histone effect score |
| *(+ additional AlphaGenome modality columns)* | |

**Standard preprocessing:**
```python
df = pd.read_csv('data/variant_cc_with_alphgenome.csv')
df['total_AC'] = df['case_AC'] + df['ctrl_AC']
df_filtered = df[df['total_AC'] >= 3]  # AC>=3 filter
df_unique = df_filtered.sort_values('total_AC', ascending=False) \
                       .drop_duplicates('variant_id', keep='first')
# Result: 9,943 unique variants
```

## Accessing ADSP Data

The individual-level genetic data used in this study were obtained from the Alzheimer's Disease Sequencing Project (ADSP).

To access ADSP data:

1. Visit [NIAGADS](https://adsp.niagads.org/)
2. Submit a data access request through the NIAGADS Data Sharing Service
3. Approval is required from the ADSP Data Access Committee
4. Once approved, WGS data (Release 4) can be downloaded

**ADSP WGS R4 includes:**
- 24,595 participants (6,296 AD cases, 18,299 controls)
- Whole genome sequencing data
- Phenotype and covariate information

For questions about data access, contact NIAGADS at https://adsp.niagads.org/.
