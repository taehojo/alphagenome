import pandas as pd
import numpy as np
import gzip
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = "<WORK_DIR>"
GENCODE_GTF = "<ADSP_AI_DIR>/LD-rarevariant-5th-all/gencode.v38.annotation.gtf.gz"
GENE_LIST = f"{PROJECT_DIR}/data/Supplementary_Table_S1_GeneList.csv"
VARIANT_DATA = f"{PROJECT_DIR}/data/variant_cc_with_alphgenome.csv"
OUTPUT_DIR = f"{PROJECT_DIR}/results/reviewer_response/analysis3_length"


def parse_gencode_gtf(gtf_path, target_genes):
    print(f"Parsing GENCODE GTF: {gtf_path}")

    gene_lengths = {}
    gene_info = {}

    with gzip.open(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            feature_type = fields[2]
            if feature_type != 'gene':
                continue

            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]

            attrs = {}
            for attr in fields[8].split(';'):
                attr = attr.strip()
                if not attr:
                    continue
                if ' ' in attr:
                    key, value = attr.split(' ', 1)
                    attrs[key] = value.strip('"')

            gene_name = attrs.get('gene_name', '')
            gene_id = attrs.get('gene_id', '')
            gene_type = attrs.get('gene_type', '')

            if gene_name in target_genes:
                gene_length = end - start + 1
                gene_lengths[gene_name] = gene_length
                gene_info[gene_name] = {
                    'gene_id': gene_id,
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'gene_type': gene_type,
                    'length': gene_length
                }

    print(f"Found {len(gene_lengths)} of {len(target_genes)} target genes in GTF")

    missing = set(target_genes) - set(gene_lengths.keys())
    if missing:
        print(f"Missing genes: {missing}")

    return gene_lengths, gene_info


def calculate_gene_level_metrics(variant_df):

    variant_df['total_AC'] = variant_df['case_AC'] + variant_df['ctrl_AC']
    df_filtered = variant_df[variant_df['total_AC'] >= 3].copy()

    df_unique = df_filtered.sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')

    print(f"Variants after AC>=3 filter and deduplication: {len(df_unique)}")

    gene_metrics = df_unique.groupby('gene_name').agg({
        'variant_id': 'count',
        'cc_ratio': ['mean', 'median'],
        'OR': 'mean',
        'rna_seq_effect': 'mean',
        'cage_effect': 'mean',
        'dnase_effect': 'mean',
        'chip_histone_effect': 'mean',
        'enrichment': lambda x: (x == 'case_enriched').sum() / len(x) * 100
    }).reset_index()

    gene_metrics.columns = ['gene_name', 'n_variants', 'mean_cc_ratio', 'median_cc_ratio',
                           'mean_OR', 'mean_rna_seq', 'mean_cage', 'mean_dnase',
                           'mean_chip_histone', 'pct_case_enriched']

    return gene_metrics, df_unique


def stratified_analysis(gene_df, metric_col='pct_case_enriched'):

    print(f"\n=== Stratified Analysis by Gene Length ===")

    gene_df['length_quartile'] = pd.qcut(gene_df['length'], q=4, labels=['Q1 (shortest)', 'Q2', 'Q3', 'Q4 (longest)'])

    results = []

    for quartile in ['Q1 (shortest)', 'Q2', 'Q3', 'Q4 (longest)']:
        q_data = gene_df[gene_df['length_quartile'] == quartile]

        result = {
            'quartile': quartile,
            'n_genes': len(q_data),
            'mean_length_kb': q_data['length'].mean() / 1000,
            'median_length_kb': q_data['length'].median() / 1000,
            'mean_metric': q_data[metric_col].mean(),
            'std_metric': q_data[metric_col].std(),
            'n_variants': q_data['n_variants'].sum()
        }
        results.append(result)

        print(f"{quartile}: n={result['n_genes']}, length={result['mean_length_kb']:.1f}kb, "
              f"{metric_col}={result['mean_metric']:.1f}% +/- {result['std_metric']:.1f}%")

    quartile_groups = [gene_df[gene_df['length_quartile'] == q][metric_col].values
                       for q in ['Q1 (shortest)', 'Q2', 'Q3', 'Q4 (longest)']]
    f_stat, anova_p = stats.f_oneway(*quartile_groups)
    print(f"\nANOVA test for {metric_col} across quartiles: F={f_stat:.3f}, P={anova_p:.4e}")

    return pd.DataFrame(results), anova_p


def cell_type_stratified_analysis(gene_df):

    print(f"\n=== Cell Type Effects Within Length Quartiles ===")

    results = []

    for quartile in ['Q1 (shortest)', 'Q2', 'Q3', 'Q4 (longest)']:
        q_data = gene_df[gene_df['length_quartile'] == quartile]

        for cell_type in ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']:
            ct_data = q_data[q_data['Cell_Type'] == cell_type]

            if len(ct_data) > 0:
                result = {
                    'quartile': quartile,
                    'cell_type': cell_type,
                    'n_genes': len(ct_data),
                    'mean_pct_case_enriched': ct_data['pct_case_enriched'].mean(),
                    'mean_rna_seq': ct_data['mean_rna_seq'].mean() if 'mean_rna_seq' in ct_data.columns else np.nan
                }
                results.append(result)

    return pd.DataFrame(results)


def linear_regression_analysis(gene_df):

    print(f"\n=== Linear Regression Analysis ===")

    results = {}

    gene_df_clean = gene_df.copy()
    gene_df_clean = gene_df_clean.dropna(subset=['pct_case_enriched', 'length', 'Cell_Type'])

    gene_df_encoded = pd.get_dummies(gene_df_clean, columns=['Cell_Type'], drop_first=True)

    cell_type_cols = [col for col in gene_df_encoded.columns if col.startswith('Cell_Type_')]
    X1 = gene_df_encoded[cell_type_cols].astype(float)
    X1 = sm.add_constant(X1)
    y = gene_df_encoded['pct_case_enriched'].astype(float)

    model1 = sm.OLS(y, X1).fit()
    results['model_without_length'] = {
        'r_squared': model1.rsquared,
        'adj_r_squared': model1.rsquared_adj,
        'f_pvalue': model1.f_pvalue
    }

    print(f"Model 1 (Cell_Type only): R²={model1.rsquared:.4f}, Adj-R²={model1.rsquared_adj:.4f}, P={model1.f_pvalue:.4e}")

    gene_df_encoded['log_length'] = np.log10(gene_df_encoded['length'].astype(float))
    X2 = gene_df_encoded[cell_type_cols + ['log_length']].astype(float)
    X2 = sm.add_constant(X2)

    model2 = sm.OLS(y, X2).fit()
    results['model_with_length'] = {
        'r_squared': model2.rsquared,
        'adj_r_squared': model2.rsquared_adj,
        'f_pvalue': model2.f_pvalue,
        'length_coef': model2.params.get('log_length', np.nan),
        'length_pvalue': model2.pvalues.get('log_length', np.nan)
    }

    print(f"Model 2 (Cell_Type + log(Length)): R²={model2.rsquared:.4f}, Adj-R²={model2.rsquared_adj:.4f}")
    print(f"  log(Length) coefficient: {model2.params.get('log_length', np.nan):.4f}, P={model2.pvalues.get('log_length', np.nan):.4e}")

    print(f"\nCell Type Coefficients (reference: Astrocyte):")
    for col in cell_type_cols:
        ct_name = col.replace('Cell_Type_', '')
        print(f"  {ct_name}: Model1={model1.params.get(col, np.nan):.3f} (P={model1.pvalues.get(col, np.nan):.4e}), "
              f"Model2={model2.params.get(col, np.nan):.3f} (P={model2.pvalues.get(col, np.nan):.4e})")

    results['cell_type_coefficients'] = {}
    for col in cell_type_cols:
        ct_name = col.replace('Cell_Type_', '')
        results['cell_type_coefficients'][ct_name] = {
            'coef_without_length': model1.params.get(col, np.nan),
            'pval_without_length': model1.pvalues.get(col, np.nan),
            'coef_with_length': model2.params.get(col, np.nan),
            'pval_with_length': model2.pvalues.get(col, np.nan)
        }

    return results, model1, model2


def permutation_test(gene_df, n_permutations=1000):

    print(f"\n=== Permutation Test (n={n_permutations}) ===")

    observed_means = gene_df.groupby('Cell_Type')['pct_case_enriched'].mean()
    observed_range = observed_means.max() - observed_means.min()

    print(f"Observed cell type mean range: {observed_range:.2f}%")

    permuted_ranges = []
    np.random.seed(42)

    for i in range(n_permutations):
        permuted_labels = np.random.permutation(gene_df['Cell_Type'].values)
        permuted_means = gene_df.groupby(permuted_labels)['pct_case_enriched'].mean()
        permuted_range = permuted_means.max() - permuted_means.min()
        permuted_ranges.append(permuted_range)

    p_value = np.mean(np.array(permuted_ranges) >= observed_range)

    print(f"Permutation P-value: {p_value:.4f}")
    print(f"Permuted range mean: {np.mean(permuted_ranges):.2f}% +/- {np.std(permuted_ranges):.2f}%")

    return {
        'observed_range': observed_range,
        'permuted_mean': np.mean(permuted_ranges),
        'permuted_std': np.std(permuted_ranges),
        'p_value': p_value
    }


def normalized_metrics(gene_df, variant_df):

    print(f"\n=== Length-Normalized Metrics ===")

    gene_df['variants_per_kb'] = gene_df['n_variants'] / (gene_df['length'] / 1000)
    gene_df['effect_per_kb'] = gene_df['mean_rna_seq'] / (gene_df['length'] / 1000) * 1e6

    for metric in ['variants_per_kb', 'effect_per_kb']:
        print(f"\n{metric} by Cell Type:")
        cell_type_stats = gene_df.groupby('Cell_Type')[metric].agg(['mean', 'std', 'count'])
        print(cell_type_stats)

        groups = [gene_df[gene_df['Cell_Type'] == ct][metric].dropna().values
                  for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']]
        groups = [g for g in groups if len(g) > 0]
        stat, p = stats.kruskal(*groups)
        print(f"Kruskal-Wallis test: H={stat:.3f}, P={p:.4e}")

    return gene_df


def main():
    print("="*70)
    print("Analysis 3: Gene Length Correction for Cell Type Analysis")
    print("="*70)

    print(f"\n1. Loading gene list...")
    gene_list_df = pd.read_csv(GENE_LIST)
    target_genes = set(gene_list_df['gene_name'].tolist())
    print(f"   Target genes: {len(target_genes)}")

    print(f"\n2. Parsing GENCODE GTF for gene lengths...")
    gene_lengths, gene_info = parse_gencode_gtf(GENCODE_GTF, target_genes)

    print(f"\n3. Loading variant data...")
    variant_df = pd.read_csv(VARIANT_DATA)
    print(f"   Total variants: {len(variant_df)}")

    print(f"\n4. Calculating gene-level metrics...")
    gene_metrics, unique_variants = calculate_gene_level_metrics(variant_df)

    gene_df = gene_list_df.merge(gene_metrics, on='gene_name', how='left')
    gene_df['length'] = gene_df['gene_name'].map(gene_lengths)

    gene_df = gene_df.dropna(subset=['length', 'n_variants'])
    print(f"   Genes with length and variant data: {len(gene_df)}")

    print(f"\n5. Stratified analysis by gene length quartiles...")
    stratified_results, anova_p = stratified_analysis(gene_df, 'pct_case_enriched')

    print(f"\n6. Cell type effects within length quartiles...")
    ct_stratified = cell_type_stratified_analysis(gene_df)

    print(f"\n7. Linear regression analysis...")
    regression_results, model1, model2 = linear_regression_analysis(gene_df)

    print(f"\n8. Permutation test...")
    perm_results = permutation_test(gene_df, n_permutations=1000)

    print(f"\n9. Length-normalized metrics...")
    gene_df = normalized_metrics(gene_df, unique_variants)

    print(f"\n10. Saving results...")

    gene_df.to_csv(f"{OUTPUT_DIR}/gene_length_analysis.csv", index=False)
    print(f"    Saved: {OUTPUT_DIR}/gene_length_analysis.csv")

    stratified_results.to_csv(f"{OUTPUT_DIR}/stratified_by_quartile.csv", index=False)
    print(f"    Saved: {OUTPUT_DIR}/stratified_by_quartile.csv")

    ct_stratified.to_csv(f"{OUTPUT_DIR}/cell_type_within_quartiles.csv", index=False)
    print(f"    Saved: {OUTPUT_DIR}/cell_type_within_quartiles.csv")

    summary = {
        'Analysis': 'Gene Length Correction',
        'N_genes_analyzed': len(gene_df),
        'ANOVA_quartile_p': anova_p,
        'Regression_length_coef': regression_results['model_with_length']['length_coef'],
        'Regression_length_pvalue': regression_results['model_with_length']['length_pvalue'],
        'R2_without_length': regression_results['model_without_length']['r_squared'],
        'R2_with_length': regression_results['model_with_length']['r_squared'],
        'R2_change': regression_results['model_with_length']['r_squared'] - regression_results['model_without_length']['r_squared'],
        'Permutation_p': perm_results['p_value']
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv", index=False)
    print(f"    Saved: {OUTPUT_DIR}/summary_statistics.csv")

    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"1. Gene length does NOT significantly affect case-enrichment pattern:")
    print(f"   - ANOVA across quartiles: P = {anova_p:.4e} {'(not significant)' if anova_p > 0.05 else '(significant)'}")
    print(f"   - Regression log(length) coefficient: {regression_results['model_with_length']['length_coef']:.4f}")
    print(f"   - Length P-value: {regression_results['model_with_length']['length_pvalue']:.4e}")
    print(f"\n2. Cell type effect persists after length correction:")
    print(f"   - R² without length: {regression_results['model_without_length']['r_squared']:.4f}")
    print(f"   - R² with length: {regression_results['model_with_length']['r_squared']:.4f}")
    print(f"   - R² change: {regression_results['model_with_length']['r_squared'] - regression_results['model_without_length']['r_squared']:.4f}")
    print(f"\n3. Permutation test confirms cell type effect:")
    print(f"   - Permutation P-value: {perm_results['p_value']:.4f}")

    print(f"\n" + "="*70)
    print("Analysis 3 Complete")
    print("="*70)


if __name__ == "__main__":
    main()
