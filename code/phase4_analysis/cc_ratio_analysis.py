import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

ANALYSIS_DIR = '<ANALYSIS_DIR>'

TABLE1_GENES = [
    'SORT1', 'CR1', 'ADAM17', 'PRKD3', 'NCK2', 'BIN1', 'WDR12', 'INPP5D', 'MME', 'IDUA',
    'CLNK', 'RHOH', 'ANKH', 'COX7C', 'TNIP1', 'RASGEF1C', 'HLA-DQA1', 'UNC5CL', 'TREM2',
    'TREML2', 'CD2AP', 'HS3ST5', 'UMAD1', 'ICA1', 'TMEM106B', 'JAZF1', 'EPDR1', 'SEC61G',
    'SPDYE3', 'EPHA1', 'CTSB', 'PTK2B', 'CLU', 'SHARPIN', 'ABCA1', 'USP6NL', 'ANK3',
    'TSPAN14', 'BLNK', 'PLEKHA1', 'SPI1', 'MS4A4A', 'EED', 'SORL1', 'TPCN1', 'FERMT2',
    'SLC24A4', 'SPPL2A', 'MINDY2', 'APH1B', 'SNX1', 'CTSH', 'DOC2A', 'BCKDK', 'IL34',
    'MAF', 'PLCG2', 'FOXF1', 'PRDM7', 'WDR81', 'SCIMP', 'MYO15A', 'GRN', 'WNT3', 'ABI3',
    'TSPOAP1', 'ACE', 'ABCA7', 'KLF16', 'SIGLEC11', 'LILRB2', 'RBCK1', 'CASS4', 'SLC2A4RG',
    'APP', 'ADAMTS1'
]

TABLE2_GENES = [
    'APOE', 'PSEN1', 'PSEN2', 'APBB3', 'CASP7', 'MS4A6A', 'PILRA', 'ADAM10', 'RIN3', 'PICALM'
]

ALL_86_GENES = TABLE1_GENES + TABLE2_GENES


def load_variant_data():
    print("=" * 70)
    print("LOADING VARIANT DATA")
    print("=" * 70)

    df = pd.read_csv(f'{ANALYSIS_DIR}/variant_level_case_control.csv')
    print(f"  Loaded {len(df):,} variant-gene-population records")

    return df


def calculate_cc_ratio(df):
    print("\n" + "=" * 70)
    print("CALCULATING CASE/CONTROL RATIOS")
    print("=" * 70)

    df = df.copy()

    df['cc_ratio'] = np.where(
        df['ctrl_AF'] > 0,
        df['case_AF'] / df['ctrl_AF'],
        np.where(df['case_AF'] > 0, np.inf, np.nan)
    )

    df['enrichment'] = np.select(
        [
            (df['case_AC'] > 0) & (df['ctrl_AC'] == 0),
            (df['case_AC'] == 0) & (df['ctrl_AC'] > 0),
            df['cc_ratio'] > 1,
            df['cc_ratio'] < 1,
            df['cc_ratio'] == 1
        ],
        ['case_only', 'ctrl_only', 'case_enriched', 'ctrl_enriched', 'neutral'],
        default='undefined'
    )

    df['log2_cc_ratio'] = np.where(
        (df['cc_ratio'] > 0) & (df['cc_ratio'] < np.inf),
        np.log2(df['cc_ratio']),
        np.nan
    )

    return df


def summarize_by_population(df):
    print("\n" + "=" * 70)
    print("CC RATIO SUMMARY BY POPULATION")
    print("=" * 70)

    populations = df['population'].unique()

    results = []
    for pop in populations:
        pop_df = df[df['population'] == pop]

        counts = pop_df['enrichment'].value_counts()

        shared = pop_df[pop_df['cc_ratio'].notna() & (pop_df['cc_ratio'] < np.inf)]

        n_total = len(pop_df)
        n_case_only = counts.get('case_only', 0)
        n_ctrl_only = counts.get('ctrl_only', 0)
        n_case_enriched = counts.get('case_enriched', 0)
        n_ctrl_enriched = counts.get('ctrl_enriched', 0)
        n_shared = len(shared)

        if n_shared > 0:
            median_cc = shared['cc_ratio'].median()
            mean_cc = shared['cc_ratio'].mean()
            mean_log2 = shared['log2_cc_ratio'].mean()

            t_stat, t_p = stats.ttest_1samp(shared['log2_cc_ratio'].dropna(), 0)

            from scipy.stats import binomtest
            binom_result = binomtest(n_case_enriched, n_case_enriched + n_ctrl_enriched, 0.5)
            binom_p = binom_result.pvalue
        else:
            median_cc = mean_cc = mean_log2 = t_stat = t_p = binom_p = np.nan

        results.append({
            'population': pop,
            'n_variants': n_total,
            'case_only': n_case_only,
            'ctrl_only': n_ctrl_only,
            'case_enriched': n_case_enriched,
            'ctrl_enriched': n_ctrl_enriched,
            'shared': n_shared,
            'median_cc_ratio': median_cc,
            'mean_cc_ratio': mean_cc,
            'mean_log2_cc': mean_log2,
            't_stat': t_stat,
            't_pvalue': t_p,
            'binom_p': binom_p
        })

        print(f"\n  {pop}:")
        print(f"    Total variants: {n_total:,}")
        print(f"    Case-only: {n_case_only} | Control-only: {n_ctrl_only}")
        print(f"    Case-enriched (CC>1): {n_case_enriched} | Control-enriched (CC<1): {n_ctrl_enriched}")
        print(f"    Median CC ratio: {median_cc:.3f}")
        print(f"    Mean log2(CC): {mean_log2:.4f} (t-test P = {t_p:.2e})")
        print(f"    Binomial test (case vs ctrl enriched): P = {binom_p:.2e}")

    return pd.DataFrame(results)


def summarize_by_gene(df):
    print("\n" + "=" * 70)
    print("CC RATIO SUMMARY BY GENE (All populations combined)")
    print("=" * 70)

    gene_results = []

    for gene in ALL_86_GENES:
        gene_df = df[df['gene_name'] == gene]

        if len(gene_df) == 0:
            continue

        shared = gene_df[gene_df['cc_ratio'].notna() & (gene_df['cc_ratio'] < np.inf)]

        n_total = len(gene_df)
        n_case_only = len(gene_df[gene_df['enrichment'] == 'case_only'])
        n_ctrl_only = len(gene_df[gene_df['enrichment'] == 'ctrl_only'])
        n_case_enriched = len(gene_df[gene_df['enrichment'] == 'case_enriched'])
        n_ctrl_enriched = len(gene_df[gene_df['enrichment'] == 'ctrl_enriched'])
        n_shared = len(shared)

        if n_shared > 0:
            median_cc = shared['cc_ratio'].median()
            mean_log2 = shared['log2_cc_ratio'].mean()

            if len(shared['log2_cc_ratio'].dropna()) > 2:
                t_stat, t_p = stats.ttest_1samp(shared['log2_cc_ratio'].dropna(), 0)
            else:
                t_stat, t_p = np.nan, np.nan

            pct_case_enriched = n_case_enriched / (n_case_enriched + n_ctrl_enriched) * 100 if (n_case_enriched + n_ctrl_enriched) > 0 else np.nan
        else:
            median_cc = mean_log2 = t_stat = t_p = pct_case_enriched = np.nan

        gene_results.append({
            'gene': gene,
            'gene_source': 'Table1_GWAS' if gene in TABLE1_GENES else 'Table2_HighConf',
            'n_variants': n_total,
            'case_only': n_case_only,
            'ctrl_only': n_ctrl_only,
            'case_enriched': n_case_enriched,
            'ctrl_enriched': n_ctrl_enriched,
            'pct_case_enriched': pct_case_enriched,
            'median_cc_ratio': median_cc,
            'mean_log2_cc': mean_log2,
            't_pvalue': t_p
        })

    gene_df = pd.DataFrame(gene_results)
    gene_df = gene_df.sort_values('median_cc_ratio', ascending=False)

    print("\n  Top 20 genes by median CC ratio:")
    print("-" * 100)
    print(f"  {'Gene':<12} {'Source':<15} {'Variants':<10} {'Case-enr':<10} {'Ctrl-enr':<10} {'%Case':<8} {'Med CC':<10} {'P-value':<12}")
    print("-" * 100)

    for _, row in gene_df.head(20).iterrows():
        pct = f"{row['pct_case_enriched']:.1f}" if not pd.isna(row['pct_case_enriched']) else "N/A"
        p_val = f"{row['t_pvalue']:.2e}" if not pd.isna(row['t_pvalue']) else "N/A"
        print(f"  {row['gene']:<12} {row['gene_source']:<15} {int(row['n_variants']):<10} {int(row['case_enriched']):<10} {int(row['ctrl_enriched']):<10} {pct:<8} {row['median_cc_ratio']:<10.3f} {p_val:<12}")

    print("\n  Summary:")
    print(f"    Genes with median CC > 1 (case-biased): {len(gene_df[gene_df['median_cc_ratio'] > 1])}")
    print(f"    Genes with median CC < 1 (control-biased): {len(gene_df[gene_df['median_cc_ratio'] < 1])}")

    sig_genes = gene_df[gene_df['t_pvalue'] < 0.05]
    print(f"    Genes with significant CC bias (P < 0.05): {len(sig_genes)}")

    return gene_df


def create_visualizations(df, pop_summary, gene_summary):
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]

    shared = df[(df['cc_ratio'] > 0) & (df['cc_ratio'] < np.inf) & (df['cc_ratio'] < 10)]

    for pop in ['NHW', 'AA', 'Hispanic', 'Asian']:
        pop_data = shared[shared['population'] == pop]['cc_ratio']
        if len(pop_data) > 0:
            ax1.hist(pop_data, bins=50, alpha=0.5, label=f'{pop} (n={len(pop_data):,})', density=True)

    ax1.axvline(x=1, color='black', linestyle='--', linewidth=2, label='CC=1')
    ax1.set_xlabel('Case/Control Ratio')
    ax1.set_ylabel('Density')
    ax1.set_title('A. CC Ratio Distribution by Population')
    ax1.legend()
    ax1.set_xlim(0, 5)

    ax2 = axes[0, 1]

    log2_data = df['log2_cc_ratio'].dropna()
    log2_data = log2_data[(log2_data > -5) & (log2_data < 5)]

    ax2.hist(log2_data, bins=100, color='#3498DB', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='log2(CC)=0')
    ax2.axvline(x=log2_data.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean={log2_data.mean():.3f}')

    ax2.set_xlabel('log2(Case/Control Ratio)')
    ax2.set_ylabel('Count')
    ax2.set_title('B. log2(CC Ratio) Distribution (All Populations)')
    ax2.legend()

    ax3 = axes[1, 0]

    gene_sorted = gene_summary.sort_values('median_cc_ratio', ascending=True).tail(30)
    colors = ['#E74C3C' if row['median_cc_ratio'] > 1 else '#3498DB'
              for _, row in gene_sorted.iterrows()]

    ax3.barh(range(len(gene_sorted)), gene_sorted['median_cc_ratio'], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(gene_sorted)))
    ax3.set_yticklabels(gene_sorted['gene'], fontsize=8)
    ax3.axvline(x=1, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Median CC Ratio')
    ax3.set_title('C. Gene-level Median CC Ratio (Top 30)')

    ax4 = axes[1, 1]

    categories = ['case_only', 'case_enriched', 'ctrl_enriched', 'ctrl_only']
    x = np.arange(len(pop_summary))
    width = 0.2

    for i, cat in enumerate(categories):
        values = pop_summary[cat].values
        offset = (i - 1.5) * width
        bars = ax4.bar(x + offset, values, width, label=cat.replace('_', ' ').title(), alpha=0.8)

    ax4.set_xlabel('Population')
    ax4.set_ylabel('Number of Variants')
    ax4.set_title('D. Variant Enrichment Categories by Population')
    ax4.set_xticks(x)
    ax4.set_xticklabels(pop_summary['population'])
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'{ANALYSIS_DIR}/cc_ratio_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: cc_ratio_analysis.png")

    plt.close()


def overall_summary(df, pop_summary, gene_summary):
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    shared = df[(df['cc_ratio'] > 0) & (df['cc_ratio'] < np.inf)]

    print(f"\n  Total variants analyzed: {len(df):,}")
    print(f"  Shared variants (both case and ctrl): {len(shared):,}")

    n_case_enriched = len(df[df['enrichment'] == 'case_enriched'])
    n_ctrl_enriched = len(df[df['enrichment'] == 'ctrl_enriched'])
    n_case_only = len(df[df['enrichment'] == 'case_only'])
    n_ctrl_only = len(df[df['enrichment'] == 'ctrl_only'])

    print(f"\n  Variant classification:")
    print(f"    Case-only: {n_case_only:,}")
    print(f"    Case-enriched (CC > 1): {n_case_enriched:,}")
    print(f"    Control-enriched (CC < 1): {n_ctrl_enriched:,}")
    print(f"    Control-only: {n_ctrl_only:,}")

    overall_median_cc = shared['cc_ratio'].median()
    overall_mean_log2 = shared['log2_cc_ratio'].mean()

    print(f"\n  Overall CC ratio statistics (shared variants):")
    print(f"    Median CC ratio: {overall_median_cc:.4f}")
    print(f"    Mean log2(CC ratio): {overall_mean_log2:.4f}")

    t_stat, t_p = stats.ttest_1samp(shared['log2_cc_ratio'].dropna(), 0)
    print(f"    t-test (log2CC != 0): t = {t_stat:.2f}, P = {t_p:.2e}")

    print("\n  INTERPRETATION:")
    print("-" * 50)
    if overall_mean_log2 > 0 and t_p < 0.05:
        print("  ✓ Significant CASE enrichment detected")
        print(f"    Rare variants in 86 AD genes are enriched in AD cases")
        print(f"    Mean fold enrichment: {2**overall_mean_log2:.3f}x")
    elif overall_mean_log2 < 0 and t_p < 0.05:
        print("  ✓ Significant CONTROL enrichment detected")
        print(f"    Rare variants in 86 AD genes are depleted in AD cases")
    else:
        print("  No significant enrichment bias detected")

    from scipy.stats import binomtest
    binom_result = binomtest(n_case_enriched, n_case_enriched + n_ctrl_enriched, 0.5)
    print(f"\n  Binomial test (case vs ctrl enriched):")
    print(f"    Case-enriched: {n_case_enriched} ({n_case_enriched/(n_case_enriched+n_ctrl_enriched)*100:.1f}%)")
    print(f"    Control-enriched: {n_ctrl_enriched} ({n_ctrl_enriched/(n_case_enriched+n_ctrl_enriched)*100:.1f}%)")
    print(f"    P-value: {binom_result.pvalue:.2e}")


def main():
    print("=" * 70)
    print("CASE/CONTROL RATIO ANALYSIS FOR 86 AD GENES")
    print("=" * 70)

    df = load_variant_data()

    df = calculate_cc_ratio(df)

    pop_summary = summarize_by_population(df)

    gene_summary = summarize_by_gene(df)

    create_visualizations(df, pop_summary, gene_summary)

    overall_summary(df, pop_summary, gene_summary)

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    df.to_csv(f'{ANALYSIS_DIR}/variant_cc_ratios.csv', index=False)
    print(f"  Saved: variant_cc_ratios.csv")

    pop_summary.to_csv(f'{ANALYSIS_DIR}/cc_ratio_by_population.csv', index=False)
    print(f"  Saved: cc_ratio_by_population.csv")

    gene_summary.to_csv(f'{ANALYSIS_DIR}/cc_ratio_by_gene.csv', index=False)
    print(f"  Saved: cc_ratio_by_gene.csv")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return df, pop_summary, gene_summary


if __name__ == '__main__':
    df, pop_summary, gene_summary = main()
