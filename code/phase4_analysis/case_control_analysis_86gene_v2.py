import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '<WORK_DIR>'
ANALYSIS_DIR = '<ANALYSIS_DIR>'
WORKER_RESULTS_DIR = f'{BASE_DIR}/worker_results'

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

POPULATIONS = ['NHW', 'AA', 'Asian', 'Hispanic']


def load_gene_coordinates():
    print("=" * 70)
    print("1. LOADING GENE COORDINATES")
    print("=" * 70)

    genes_file = f'{BASE_DIR}/protein_coding_genes.tsv'
    genes_df = pd.read_csv(genes_file, sep='\t')

    genes_86 = genes_df[genes_df['gene_name'].isin(ALL_86_GENES)].copy()

    genes_86['chr_num'] = genes_86['chr'].str.replace('chr', '').astype(str)

    print(f"  Found {len(genes_86)}/{len(ALL_86_GENES)} genes with coordinates")

    found_genes = set(genes_86['gene_name'].unique())
    missing_genes = set(ALL_86_GENES) - found_genes
    if missing_genes:
        print(f"  Missing genes: {missing_genes}")

    return genes_86


def load_frequency_data_with_genes(genes_df):
    print("\n" + "=" * 70)
    print("2. LOADING FREQUENCY DATA AND MAPPING TO GENES")
    print("=" * 70)

    all_freq_data = []

    for pop in POPULATIONS:
        print(f"\n  Processing {pop}...")

        bim_file = f'{BASE_DIR}/results/{pop}/{pop}_rare_variants.bim'
        bim_df = pd.read_csv(bim_file, sep='\t', header=None,
                             names=['chr_num', 'variant_id', 'cm', 'pos', 'alt', 'ref'])
        bim_df['chr_num'] = bim_df['chr_num'].astype(str)
        bim_pos_map = dict(zip(bim_df['variant_id'], bim_df['pos']))

        cases_file = f'{ANALYSIS_DIR}/{pop}_cases_freq.afreq'
        cases_df = pd.read_csv(cases_file, sep='\t')
        cases_df = cases_df.rename(columns={
            '#CHROM': 'chr_num',
            'ID': 'variant_id',
            'ALT_FREQS': 'case_AF',
            'OBS_CT': 'case_AN'
        })
        cases_df['chr_num'] = cases_df['chr_num'].astype(str)
        cases_df['case_AC'] = (cases_df['case_AF'] * cases_df['case_AN']).round().astype(int)

        cases_df['pos'] = cases_df['variant_id'].map(bim_pos_map)

        ctrls_file = f'{ANALYSIS_DIR}/{pop}_ctrls_freq.afreq'
        ctrls_df = pd.read_csv(ctrls_file, sep='\t')
        ctrls_df = ctrls_df.rename(columns={
            '#CHROM': 'chr_num',
            'ID': 'variant_id',
            'ALT_FREQS': 'ctrl_AF',
            'OBS_CT': 'ctrl_AN'
        })
        ctrls_df['chr_num'] = ctrls_df['chr_num'].astype(str)
        ctrls_df['ctrl_AC'] = (ctrls_df['ctrl_AF'] * ctrls_df['ctrl_AN']).round().astype(int)
        ctrls_df['pos'] = ctrls_df['variant_id'].map(bim_pos_map)

        merged = pd.merge(
            cases_df[['chr_num', 'variant_id', 'pos', 'REF', 'ALT', 'case_AF', 'case_AN', 'case_AC']],
            ctrls_df[['variant_id', 'ctrl_AF', 'ctrl_AN', 'ctrl_AC']],
            on='variant_id'
        )
        merged['population'] = pop

        print(f"    {pop}: {len(merged):,} total variants")
        print(f"    Cases: {merged['case_AN'].iloc[0]//2} samples, Controls: {merged['ctrl_AN'].iloc[0]//2} samples")

        gene_mapped = []
        for _, gene in genes_df.iterrows():
            gene_variants = merged[
                (merged['chr_num'] == gene['chr_num']) &
                (merged['pos'] >= gene['start']) &
                (merged['pos'] <= gene['end'])
            ].copy()

            if len(gene_variants) > 0:
                gene_variants['gene_name'] = gene['gene_name']
                gene_variants['gene_id'] = gene['gene_id']
                gene_mapped.append(gene_variants)

        if gene_mapped:
            pop_gene_df = pd.concat(gene_mapped, ignore_index=True)
            print(f"    Variants in 86 genes: {len(pop_gene_df):,}")
            print(f"    Genes with variants: {pop_gene_df['gene_name'].nunique()}")
            all_freq_data.append(pop_gene_df)

    combined = pd.concat(all_freq_data, ignore_index=True)
    print(f"\n  Total: {len(combined):,} variant-gene-population combinations")

    return combined


def calculate_case_control_enrichment(df):
    print("\n" + "=" * 70)
    print("3. CASE-CONTROL ENRICHMENT ANALYSIS")
    print("=" * 70)

    df = df.copy()

    df['cc_ratio'] = np.where(
        df['ctrl_AF'] > 0,
        df['case_AF'] / df['ctrl_AF'],
        np.where(df['case_AF'] > 0, np.inf, 1.0)
    )

    df['case_non_AC'] = df['case_AN'] - df['case_AC']
    df['ctrl_non_AC'] = df['ctrl_AN'] - df['ctrl_AC']

    df['OR'] = np.where(
        (df['ctrl_AC'] > 0) & (df['case_non_AC'] > 0),
        (df['case_AC'] * df['ctrl_non_AC']) / (df['ctrl_AC'] * df['case_non_AC']),
        np.nan
    )

    def fisher_test(row):
        try:
            table = [[int(row['case_AC']), int(row['case_non_AC'])],
                     [int(row['ctrl_AC']), int(row['ctrl_non_AC'])]]
            _, p = stats.fisher_exact(table)
            return p
        except:
            return np.nan

    print("  Calculating Fisher's exact test for each variant...")
    df['fisher_p'] = df.apply(fisher_test, axis=1)

    print("\n  Enrichment Summary by Population:")
    print("-" * 70)

    results = []
    for pop in POPULATIONS:
        pop_df = df[df['population'] == pop].copy()

        pop_df = pop_df[(pop_df['case_AC'] > 0) | (pop_df['ctrl_AC'] > 0)]

        n_variants = len(pop_df)
        n_case_only = len(pop_df[(pop_df['case_AC'] > 0) & (pop_df['ctrl_AC'] == 0)])
        n_ctrl_only = len(pop_df[(pop_df['case_AC'] == 0) & (pop_df['ctrl_AC'] > 0)])
        n_shared = len(pop_df[(pop_df['case_AC'] > 0) & (pop_df['ctrl_AC'] > 0)])

        shared = pop_df[(pop_df['case_AC'] > 0) & (pop_df['ctrl_AC'] > 0)]
        mean_or = shared['OR'].median() if len(shared) > 0 else np.nan

        n_enriched = len(shared[shared['OR'] > 1])
        n_depleted = len(shared[shared['OR'] < 1])
        if n_enriched + n_depleted > 0:
            from scipy.stats import binomtest
            binom_result = binomtest(n_enriched, n_enriched + n_depleted, 0.5)
            binom_p = binom_result.pvalue
        else:
            binom_p = np.nan

        results.append({
            'population': pop,
            'n_variants': n_variants,
            'case_only': n_case_only,
            'ctrl_only': n_ctrl_only,
            'shared': n_shared,
            'median_OR': mean_or,
            'n_OR_gt_1': n_enriched,
            'n_OR_lt_1': n_depleted,
            'binom_p': binom_p
        })

        print(f"\n  {pop}:")
        print(f"    Total variants: {n_variants:,}")
        print(f"    Case-only: {n_case_only:,} | Control-only: {n_ctrl_only:,} | Shared: {n_shared:,}")
        print(f"    Median OR (shared): {mean_or:.3f}" if not np.isnan(mean_or) else "    Median OR: N/A")
        print(f"    OR > 1: {n_enriched} | OR < 1: {n_depleted}")
        print(f"    Binomial test P: {binom_p:.4f}" if not np.isnan(binom_p) else "    Binomial test P: N/A")

    results_df = pd.DataFrame(results)

    return df, results_df


def gene_level_burden_analysis(df):
    print("\n" + "=" * 70)
    print("4. GENE-LEVEL BURDEN ANALYSIS")
    print("=" * 70)

    gene_results = []

    for pop in POPULATIONS:
        pop_df = df[df['population'] == pop]

        for gene in ALL_86_GENES:
            gene_df = pop_df[pop_df['gene_name'] == gene]

            if len(gene_df) == 0:
                continue

            total_case_AC = gene_df['case_AC'].sum()
            total_ctrl_AC = gene_df['ctrl_AC'].sum()
            total_case_AN = gene_df['case_AN'].iloc[0] if len(gene_df) > 0 else 0
            total_ctrl_AN = gene_df['ctrl_AN'].iloc[0] if len(gene_df) > 0 else 0

            case_non_AC = total_case_AN - total_case_AC
            ctrl_non_AC = total_ctrl_AN - total_ctrl_AC

            if total_ctrl_AC > 0 and case_non_AC > 0:
                burden_OR = (total_case_AC * ctrl_non_AC) / (total_ctrl_AC * case_non_AC)
            else:
                burden_OR = np.nan

            try:
                table = [[int(total_case_AC), int(case_non_AC)],
                         [int(total_ctrl_AC), int(ctrl_non_AC)]]
                _, burden_p = stats.fisher_exact(table)
            except:
                burden_p = np.nan

            case_burden_freq = total_case_AC / total_case_AN if total_case_AN > 0 else 0
            ctrl_burden_freq = total_ctrl_AC / total_ctrl_AN if total_ctrl_AN > 0 else 0

            gene_results.append({
                'population': pop,
                'gene': gene,
                'gene_source': 'Table1_GWAS' if gene in TABLE1_GENES else 'Table2_HighConf',
                'n_variants': len(gene_df),
                'case_AC': total_case_AC,
                'ctrl_AC': total_ctrl_AC,
                'case_AN': total_case_AN,
                'ctrl_AN': total_ctrl_AN,
                'case_burden_freq': case_burden_freq,
                'ctrl_burden_freq': ctrl_burden_freq,
                'burden_OR': burden_OR,
                'burden_p': burden_p
            })

    gene_df = pd.DataFrame(gene_results)

    print("\n  Gene-level burden summary by population:")
    print("-" * 70)

    for pop in POPULATIONS:
        pop_genes = gene_df[gene_df['population'] == pop]
        sig_genes = pop_genes[pop_genes['burden_p'] < 0.05]

        print(f"\n  {pop}:")
        print(f"    Genes analyzed: {len(pop_genes)}")
        print(f"    Significant genes (P < 0.05): {len(sig_genes)}")

        if len(sig_genes) > 0:
            sig_sorted = sig_genes.sort_values('burden_OR', ascending=False)
            print(f"    Top significant genes:")
            for _, row in sig_sorted.head(5).iterrows():
                direction = "↑" if row['burden_OR'] > 1 else "↓"
                print(f"      {row['gene']}: OR={row['burden_OR']:.2f} {direction}, P={row['burden_p']:.4f}")

    return gene_df


def meta_analysis(gene_burden_df):
    print("\n" + "=" * 70)
    print("5. META-ANALYSIS ACROSS POPULATIONS")
    print("=" * 70)

    meta_results = []

    for gene in ALL_86_GENES:
        gene_data = gene_burden_df[gene_burden_df['gene'] == gene]

        if len(gene_data) == 0:
            continue

        total_case_AC = gene_data['case_AC'].sum()
        total_ctrl_AC = gene_data['ctrl_AC'].sum()
        total_case_AN = gene_data['case_AN'].sum()
        total_ctrl_AN = gene_data['ctrl_AN'].sum()

        case_non_AC = total_case_AN - total_case_AC
        ctrl_non_AC = total_ctrl_AN - total_ctrl_AC

        if total_ctrl_AC > 0 and case_non_AC > 0:
            meta_OR = (total_case_AC * ctrl_non_AC) / (total_ctrl_AC * case_non_AC)
        else:
            meta_OR = np.nan

        try:
            table = [[int(total_case_AC), int(case_non_AC)],
                     [int(total_ctrl_AC), int(ctrl_non_AC)]]
            _, meta_p = stats.fisher_exact(table)
        except:
            meta_p = np.nan

        meta_results.append({
            'gene': gene,
            'gene_source': 'Table1_GWAS' if gene in TABLE1_GENES else 'Table2_HighConf',
            'n_populations': len(gene_data),
            'total_variants': gene_data['n_variants'].sum(),
            'total_case_AC': total_case_AC,
            'total_ctrl_AC': total_ctrl_AC,
            'total_case_AN': total_case_AN,
            'total_ctrl_AN': total_ctrl_AN,
            'meta_OR': meta_OR,
            'meta_p': meta_p
        })

    meta_df = pd.DataFrame(meta_results)
    meta_df = meta_df.sort_values('meta_p')

    print("\n  Meta-analysis results:")
    print("-" * 70)

    sig_genes = meta_df[meta_df['meta_p'] < 0.05]
    print(f"  Genes with P < 0.05: {len(sig_genes)}")

    bonf_threshold = 0.05 / len(meta_df) if len(meta_df) > 0 else 0.05
    bonf_sig = meta_df[meta_df['meta_p'] < bonf_threshold]
    print(f"  Genes passing Bonferroni (P < {bonf_threshold:.6f}): {len(bonf_sig)}")

    print("\n  Top 20 genes by meta-analysis P-value:")
    print("-" * 90)
    print(f"  {'Gene':<12} {'Source':<15} {'Case AC':<10} {'Ctrl AC':<10} {'OR':<8} {'P-value':<12} {'Sig'}")
    print("-" * 90)

    for _, row in meta_df.head(20).iterrows():
        if np.isnan(row['meta_OR']):
            continue
        sig = '***' if row['meta_p'] < bonf_threshold else ('**' if row['meta_p'] < 0.01 else ('*' if row['meta_p'] < 0.05 else ''))
        print(f"  {row['gene']:<12} {row['gene_source']:<15} {int(row['total_case_AC']):<10} {int(row['total_ctrl_AC']):<10} {row['meta_OR']:<8.3f} {row['meta_p']:<12.2e} {sig}")

    return meta_df


def create_visualizations(df, gene_burden_df, meta_df):
    print("\n" + "=" * 70)
    print("6. CREATING VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    meta_df_plot = meta_df.dropna(subset=['meta_OR'])
    meta_df_plot = meta_df_plot[meta_df_plot['meta_OR'] < 10]

    colors = ['#E74C3C' if row['meta_p'] < 0.05 else '#3498DB'
              for _, row in meta_df_plot.iterrows()]

    ax1.scatter(range(len(meta_df_plot)), meta_df_plot['meta_OR'], c=colors, alpha=0.7, s=50)
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Gene (sorted by P-value)')
    ax1.set_ylabel('Meta-analysis Odds Ratio')
    ax1.set_title('A. Gene Burden Odds Ratios (Meta-analysis)')

    for i, (_, row) in enumerate(meta_df_plot.head(10).iterrows()):
        if row['meta_p'] < 0.05:
            ax1.annotate(row['gene'], (i, row['meta_OR']), fontsize=8,
                        xytext=(5, 5), textcoords='offset points')

    ax2 = axes[0, 1]
    meta_df_plot['-log10(p)'] = -np.log10(meta_df_plot['meta_p'].clip(lower=1e-10))

    table1_mask = meta_df_plot['gene_source'] == 'Table1_GWAS'
    ax2.scatter(range(len(meta_df_plot[table1_mask])),
                meta_df_plot[table1_mask]['-log10(p)'],
                c='#3498DB', alpha=0.7, label='Table 1 (GWAS)', s=50)

    n_table1 = len(meta_df_plot[table1_mask])
    ax2.scatter(range(n_table1, n_table1 + len(meta_df_plot[~table1_mask])),
                meta_df_plot[~table1_mask]['-log10(p)'].values,
                c='#E74C3C', alpha=0.7, label='Table 2 (High-Conf)', s=50)

    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='P=0.05')
    bonf = 0.05/len(meta_df_plot) if len(meta_df_plot) > 0 else 0.05
    ax2.axhline(y=-np.log10(bonf), color='darkred', linestyle='--', label='Bonferroni')
    ax2.set_xlabel('Gene')
    ax2.set_ylabel('-log10(P-value)')
    ax2.set_title('B. Gene-level Association Significance')
    ax2.legend(loc='upper right')

    ax3 = axes[1, 0]

    gene_agg = df.groupby('gene_name').agg({
        'case_AF': 'mean',
        'ctrl_AF': 'mean'
    }).reset_index()

    ax3.scatter(gene_agg['ctrl_AF'] * 1000, gene_agg['case_AF'] * 1000,
                alpha=0.6, s=50, c='#2ECC71')

    max_val = max(gene_agg['case_AF'].max(), gene_agg['ctrl_AF'].max()) * 1000
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    ax3.set_xlabel('Control Mean AF (per 1000)')
    ax3.set_ylabel('Case Mean AF (per 1000)')
    ax3.set_title('C. Case vs Control Allele Frequency by Gene')
    ax3.legend()

    ax4 = axes[1, 1]

    top_genes = meta_df.head(20)['gene'].tolist()
    pivot_data = gene_burden_df[gene_burden_df['gene'].isin(top_genes)].pivot(
        index='gene', columns='population', values='burden_OR'
    )

    if len(pivot_data) > 0:
        pivot_data = pivot_data.reindex([g for g in top_genes if g in pivot_data.index])
        sns.heatmap(pivot_data, ax=ax4, cmap='RdBu_r', center=1,
                    vmin=0.5, vmax=2, annot=True, fmt='.2f', cbar_kws={'label': 'OR'})
        ax4.set_title('D. Top 20 Genes: Population-specific ORs')
        ax4.set_xlabel('Population')
        ax4.set_ylabel('Gene')
    else:
        ax4.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax4.set_title('D. Top 20 Genes: Population-specific ORs')

    plt.tight_layout()
    plt.savefig(f'{ANALYSIS_DIR}/case_control_86gene_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: case_control_86gene_analysis.png")

    plt.close()


def generate_summary_report(pop_results, gene_burden_df, meta_df):
    print("\n" + "=" * 70)
    print("7. FINAL SUMMARY REPORT")
    print("=" * 70)

    print("\n  STUDY SUMMARY")
    print("-" * 70)
    print(f"  Target genes: {len(ALL_86_GENES)} (76 GWAS + 10 High-confidence)")
    print(f"  Populations analyzed: {len(POPULATIONS)}")

    print("\n  Sample sizes by population:")
    sample_sizes = {
        'NHW': (3245, 5388),
        'AA': (1114, 3506),
        'Asian': (39, 2560),
        'Hispanic': (1898, 6845)
    }
    total_cases = 0
    total_controls = 0
    for pop, (cases, ctrls) in sample_sizes.items():
        print(f"    {pop}: {cases:,} cases, {ctrls:,} controls")
        total_cases += cases
        total_controls += ctrls
    print(f"    Total: {total_cases:,} cases, {total_controls:,} controls")

    print("\n  KEY FINDINGS")
    print("-" * 70)

    if len(meta_df) > 0:
        sig_genes = meta_df[meta_df['meta_p'] < 0.05]
        bonf_threshold = 0.05 / len(meta_df)
        bonf_sig = meta_df[meta_df['meta_p'] < bonf_threshold]

        print(f"  1. Nominally significant genes (P < 0.05): {len(sig_genes)}")
        print(f"  2. Bonferroni-significant genes: {len(bonf_sig)}")

        risk_genes = sig_genes[sig_genes['meta_OR'] > 1]
        prot_genes = sig_genes[sig_genes['meta_OR'] < 1]
        print(f"  3. Risk genes (OR > 1, P < 0.05): {len(risk_genes)}")
        print(f"  4. Protective genes (OR < 1, P < 0.05): {len(prot_genes)}")

        if len(risk_genes) > 0:
            print("\n  Top Risk Genes:")
            for _, row in risk_genes.sort_values('meta_p').head(5).iterrows():
                print(f"    {row['gene']}: OR = {row['meta_OR']:.2f}, P = {row['meta_p']:.2e}")

        if len(prot_genes) > 0:
            print("\n  Top Protective Genes:")
            for _, row in prot_genes.sort_values('meta_p').head(5).iterrows():
                print(f"    {row['gene']}: OR = {row['meta_OR']:.2f}, P = {row['meta_p']:.2e}")

    return


def main():
    print("=" * 70)
    print("ADSP 86-GENE RARE VARIANT CASE-CONTROL ANALYSIS V2")
    print("=" * 70)

    genes_df = load_gene_coordinates()

    freq_df = load_frequency_data_with_genes(genes_df)

    enriched_df, pop_results = calculate_case_control_enrichment(freq_df)

    gene_burden_df = gene_level_burden_analysis(enriched_df)

    meta_df = meta_analysis(gene_burden_df)

    create_visualizations(enriched_df, gene_burden_df, meta_df)

    generate_summary_report(pop_results, gene_burden_df, meta_df)

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    enriched_df.to_csv(f'{ANALYSIS_DIR}/variant_level_case_control.csv', index=False)
    print(f"  Saved: variant_level_case_control.csv")

    gene_burden_df.to_csv(f'{ANALYSIS_DIR}/gene_burden_by_population.csv', index=False)
    print(f"  Saved: gene_burden_by_population.csv")

    meta_df.to_csv(f'{ANALYSIS_DIR}/meta_analysis_results.csv', index=False)
    print(f"  Saved: meta_analysis_results.csv")

    pop_results.to_csv(f'{ANALYSIS_DIR}/population_enrichment_summary.csv', index=False)
    print(f"  Saved: population_enrichment_summary.csv")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return enriched_df, gene_burden_df, meta_df


if __name__ == '__main__':
    enriched_df, gene_burden_df, meta_df = main()
