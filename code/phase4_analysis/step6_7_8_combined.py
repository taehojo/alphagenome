import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('<WORK_DIR>')
OUTPUT_DIR = BASE_DIR / 'analysis/comprehensive_analysis_FULL_1.8M/case_control_86gene'
WORKER_DIR = BASE_DIR / 'worker_results'

MODALITIES = [
    'rna_seq_effect',
    'splice_junctions_effect',
    'splice_sites_effect',
    'splice_site_usage_effect',
    'cage_effect',
    'dnase_effect',
    'chip_histone_effect',
    'chip_tf_effect'
]

AD_GENES = set([
    'GRN', 'SLC2A4RG', 'TREM2', 'PILRA', 'SHARPIN', 'ABCA7', 'ADAM17', 'HS3ST5', 'SNX1', 'SCIMP',
    'ADAMTS1', 'WDR81', 'PICALM', 'PLEKHA1', 'CLU', 'SPDYE3', 'RBCK1', 'ABI3', 'CASS4', 'MYO15A',
    'CR1', 'RIN3', 'CTSH', 'PSEN1', 'HLA-DQA1', 'JAZF1', 'WNT3', 'PRKD3', 'COX7C', 'APP',
    'IL34', 'EPHA1', 'RHOH', 'ABCA1', 'PRDM7', 'RASGEF1C', 'FERMT2', 'PLCG2', 'USP6NL', 'SLC24A4',
    'CASP7', 'SPI1', 'UMAD1', 'MINDY2', 'CTSB', 'BLNK', 'CLNK', 'LILRB2', 'PTK2B', 'KLF16',
    'BIN1', 'IDUA', 'TMEM106B', 'PSEN2', 'ADAM10', 'WDR12', 'MS4A4A', 'TNIP1', 'EED', 'TSPAN14',
    'ANK3', 'EPDR1', 'SPPL2A', 'TREML2', 'SORL1', 'FOXF1', 'NCK2', 'APH1B', 'ACE', 'CD2AP',
    'TSPOAP1', 'MS4A6A', 'INPP5D', 'SEC61G', 'ICA1', 'TPCN1', 'MME', 'SORT1', 'ANKH', 'DOC2A',
    'MAF', 'APBB3', 'SIGLEC11', 'BCKDK', 'UNC5CL'
])

CELL_TYPES = {
    'Neuron': ['APP', 'PSEN1', 'PSEN2', 'MAPT', 'BIN1', 'CLU', 'SORL1', 'ANK3', 'PTK2B', 'ADAM10',
               'APH1B', 'FERMT2', 'SLC24A4', 'CASS4', 'PICALM', 'CD2AP', 'EPHA1'],
    'Microglia': ['TREM2', 'PLCG2', 'ABI3', 'INPP5D', 'SPI1', 'CD33', 'MS4A4A', 'MS4A6A',
                  'CR1', 'PILRA', 'LILRB2', 'TREML2', 'SCIMP', 'CLNK', 'BLNK'],
    'Astrocyte': ['CLU', 'APOE', 'ABCA7', 'ABCA1', 'GRN', 'CTSH', 'CTSB'],
    'Oligodendrocyte': ['MAF', 'ANK3'],
    'Ubiquitous': ['ADAM17', 'JAZF1', 'UMAD1', 'RHOH', 'RASGEF1C', 'HS3ST5', 'SNX1',
                   'PLEKHA1', 'WDR81', 'WDR12', 'MINDY2', 'TSPAN14', 'EPDR1', 'NCK2',
                   'TMEM106B', 'SPPL2A', 'EED', 'ACE', 'TPCN1', 'MME', 'ICA1', 'SORT1',
                   'ANKH', 'FOXF1', 'USP6NL', 'IDUA', 'KLF16', 'COX7C', 'SPDYE3',
                   'RBCK1', 'SHARPIN', 'TNIP1', 'TSPOAP1', 'CASP7', 'PRKD3', 'WNT3',
                   'HLA-DQA1', 'MYO15A', 'PRDM7', 'RIN3', 'ADAMTS1', 'IL34', 'DOC2A',
                   'APBB3', 'SIGLEC11', 'BCKDK', 'UNC5CL', 'SEC61G']
}

def calculate_gene_burden_streaming():
    print("=" * 80)
    print("Calculating gene-level burden from AlphaGenome results (streaming)...")
    print("=" * 80)

    gene_stats = defaultdict(lambda: {
        'n_variants': 0,
        **{f'{mod}_sum': 0.0 for mod in MODALITIES},
        **{f'{mod}_values': [] for mod in MODALITIES}
    })

    for i in range(5):
        pkl_file = WORKER_DIR / f'results_{i:03d}.pkl'
        print(f"Processing {pkl_file.name}...")

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        for variant in data:
            gene = variant['gene_name']
            gene_stats[gene]['n_variants'] += 1

            for mod in MODALITIES:
                if mod in variant:
                    val = variant[mod]
                    gene_stats[gene][f'{mod}_sum'] += val
                    gene_stats[gene][f'{mod}_values'].append(val)

    print("\nConverting to DataFrame...")
    records = []
    for gene, stats in gene_stats.items():
        if stats['n_variants'] < 5:
            continue

        record = {
            'gene': gene,
            'n_variants': stats['n_variants'],
            'is_ad_gene': gene in AD_GENES
        }

        for mod in MODALITIES:
            vals = stats[f'{mod}_values']
            if vals:
                record[f'{mod}_sum'] = stats[f'{mod}_sum']
                record[f'{mod}_mean'] = np.mean(vals)
                record[f'{mod}_median'] = np.median(vals)

        records.append(record)

    burden_df = pd.DataFrame(records)
    print(f"Total genes: {len(burden_df)}")
    print(f"AD genes: {burden_df['is_ad_gene'].sum()}")

    return burden_df

def step6_gene_burden(burden_df, cc_df):
    print("\n" + "=" * 80)
    print("STEP 6: Gene-Level Burden Analysis")
    print("=" * 80)

    print("\n--- Step 6a: AD vs Non-AD Genes ---")
    ad_burden = burden_df[burden_df['is_ad_gene']]
    nonad_burden = burden_df[~burden_df['is_ad_gene']]

    print(f"AD genes: {len(ad_burden)}")
    print(f"Non-AD genes: {len(nonad_burden)}")

    results_6a = []
    print("\n" + "-" * 100)
    print(f"{'Modality':<25} {'AD Mean':>12} {'Non-AD Mean':>12} {'Ratio':>10} {'P-value':>12}")
    print("-" * 100)

    for mod in MODALITIES:
        mean_col = f'{mod}_mean'
        ad_vals = ad_burden[mean_col].dropna()
        nonad_vals = nonad_burden[mean_col].dropna()

        if len(ad_vals) > 0 and len(nonad_vals) > 0:
            stat, pval = stats.mannwhitneyu(ad_vals, nonad_vals, alternative='two-sided')
            ad_mean = ad_vals.mean()
            nonad_mean = nonad_vals.mean()
            ratio = ad_mean / nonad_mean if nonad_mean > 0 else np.nan

            pval_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
            print(f"{mod.replace('_effect', ''):<25} {ad_mean:>12.4f} {nonad_mean:>12.4f} {ratio:>10.4f} {pval_str:>12}")

            results_6a.append({
                'Modality': mod.replace('_effect', ''),
                'AD_n': len(ad_vals), 'NonAD_n': len(nonad_vals),
                'AD_mean': ad_mean, 'NonAD_mean': nonad_mean,
                'Ratio': ratio, 'P_value': pval
            })

    print("-" * 100)
    pd.DataFrame(results_6a).to_csv(OUTPUT_DIR / 'table3_ad_vs_nonad_burden.csv', index=False)

    print("\n--- Step 6b: Case-enriched vs Control-enriched Variants ---")

    cc_finite = cc_df[np.isfinite(cc_df['cc_ratio'])].copy()
    case_enriched = cc_finite[cc_finite['cc_ratio'] > 1]
    ctrl_enriched = cc_finite[cc_finite['cc_ratio'] < 1]

    print(f"Case-enriched variants: {len(case_enriched):,}")
    print(f"Control-enriched variants: {len(ctrl_enriched):,}")

    results_6b = []
    print("\n" + "-" * 100)
    print(f"{'Modality':<25} {'Case Mean':>12} {'Ctrl Mean':>12} {'Ratio':>10} {'P-value':>12}")
    print("-" * 100)

    for mod in MODALITIES:
        case_vals = case_enriched[mod].dropna()
        ctrl_vals = ctrl_enriched[mod].dropna()

        if len(case_vals) > 0 and len(ctrl_vals) > 0:
            stat, pval = stats.mannwhitneyu(case_vals, ctrl_vals, alternative='two-sided')
            case_mean = case_vals.mean()
            ctrl_mean = ctrl_vals.mean()
            ratio = case_mean / ctrl_mean if ctrl_mean > 0 else np.nan

            pval_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
            print(f"{mod.replace('_effect', ''):<25} {case_mean:>12.4f} {ctrl_mean:>12.4f} {ratio:>10.4f} {pval_str:>12}")

            results_6b.append({
                'Modality': mod.replace('_effect', ''),
                'Case_n': len(case_vals), 'Ctrl_n': len(ctrl_vals),
                'Case_mean': case_mean, 'Ctrl_mean': ctrl_mean,
                'Ratio': ratio, 'P_value': pval
            })

    print("-" * 100)
    pd.DataFrame(results_6b).to_csv(OUTPUT_DIR / 'case_vs_ctrl_enriched_variants.csv', index=False)

    return results_6a, results_6b

def step7_matched_control(burden_df):
    print("\n" + "=" * 80)
    print("STEP 7: Matched Control Analysis (1:5 matching)")
    print("=" * 80)

    ad_genes = burden_df[burden_df['is_ad_gene']].copy()
    nonad_genes = burden_df[~burden_df['is_ad_gene']].copy()

    print(f"AD genes for matching: {len(ad_genes)}")
    print(f"Non-AD genes pool: {len(nonad_genes)}")

    matched_pairs = []

    for _, ad_row in ad_genes.iterrows():
        ad_gene = ad_row['gene']
        ad_nvars = ad_row['n_variants']

        lower = ad_nvars * 0.8
        upper = ad_nvars * 1.2

        candidates = nonad_genes[
            (nonad_genes['n_variants'] >= lower) &
            (nonad_genes['n_variants'] <= upper)
        ]

        if len(candidates) >= 5:
            controls = candidates.sample(n=5, random_state=42)

            for mod in MODALITIES:
                mean_col = f'{mod}_mean'
                if mean_col in ad_row and pd.notna(ad_row[mean_col]):
                    ad_val = ad_row[mean_col]
                    ctrl_vals = controls[mean_col].dropna().values

                    for ctrl_val in ctrl_vals:
                        matched_pairs.append({
                            'ad_gene': ad_gene,
                            'modality': mod,
                            'ad_value': ad_val,
                            'ctrl_value': ctrl_val,
                            'ad_n_variants': ad_nvars
                        })

    if not matched_pairs:
        print("No matched pairs found!")
        return []

    matched_df = pd.DataFrame(matched_pairs)
    print(f"Total matched pairs: {len(matched_df)}")

    results_7 = []
    print("\n" + "-" * 100)
    print(f"{'Modality':<25} {'N pairs':>10} {'AD Mean':>12} {'Ctrl Mean':>12} {'P-value':>12}")
    print("-" * 100)

    for mod in MODALITIES:
        mod_data = matched_df[matched_df['modality'] == mod]
        if len(mod_data) < 10:
            continue

        ad_vals = mod_data['ad_value'].values
        ctrl_vals = mod_data['ctrl_value'].values

        try:
            stat, pval = stats.wilcoxon(ad_vals, ctrl_vals, alternative='two-sided')
        except:
            pval = 1.0

        pval_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
        print(f"{mod.replace('_effect', ''):<25} {len(mod_data):>10} {ad_vals.mean():>12.4f} {ctrl_vals.mean():>12.4f} {pval_str:>12}")

        results_7.append({
            'Modality': mod.replace('_effect', ''),
            'N_pairs': len(mod_data),
            'AD_mean': ad_vals.mean(),
            'Ctrl_mean': ctrl_vals.mean(),
            'Ratio': ad_vals.mean() / ctrl_vals.mean() if ctrl_vals.mean() > 0 else np.nan,
            'Wilcoxon_p': pval
        })

    print("-" * 100)
    pd.DataFrame(results_7).to_csv(OUTPUT_DIR / 'table4_matched_control.csv', index=False)

    return results_7

def step8_cell_type(cc_df):
    print("\n" + "=" * 80)
    print("STEP 8: Cell Type Analysis")
    print("=" * 80)

    gene_celltype = {}
    for celltype, genes in CELL_TYPES.items():
        for gene in genes:
            if gene not in gene_celltype:
                gene_celltype[gene] = celltype

    cc_df['cell_type'] = cc_df['gene_name'].map(gene_celltype)

    ct_df = cc_df[cc_df['cell_type'].notna() & np.isfinite(cc_df['cc_ratio'])].copy()
    print(f"Variants with cell type annotation: {len(ct_df):,}")

    print("\nVariant counts by cell type:")
    print(ct_df['cell_type'].value_counts())

    results_8 = []
    cell_types = ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']

    print("\n" + "-" * 100)
    print(f"{'Cell Type':<15} {'N variants':>12} {'Mean CC':>10} {'Mean RNA':>12} {'Mean CAGE':>12} {'Mean ChIP-H':>12}")
    print("-" * 100)

    for ct in cell_types:
        ct_subset = ct_df[ct_df['cell_type'] == ct]
        if len(ct_subset) < 10:
            continue

        result = {
            'Cell_Type': ct,
            'N_variants': len(ct_subset),
            'N_genes': ct_subset['gene_name'].nunique(),
            'Mean_CC_ratio': ct_subset['cc_ratio'].mean(),
            'Median_CC_ratio': ct_subset['cc_ratio'].median()
        }

        for mod in MODALITIES:
            result[f'{mod}_mean'] = ct_subset[mod].mean()

        print(f"{ct:<15} {len(ct_subset):>12,} {ct_subset['cc_ratio'].mean():>10.3f} "
              f"{ct_subset['rna_seq_effect'].mean():>12.4f} {ct_subset['cage_effect'].mean():>12.2f} "
              f"{ct_subset['chip_histone_effect'].mean():>12.2f}")

        results_8.append(result)

    print("-" * 100)

    print("\n--- Pairwise Cell Type Comparisons (CC ratio) ---")
    for i, ct1 in enumerate(cell_types):
        for ct2 in cell_types[i+1:]:
            vals1 = ct_df[ct_df['cell_type'] == ct1]['cc_ratio']
            vals2 = ct_df[ct_df['cell_type'] == ct2]['cc_ratio']

            if len(vals1) > 10 and len(vals2) > 10:
                stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                print(f"{ct1} vs {ct2}: p = {pval:.2e} {sig}")

    pd.DataFrame(results_8).to_csv(OUTPUT_DIR / 'cell_type_analysis.csv', index=False)

    return results_8

def create_combined_visualization(results_6a, results_6b, results_7, results_8, cc_df):
    print("\n" + "=" * 80)
    print("Creating Combined Visualization...")
    print("=" * 80)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax1 = axes[0, 0]
    if results_6a:
        df = pd.DataFrame(results_6a)
        x = np.arange(len(df))
        width = 0.35
        ax1.bar(x - width/2, df['AD_mean'], width, label='AD genes', color='red', alpha=0.7)
        ax1.bar(x + width/2, df['NonAD_mean'], width, label='Non-AD genes', color='blue', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', '\n') for m in df['Modality']], fontsize=7, rotation=45, ha='right')
        ax1.set_ylabel('Mean Effect')
        ax1.set_title('A. Step 6a: AD vs Non-AD Gene Burden')
        ax1.legend()

    ax2 = axes[0, 1]
    if results_6b:
        df = pd.DataFrame(results_6b)
        x = np.arange(len(df))
        ax2.bar(x - width/2, df['Case_mean'], width, label='Case-enriched', color='red', alpha=0.7)
        ax2.bar(x + width/2, df['Ctrl_mean'], width, label='Ctrl-enriched', color='blue', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', '\n') for m in df['Modality']], fontsize=7, rotation=45, ha='right')
        ax2.set_ylabel('Mean Effect')
        ax2.set_title('B. Step 6b: Case vs Ctrl Enriched Variants')
        ax2.legend()

    ax3 = axes[0, 2]
    if results_7:
        df = pd.DataFrame(results_7)
        x = np.arange(len(df))
        ax3.bar(x - width/2, df['AD_mean'], width, label='AD genes', color='red', alpha=0.7)
        ax3.bar(x + width/2, df['Ctrl_mean'], width, label='Matched controls', color='blue', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('_', '\n') for m in df['Modality']], fontsize=7, rotation=45, ha='right')
        ax3.set_ylabel('Mean Effect')
        ax3.set_title('C. Step 7: Matched Control Analysis')
        ax3.legend()

    ax4 = axes[1, 0]
    if results_8:
        df = pd.DataFrame(results_8)
        colors = {'Neuron': 'red', 'Microglia': 'green', 'Astrocyte': 'blue', 'Ubiquitous': 'gray'}
        bars = ax4.bar(df['Cell_Type'], df['Mean_CC_ratio'], color=[colors.get(ct, 'gray') for ct in df['Cell_Type']])
        ax4.axhline(y=1, color='black', linestyle='--')
        ax4.set_ylabel('Mean CC Ratio')
        ax4.set_title('D. Step 8: CC Ratio by Cell Type')

    ax5 = axes[1, 1]
    if results_8:
        df = pd.DataFrame(results_8)
        x = np.arange(len(df))
        width = 0.2
        for i, mod in enumerate(['rna_seq_effect', 'cage_effect', 'chip_histone_effect']):
            vals = df[f'{mod}_mean'].values
            ax5.bar(x + i*width, vals, width, label=mod.replace('_effect', ''))
        ax5.set_xticks(x + width)
        ax5.set_xticklabels(df['Cell_Type'])
        ax5.set_ylabel('Mean Effect')
        ax5.set_title('E. Key Modalities by Cell Type')
        ax5.legend(fontsize=8)

    ax6 = axes[1, 2]
    if results_6a and results_6b:
        df_6a = pd.DataFrame(results_6a)
        df_6b = pd.DataFrame(results_6b)
        x = np.arange(len(df_6a))

        ax6.bar(x - width/2, df_6a['Ratio'], width, label='AD/Non-AD', color='green', alpha=0.7)
        ax6.bar(x + width/2, df_6b['Ratio'], width, label='Case/Ctrl', color='purple', alpha=0.7)
        ax6.axhline(y=1, color='black', linestyle='--')
        ax6.set_xticks(x)
        ax6.set_xticklabels([m.replace('_', '\n') for m in df_6a['Modality']], fontsize=7, rotation=45, ha='right')
        ax6.set_ylabel('Effect Ratio')
        ax6.set_title('F. Summary: Effect Ratios by Comparison')
        ax6.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'steps_6_7_8_combined.png', dpi=150, bbox_inches='tight')
    print(f"Saved: steps_6_7_8_combined.png")

def main():
    print("=" * 80)
    print("ADSP 86-Gene Analysis: Steps 6, 7, 8")
    print("=" * 80)

    print("\nLoading case-control data with AlphaGenome effects...")
    cc_df = pd.read_csv(OUTPUT_DIR / 'variant_cc_with_alphgenome.csv')
    cc_df = cc_df[cc_df['ag_match_type'] != ''].copy()
    print(f"Loaded {len(cc_df):,} variants")

    burden_df = calculate_gene_burden_streaming()

    results_6a, results_6b = step6_gene_burden(burden_df, cc_df)

    results_7 = step7_matched_control(burden_df)

    results_8 = step8_cell_type(cc_df)

    create_combined_visualization(results_6a, results_6b, results_7, results_8, cc_df)

    print("\n" + "=" * 80)
    print("STEPS 6, 7, 8 COMPLETE")
    print("=" * 80)

    print("\nOutput files:")
    print("  - table3_ad_vs_nonad_burden.csv")
    print("  - case_vs_ctrl_enriched_variants.csv")
    print("  - table4_matched_control.csv")
    print("  - cell_type_analysis.csv")
    print("  - steps_6_7_8_combined.png")

if __name__ == '__main__':
    main()
