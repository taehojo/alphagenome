import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('<ANALYSIS_DIR>')

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

def main():
    print("=" * 80)
    print("Step 5: Variant-Level Enrichment Analysis")
    print("=" * 80)

    print("\nLoading variant data with AlphaGenome effects...")
    df = pd.read_csv(OUTPUT_DIR / 'variant_cc_with_alphgenome.csv')

    df = df[df['ag_match_type'] != ''].copy()
    print(f"Total matched variants: {len(df):,}")

    df = df[np.isfinite(df['cc_ratio'])].copy()
    print(f"Variants with finite CC ratio: {len(df):,}")

    results = []

    print("\n" + "=" * 80)
    print("Analyzing each modality: High-effect (top 10%) vs Low-effect (bottom 90%)")
    print("=" * 80)

    for mod in MODALITIES:
        print(f"\n--- {mod} ---")

        mod_df = df[df[mod].notna()].copy()

        if len(mod_df) < 100:
            print(f"  Skipping: insufficient data ({len(mod_df)} variants)")
            continue

        threshold = mod_df[mod].quantile(0.90)
        print(f"  90th percentile threshold: {threshold:.6f}")

        high_effect = mod_df[mod_df[mod] >= threshold].copy()
        low_effect = mod_df[mod_df[mod] < threshold].copy()

        n_high = len(high_effect)
        n_low = len(low_effect)

        print(f"  High-effect (top 10%): {n_high:,} variants")
        print(f"  Low-effect (bottom 90%): {n_low:,} variants")

        high_cc_median = high_effect['cc_ratio'].median()
        low_cc_median = low_effect['cc_ratio'].median()

        high_cc_mean = high_effect['cc_ratio'].mean()
        low_cc_mean = low_effect['cc_ratio'].mean()

        print(f"  High-effect median CC ratio: {high_cc_median:.4f}")
        print(f"  Low-effect median CC ratio: {low_cc_median:.4f}")

        interaction_ratio = high_cc_median / low_cc_median if low_cc_median > 0 else np.nan
        interaction_ratio_mean = high_cc_mean / low_cc_mean if low_cc_mean > 0 else np.nan

        print(f"  Interaction ratio (median): {interaction_ratio:.4f}")

        stat, pvalue = stats.mannwhitneyu(
            high_effect['cc_ratio'].values,
            low_effect['cc_ratio'].values,
            alternative='two-sided'
        )

        print(f"  Mann-Whitney U p-value: {pvalue:.2e}")

        high_case_enriched = ((high_effect['enrichment'] == 'case_enriched') |
                              (high_effect['enrichment'] == 'case_only')).sum()
        low_case_enriched = ((low_effect['enrichment'] == 'case_enriched') |
                             (low_effect['enrichment'] == 'case_only')).sum()

        high_pct_case = 100 * high_case_enriched / n_high if n_high > 0 else 0
        low_pct_case = 100 * low_case_enriched / n_low if n_low > 0 else 0

        print(f"  High-effect % case-enriched: {high_pct_case:.1f}%")
        print(f"  Low-effect % case-enriched: {low_pct_case:.1f}%")

        results.append({
            'Modality': mod.replace('_effect', ''),
            'N_High': n_high,
            'N_Low': n_low,
            'High_CC_median': high_cc_median,
            'Low_CC_median': low_cc_median,
            'High_CC_mean': high_cc_mean,
            'Low_CC_mean': low_cc_mean,
            'Interaction_median': interaction_ratio,
            'Interaction_mean': interaction_ratio_mean,
            'MannWhitney_U': stat,
            'P_value': pvalue,
            'High_pct_case_enriched': high_pct_case,
            'Low_pct_case_enriched': low_pct_case,
            'Threshold_90pct': threshold
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Interaction_median', ascending=False)

    results_df.to_csv(OUTPUT_DIR / 'variant_enrichment_by_modality.csv', index=False)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Variant-Level Enrichment Analysis")
    print("=" * 80)

    print("\n" + "-" * 100)
    print(f"{'Modality':<20} {'N High':>8} {'N Low':>8} {'High C/C':>10} {'Low C/C':>10} {'Interaction':>12} {'P-value':>12}")
    print("-" * 100)

    for _, row in results_df.iterrows():
        pval_str = f"{row['P_value']:.2e}" if row['P_value'] < 0.001 else f"{row['P_value']:.4f}"
        print(f"{row['Modality']:<20} {row['N_High']:>8,} {row['N_Low']:>8,} "
              f"{row['High_CC_median']:>10.4f} {row['Low_CC_median']:>10.4f} "
              f"{row['Interaction_median']:>12.4f} {pval_str:>12}")

    print("-" * 100)

    sig_modalities = results_df[results_df['P_value'] < 0.05]
    print(f"\nStatistically significant modalities (p < 0.05): {len(sig_modalities)}/{len(results_df)}")

    bonferroni = 0.05 / len(results_df)
    sig_bonf = results_df[results_df['P_value'] < bonferroni]
    print(f"Significant after Bonferroni correction (p < {bonferroni:.4f}): {len(sig_bonf)}/{len(results_df)}")

    print("\n" + "=" * 80)
    print("Enrichment Direction Analysis")
    print("=" * 80)

    positive_enrichment = results_df[results_df['Interaction_median'] > 1]
    negative_enrichment = results_df[results_df['Interaction_median'] < 1]

    print(f"\nModalities with High > Low CC ratio (Interaction > 1): {len(positive_enrichment)}")
    for _, row in positive_enrichment.iterrows():
        sig = "***" if row['P_value'] < bonferroni else ("**" if row['P_value'] < 0.01 else ("*" if row['P_value'] < 0.05 else ""))
        print(f"  {row['Modality']}: {row['Interaction_median']:.4f} {sig}")

    print(f"\nModalities with High < Low CC ratio (Interaction < 1): {len(negative_enrichment)}")
    for _, row in negative_enrichment.iterrows():
        sig = "***" if row['P_value'] < bonferroni else ("**" if row['P_value'] < 0.01 else ("*" if row['P_value'] < 0.05 else ""))
        print(f"  {row['Modality']}: {row['Interaction_median']:.4f} {sig}")

    print("\n" + "=" * 80)
    print("Creating Visualization...")
    print("=" * 80)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    modalities = results_df['Modality'].values
    interactions = results_df['Interaction_median'].values
    pvalues = results_df['P_value'].values

    colors = ['#d62728' if p < bonferroni else '#ff7f0e' if p < 0.05 else '#1f77b4' for p in pvalues]
    bars = ax1.barh(modalities, interactions, color=colors)
    ax1.axvline(x=1, color='black', linestyle='--', linewidth=1, label='No enrichment')
    ax1.set_xlabel('Interaction Ratio (High CC / Low CC)')
    ax1.set_title('A. Modality Interaction Ratios\n(Red=Bonferroni sig, Orange=p<0.05)')
    ax1.set_xlim(0.8, max(interactions) * 1.1)

    for i, (inter, pval) in enumerate(zip(interactions, pvalues)):
        sig = "***" if pval < bonferroni else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        ax1.text(inter + 0.01, i, sig, va='center', fontsize=10)

    ax2 = axes[0, 1]
    top_mod = results_df.iloc[0]['Modality'] + '_effect'

    mod_df = df[df[top_mod].notna()].copy()
    threshold = mod_df[top_mod].quantile(0.90)
    high_effect = mod_df[mod_df[top_mod] >= threshold]['cc_ratio']
    low_effect = mod_df[mod_df[top_mod] < threshold]['cc_ratio']

    high_effect_clip = high_effect.clip(upper=5)
    low_effect_clip = low_effect.clip(upper=5)

    ax2.hist(low_effect_clip, bins=50, alpha=0.6, label=f'Low-effect (n={len(low_effect):,})', color='blue')
    ax2.hist(high_effect_clip, bins=50, alpha=0.6, label=f'High-effect (n={len(high_effect):,})', color='red')
    ax2.axvline(x=1, color='black', linestyle='--', linewidth=1)
    ax2.axvline(x=high_effect.median(), color='red', linestyle='-', linewidth=2, label=f'High median={high_effect.median():.2f}')
    ax2.axvline(x=low_effect.median(), color='blue', linestyle='-', linewidth=2, label=f'Low median={low_effect.median():.2f}')
    ax2.set_xlabel('Case/Control Ratio')
    ax2.set_ylabel('Count')
    ax2.set_title(f'B. CC Ratio Distribution: {top_mod.replace("_effect", "")}')
    ax2.legend(fontsize=8)

    ax3 = axes[1, 0]
    x = np.arange(len(modalities))
    width = 0.35

    ax3.bar(x - width/2, results_df['High_CC_median'].values, width, label='High-effect (top 10%)', color='red', alpha=0.7)
    ax3.bar(x + width/2, results_df['Low_CC_median'].values, width, label='Low-effect (bottom 90%)', color='blue', alpha=0.7)
    ax3.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace('_', '\n') for m in modalities], fontsize=8, rotation=45, ha='right')
    ax3.set_ylabel('Median CC Ratio')
    ax3.set_title('C. High vs Low Effect: Median CC Ratios')
    ax3.legend()

    ax4 = axes[1, 1]
    ax4.bar(x - width/2, results_df['High_pct_case_enriched'].values, width, label='High-effect', color='red', alpha=0.7)
    ax4.bar(x + width/2, results_df['Low_pct_case_enriched'].values, width, label='Low-effect', color='blue', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.replace('_', '\n') for m in modalities], fontsize=8, rotation=45, ha='right')
    ax4.set_ylabel('% Case-Enriched Variants')
    ax4.set_title('D. % Case-Enriched by Effect Level')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'variant_enrichment_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {OUTPUT_DIR / 'variant_enrichment_analysis.png'}")

    print("\n" + "=" * 80)
    print("Gene-Level Interaction Analysis (Top Modality)")
    print("=" * 80)

    top_mod_name = results_df.iloc[0]['Modality'] + '_effect'
    print(f"\nUsing top modality: {top_mod_name}")

    gene_interactions = []
    for gene in df['gene_name'].unique():
        gene_df = df[(df['gene_name'] == gene) & (df[top_mod_name].notna())].copy()
        if len(gene_df) < 10:
            continue

        threshold = gene_df[top_mod_name].quantile(0.90)
        high = gene_df[gene_df[top_mod_name] >= threshold]
        low = gene_df[gene_df[top_mod_name] < threshold]

        if len(high) < 2 or len(low) < 2:
            continue

        high_cc = high['cc_ratio'].median()
        low_cc = low['cc_ratio'].median()

        if low_cc > 0:
            interaction = high_cc / low_cc
            gene_interactions.append({
                'gene': gene,
                'n_variants': len(gene_df),
                'n_high': len(high),
                'n_low': len(low),
                'high_cc': high_cc,
                'low_cc': low_cc,
                'interaction': interaction
            })

    gene_int_df = pd.DataFrame(gene_interactions)
    gene_int_df = gene_int_df.sort_values('interaction', ascending=False)

    print("\nTop 20 genes by interaction ratio:")
    print(gene_int_df.head(20).to_string(index=False))

    gene_int_df.to_csv(OUTPUT_DIR / 'gene_interaction_ratios.csv', index=False)

    print("\n" + "=" * 80)
    print("STEP 5 COMPLETE: Variant-Level Enrichment Analysis")
    print("=" * 80)
    print(f"\nKey findings:")
    print(f"  - Analyzed {len(results_df)} modalities")
    print(f"  - {len(sig_modalities)} modalities show significant enrichment (p < 0.05)")
    print(f"  - {len(sig_bonf)} modalities survive Bonferroni correction")
    print(f"  - Top modality: {results_df.iloc[0]['Modality']} (Interaction = {results_df.iloc[0]['Interaction_median']:.4f})")

    print(f"\nOutput files:")
    print(f"  - variant_enrichment_by_modality.csv")
    print(f"  - gene_interaction_ratios.csv")
    print(f"  - variant_enrichment_analysis.png")

if __name__ == '__main__':
    main()
