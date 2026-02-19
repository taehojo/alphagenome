import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '<WORK_DIR>/data/variant_cc_with_alphgenome.csv'
OUT_DIR = '<WORK_DIR>/results/continuous_analysis'

MODALITIES = ['rna_seq_effect', 'cage_effect', 'chip_histone_effect', 'dnase_effect']
MODALITY_LABELS = {
    'rna_seq_effect': 'RNA-seq',
    'cage_effect': 'CAGE',
    'chip_histone_effect': 'ChIP-histone',
    'dnase_effect': 'DNase'
}


def load_and_filter_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw data: {len(df):,} rows")

    df['total_AC'] = df['case_AC'] + df['ctrl_AC']
    df = df[df['total_AC'] >= 3].copy()
    print(f"  After AC>=3 filter: {len(df):,} rows")

    df = df.sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')
    print(f"  After deduplication: {len(df):,} unique variants")

    return df


def sliding_threshold_analysis(df, modality, thresholds=range(5, 96)):
    df_valid = df[df[modality].notna()].copy()
    results = []

    for t in thresholds:
        cutoff = df_valid[modality].quantile(t / 100)
        high = df_valid[df_valid[modality] >= cutoff]
        low = df_valid[df_valid[modality] < cutoff]

        if len(high) >= 10 and len(low) >= 10:
            ir = high['cc_ratio'].median() / low['cc_ratio'].median()

            pct_high = (high['cc_ratio'] > 1).mean() * 100
            pct_low = (low['cc_ratio'] > 1).mean() * 100

            stat, mw_p = stats.mannwhitneyu(
                high['cc_ratio'], low['cc_ratio'], alternative='two-sided'
            )

            results.append({
                'threshold_pct': t,
                'n_high': len(high),
                'n_low': len(low),
                'cutoff_value': cutoff,
                'median_cc_high': high['cc_ratio'].median(),
                'median_cc_low': low['cc_ratio'].median(),
                'interaction_ratio': ir,
                'pct_case_high': pct_high,
                'pct_case_low': pct_low,
                'difference_pct': pct_high - pct_low,
                'mannwhitney_p': mw_p
            })

    return pd.DataFrame(results)


def decile_analysis(df, modality):
    df_valid = df[df[modality].notna()].copy()

    try:
        df_valid['decile'] = pd.qcut(df_valid[modality], 10, labels=False, duplicates='drop') + 1
        n_deciles = df_valid['decile'].nunique()
    except ValueError:
        df_valid['rank_pct'] = df_valid[modality].rank(pct=True)
        df_valid['decile'] = np.ceil(df_valid['rank_pct'] * 10).astype(int)
        df_valid.loc[df_valid['decile'] == 0, 'decile'] = 1
        n_deciles = 10

    results = []
    for dec in sorted(df_valid['decile'].unique()):
        subset = df_valid[df_valid['decile'] == dec]

        n_bootstrap = 1000
        bootstrap_pcts = []
        for _ in range(n_bootstrap):
            sample = subset['cc_ratio'].sample(n=len(subset), replace=True)
            bootstrap_pcts.append((sample > 1).mean() * 100)

        ci_low = np.percentile(bootstrap_pcts, 2.5)
        ci_high = np.percentile(bootstrap_pcts, 97.5)

        results.append({
            'decile': int(dec),
            'n_variants': len(subset),
            'median_cc_ratio': subset['cc_ratio'].median(),
            'mean_cc_ratio': subset['cc_ratio'].mean(),
            'pct_case_enriched': (subset['cc_ratio'] > 1).mean() * 100,
            'pct_case_enriched_ci_low': ci_low,
            'pct_case_enriched_ci_high': ci_high,
            'mean_effect_score': subset[modality].mean(),
            'min_effect_score': subset[modality].min(),
            'max_effect_score': subset[modality].max()
        })

    return pd.DataFrame(results)


def correlation_analysis(df, modalities):
    results = []

    for mod in modalities:
        valid = df[df[mod].notna()]
        r, p = stats.spearmanr(valid[mod], valid['cc_ratio'])

        n_bootstrap = 1000
        bootstrap_r = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(valid), size=len(valid), replace=True)
            sample_mod = valid[mod].iloc[idx].values
            sample_cc = valid['cc_ratio'].iloc[idx].values
            r_boot, _ = stats.spearmanr(sample_mod, sample_cc)
            bootstrap_r.append(r_boot)

        ci_low = np.percentile(bootstrap_r, 2.5)
        ci_high = np.percentile(bootstrap_r, 97.5)

        results.append({
            'modality': mod,
            'modality_label': MODALITY_LABELS[mod],
            'n_variants': len(valid),
            'spearman_r': r,
            'p_value': p,
            'ci_low': ci_low,
            'ci_high': ci_high
        })

    return pd.DataFrame(results)


def create_figure(sliding_results, decile_results, corr_results, modalities):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    ax = axes[0, 0]
    for i, mod in enumerate(modalities):
        data = sliding_results[mod]
        ax.plot(data['threshold_pct'], data['interaction_ratio'],
                color=colors[i], linewidth=2, label=MODALITY_LABELS[mod])

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Percentile Threshold (%)', fontsize=12)
    ax.set_ylabel('Interaction Ratio', fontsize=12)
    ax.set_title('A. Interaction Ratio Across Thresholds', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(5, 95)
    ax.grid(True, alpha=0.3)

    y_min, y_max = ax.get_ylim()
    ax.fill_between([5, 95], [1, 1], [y_max, y_max], alpha=0.1, color='green')
    ax.fill_between([5, 95], [y_min, y_min], [1, 1], alpha=0.1, color='red')

    ax = axes[0, 1]
    for i, mod in enumerate(modalities):
        data = sliding_results[mod]
        ax.plot(data['threshold_pct'], data['difference_pct'],
                color=colors[i], linewidth=2, label=MODALITY_LABELS[mod])

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Percentile Threshold (%)', fontsize=12)
    ax.set_ylabel('Difference in % Case-enriched\n(High - Low)', fontsize=12)
    ax.set_title('B. Case-enrichment Difference Across Thresholds', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(5, 95)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for i, mod in enumerate(modalities):
        data = decile_results[mod]
        ax.plot(data['decile'], data['pct_case_enriched'],
                color=colors[i], linewidth=2, marker='o', markersize=6,
                label=MODALITY_LABELS[mod])

        ax.fill_between(data['decile'],
                       data['pct_case_enriched_ci_low'],
                       data['pct_case_enriched_ci_high'],
                       color=colors[i], alpha=0.2)

    ax.set_xlabel('Decile (1=lowest effect, 10=highest effect)', fontsize=12)
    ax.set_ylabel('% Case-enriched Variants', fontsize=12)
    ax.set_title('C. Decile Trend Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]

    y_pos = np.arange(len(modalities))
    bars = ax.barh(y_pos, corr_results['spearman_r'], color=colors, alpha=0.8)

    xerr = np.array([
        corr_results['spearman_r'] - corr_results['ci_low'],
        corr_results['ci_high'] - corr_results['spearman_r']
    ])
    ax.errorbar(corr_results['spearman_r'], y_pos, xerr=xerr,
                fmt='none', color='black', capsize=5)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(corr_results['modality_label'])
    ax.set_xlabel('Spearman Correlation (r)', fontsize=12)
    ax.set_title('D. Effect Score vs CC Ratio Correlation', fontsize=14, fontweight='bold')

    for i, (r, p) in enumerate(zip(corr_results['spearman_r'], corr_results['p_value'])):
        if p < 0.001:
            p_text = f'P < 0.001'
        else:
            p_text = f'P = {p:.3f}'
        ax.text(r + 0.003, i, f'r = {r:.4f}\n{p_text}',
                va='center', fontsize=9)

    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-0.02, max(corr_results['spearman_r']) * 1.5)

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Continuous Threshold Analysis for Interaction Ratio Robustness")
    print("=" * 60)

    df = load_and_filter_data()

    sliding_results = {}
    decile_results = {}

    print("\n" + "=" * 40)
    print("1. SLIDING THRESHOLD ANALYSIS")
    print("=" * 40)

    for mod in MODALITIES:
        result = sliding_threshold_analysis(df, mod)
        sliding_results[mod] = result

        ir_min = result['interaction_ratio'].min()
        ir_max = result['interaction_ratio'].max()
        ir_mean = result['interaction_ratio'].mean()
        pct_above_1 = (result['interaction_ratio'] > 1).mean() * 100

        print(f"\n{MODALITY_LABELS[mod]}:")
        print(f"  IR range: {ir_min:.4f} - {ir_max:.4f}")
        print(f"  IR mean: {ir_mean:.4f}")
        print(f"  % thresholds with IR > 1: {pct_above_1:.1f}%")
        print(f"  Diff range: {result['difference_pct'].min():.2f}% - {result['difference_pct'].max():.2f}%")

    all_sliding = []
    for mod in MODALITIES:
        temp = sliding_results[mod].copy()
        temp['modality'] = mod
        temp['modality_label'] = MODALITY_LABELS[mod]
        all_sliding.append(temp)

    pd.concat(all_sliding).to_csv(f'{OUT_DIR}/sliding_threshold_results.csv', index=False)
    print(f"\nSaved: {OUT_DIR}/sliding_threshold_results.csv")

    print("\n" + "=" * 40)
    print("2. DECILE TREND ANALYSIS")
    print("=" * 40)

    for mod in MODALITIES:
        result = decile_analysis(df, mod)
        decile_results[mod] = result

        r_trend, p_trend = stats.spearmanr(result['decile'], result['pct_case_enriched'])

        print(f"\n{MODALITY_LABELS[mod]}:")
        print(f"  Decile 1 (lowest): {result['pct_case_enriched'].iloc[0]:.1f}% case-enriched")
        print(f"  Decile 10 (highest): {result['pct_case_enriched'].iloc[-1]:.1f}% case-enriched")
        print(f"  Trend: r = {r_trend:.4f}, P = {p_trend:.2e}")

    all_decile = []
    for mod in MODALITIES:
        temp = decile_results[mod].copy()
        temp['modality'] = mod
        temp['modality_label'] = MODALITY_LABELS[mod]
        all_decile.append(temp)

    pd.concat(all_decile).to_csv(f'{OUT_DIR}/decile_trend_results.csv', index=False)
    print(f"\nSaved: {OUT_DIR}/decile_trend_results.csv")

    print("\n" + "=" * 40)
    print("3. SPEARMAN CORRELATION ANALYSIS")
    print("=" * 40)

    corr_results = correlation_analysis(df, MODALITIES)

    for _, row in corr_results.iterrows():
        sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
        print(f"\n{row['modality_label']}:")
        print(f"  N = {row['n_variants']:,}")
        print(f"  Spearman r = {row['spearman_r']:.4f} (95% CI: {row['ci_low']:.4f} - {row['ci_high']:.4f})")
        print(f"  P = {row['p_value']:.2e} {sig}")

    corr_results.to_csv(f'{OUT_DIR}/correlation_results.csv', index=False)
    print(f"\nSaved: {OUT_DIR}/correlation_results.csv")

    print("\n" + "=" * 40)
    print("4. GENERATING SUPPLEMENTARY FIGURE")
    print("=" * 40)

    fig = create_figure(sliding_results, decile_results, corr_results, MODALITIES)
    fig.savefig(f'{OUT_DIR}/FigureS3_continuous_analysis.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{OUT_DIR}/FigureS3_continuous_analysis.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUT_DIR}/FigureS3_continuous_analysis.png")
    print(f"Saved: {OUT_DIR}/FigureS3_continuous_analysis.pdf")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ir_above_1 = True
    for mod in MODALITIES:
        pct_above = (sliding_results[mod]['interaction_ratio'] > 1).mean() * 100
        if pct_above < 90:
            all_ir_above_1 = False

    all_corr_sig = all(corr_results['p_value'] < 0.05)

    print(f"\nThreshold Independence:")
    for mod in MODALITIES:
        pct = (sliding_results[mod]['interaction_ratio'] > 1).mean() * 100
        status = "PASS" if pct >= 90 else "PARTIAL"
        print(f"  {MODALITY_LABELS[mod]}: {pct:.1f}% of thresholds show IR > 1 [{status}]")

    print(f"\nCorrelation Significance:")
    for _, row in corr_results.iterrows():
        status = "PASS" if row['p_value'] < 0.05 else "FAIL"
        print(f"  {row['modality_label']}: P = {row['p_value']:.2e} [{status}]")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
