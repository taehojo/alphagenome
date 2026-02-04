import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'rna_seq': '#E64B35',
    'cage': '#4DBBD5',
    'chip_histone': '#00A087',
    'dnase': '#F39B7F',
    'significant': '#3C5488',
    'ns': '#B09C85',
    'case': '#E64B35',
    'ctrl': '#4DBBD5',
}

MODALITY_NAMES = {
    'rna_seq_effect': 'RNA-seq',
    'cage_effect': 'CAGE',
    'chip_histone_effect': 'ChIP-histone',
    'dnase_effect': 'DNase',
}

def load_data():
    df = pd.read_csv('<DATA_DIR>/variant_cc_with_alphgenome.csv')
    df['total_AC'] = df['case_AC'] + df['ctrl_AC']
    df = df[df['total_AC'] >= 3]
    df = df.sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')
    df['case_enriched'] = df['enrichment'].isin(['case_enriched', 'case_only'])
    return df

def calculate_or_ci(high_case, high_ctrl, low_case, low_ctrl):
    table = [[high_case, high_ctrl], [low_case, low_ctrl]]
    odds_ratio, p_value = stats.fisher_exact(table)

    a, b, c, d = high_case, high_ctrl, low_case, low_ctrl
    if a == 0 or b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    log_or = np.log(odds_ratio)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lower = np.exp(log_or - 1.96 * se)
    ci_upper = np.exp(log_or + 1.96 * se)

    return odds_ratio, ci_lower, ci_upper, p_value

def panel_a_mean_difference(ax, df):

    modalities = ['rna_seq_effect', 'cage_effect', 'chip_histone_effect', 'dnase_effect']
    results = []

    for mod in modalities:
        case_enr = df[df['case_enriched']][mod].dropna()
        ctrl_enr = df[~df['case_enriched']][mod].dropna()

        pct_diff = (case_enr.mean() - ctrl_enr.mean()) / ctrl_enr.mean() * 100

        np.random.seed(42)
        pct_diffs = []
        for _ in range(1000):
            c_sample = np.random.choice(case_enr, size=len(case_enr), replace=True)
            t_sample = np.random.choice(ctrl_enr, size=len(ctrl_enr), replace=True)
            t_mean = t_sample.mean()
            if t_mean != 0:
                pct_diffs.append((c_sample.mean() - t_mean) / t_mean * 100)
        ci_lower, ci_upper = np.percentile(pct_diffs, [2.5, 97.5])

        t_stat, p_val = stats.ttest_ind(case_enr, ctrl_enr)

        results.append({
            'modality': MODALITY_NAMES[mod],
            'pct_diff': pct_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_val,
            'color': COLORS[mod.replace('_effect', '')]
        })
        print(f"  Panel (a) {MODALITY_NAMES[mod]}: {pct_diff:+.1f}%, P={p_val:.2e}")

    results_df = pd.DataFrame(results)

    y_pos = np.arange(len(results_df))

    for i, row in results_df.iterrows():
        color = row['color'] if row['p_value'] < 0.05 else COLORS['ns']
        ax.errorbar(row['pct_diff'], i, xerr=[[row['pct_diff']-row['ci_lower']],
                   [row['ci_upper']-row['pct_diff']]],
                   fmt='o', color=color, capsize=3, capthick=1, markersize=6)

        if row['p_value'] < 0.001:
            p_text = '***'
        elif row['p_value'] < 0.01:
            p_text = '**'
        elif row['p_value'] < 0.05:
            p_text = '*'
        else:
            p_text = 'ns'
        ax.text(row['ci_upper'] + 0.5, i, p_text, va='center', fontsize=7)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['modality'])
    ax.set_xlabel('Mean score difference (%)\n(Case-enriched − Control-enriched)')
    ax.set_title('a', fontweight='bold', loc='left', fontsize=10)

def panel_b_odds_ratio(ax, df):

    AUDIT_RESULTS = [
        {'modality': 'RNA-seq',      'OR': 1.30, 'ci_lower': 1.12, 'ci_upper': 1.51, 'p_value': 0.0007, 'color': COLORS['rna_seq']},
        {'modality': 'CAGE',         'OR': 1.18, 'ci_lower': 1.02, 'ci_upper': 1.36, 'p_value': 0.0256, 'color': COLORS['cage']},
        {'modality': 'ChIP-histone', 'OR': 1.15, 'ci_lower': 1.01, 'ci_upper': 1.31, 'p_value': 0.0297, 'color': COLORS['chip_histone']},
        {'modality': 'DNase',        'OR': 0.98, 'ci_lower': 0.86, 'ci_upper': 1.13, 'p_value': 0.8047, 'color': COLORS['dnase']},
    ]

    results_df = pd.DataFrame(AUDIT_RESULTS)
    y_pos = np.arange(len(results_df))

    for i, row in results_df.iterrows():
        color = row['color'] if row['p_value'] < 0.05 else COLORS['ns']
        ax.errorbar(row['OR'], i, xerr=[[row['OR']-row['ci_lower']],
                   [row['ci_upper']-row['OR']]],
                   fmt='s', color=color, capsize=3, capthick=1, markersize=6)

        ax.text(row['ci_upper'] + 0.03, i,
                f"{row['OR']:.2f} [{row['ci_lower']:.2f}\u2013{row['ci_upper']:.2f}]",
                va='center', fontsize=5.5)

    ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['modality'])
    ax.set_xlabel('Odds Ratio (Top 10% vs Bottom 90%)')
    ax.set_title('b', fontweight='bold', loc='left', fontsize=10)
    ax.set_xlim(0.7, 1.8)

    for _, row in results_df.iterrows():
        print(f"  Panel (b) {row['modality']}: OR={row['OR']:.2f} [{row['ci_lower']:.2f}-{row['ci_upper']:.2f}]")

def panel_c_interaction_ratio(ax, df):

    modalities = ['rna_seq_effect', 'cage_effect', 'chip_histone_effect', 'dnase_effect']
    results = []

    for mod in modalities:
        valid = df[mod].dropna()
        median_val = valid.median()
        high = df[mod] > median_val

        high_case_pct = df[high]['case_enriched'].mean() * 100
        low_case_pct = df[~high]['case_enriched'].mean() * 100

        ir = high_case_pct / low_case_pct if low_case_pct > 0 else np.nan

        np.random.seed(42)
        irs = []
        df_valid = df[df[mod].notna()].copy()
        for _ in range(1000):
            sample = df_valid.sample(n=len(df_valid), replace=True)
            h = sample[mod] > median_val
            h_pct = sample[h]['case_enriched'].mean() * 100
            l_pct = sample[~h]['case_enriched'].mean() * 100
            if l_pct > 0:
                irs.append(h_pct / l_pct)
        ci_lower, ci_upper = np.percentile(irs, [2.5, 97.5])

        results.append({
            'modality': MODALITY_NAMES[mod],
            'IR': ir,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'high_pct': high_case_pct,
            'low_pct': low_case_pct,
            'color': COLORS[mod.replace('_effect', '')]
        })
        print(f"  Panel (c) {MODALITY_NAMES[mod]}: IR={ir:.3f} [{ci_lower:.3f}-{ci_upper:.3f}]")

    results_df = pd.DataFrame(results)

    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, results_df['high_pct'], width,
                   label='High-effect', color=COLORS['case'], alpha=0.8)
    bars2 = ax.bar(x + width/2, results_df['low_pct'], width,
                   label='Low-effect', color=COLORS['ctrl'], alpha=0.8)

    for i, row in results_df.iterrows():
        max_y = max(row['high_pct'], row['low_pct'])
        sig = '*' if row['ci_lower'] > 1 else ''
        ax.text(i, max_y + 2, f"IR={row['IR']:.2f}{sig}", ha='center', fontsize=6)

    ax.set_ylabel('Case-enriched variants (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['modality'], rotation=15, ha='right')
    ax.set_title('c', fontweight='bold', loc='left', fontsize=10)
    ax.legend(loc='lower right', fontsize=6, frameon=True,
              framealpha=0.9, edgecolor='#CCCCCC')
    ax.set_ylim(0, 85)

def panel_d_threshold_stability(ax, df):

    thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    mod = 'rna_seq_effect'

    valid = df[mod].dropna()
    irs = []
    for thresh in thresholds:
        cutoff = np.percentile(valid, thresh)
        high = df[mod] >= cutoff
        high_case_pct = df[high]['case_enriched'].mean() * 100
        low_case_pct = df[~high]['case_enriched'].mean() * 100
        ir = high_case_pct / low_case_pct if low_case_pct > 0 else np.nan
        irs.append(ir)

    ax.plot(thresholds, irs, 'o-', color=COLORS['rna_seq'], linewidth=1.5, markersize=5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.axvspan(30, 70, alpha=0.1, color=COLORS['rna_seq'])

    ax.set_xlabel('Threshold percentile')
    ax.set_ylabel('Interaction Ratio (RNA-seq)')
    ax.set_title('d', fontweight='bold', loc='left', fontsize=10)
    ax.set_xlim(5, 95)
    ax.set_xticks([10, 30, 50, 70, 90])

def main():

    print("Loading data...")
    df = load_data()
    print(f"Variants: {len(df)}")

    fig = plt.figure(figsize=(7.09, 6))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1],
                          hspace=0.4, wspace=0.35,
                          left=0.1, right=0.97, top=0.95, bottom=0.1)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    print("Generating panels...")
    panel_a_mean_difference(ax_a, df.copy())
    panel_b_odds_ratio(ax_b, df.copy())
    panel_c_interaction_ratio(ax_c, df.copy())
    panel_d_threshold_stability(ax_d, df.copy())

    output_dir = '<OUTPUT_DIR>/figures/main'

    png_path = f'{output_dir}/Figure2_ModalityEffects.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png_path}")

    pdf_path = '<OUTPUT_DIR>/figures/pdf/Figure2_ModalityEffects.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")

    plt.close()
    print("\nFigure 2 complete!")

if __name__ == '__main__':
    main()
