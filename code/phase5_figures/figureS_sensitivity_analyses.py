import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

CT_COLORS = {
    'Neuron':     '#00A087',
    'Microglia':  '#E64B35',
    'Astrocyte':  '#4DBBD5',
    'Ubiquitous': '#8C8C8C',
}
CASE_COLOR = '#E64B35'
CTRL_COLOR = '#4DBBD5'

BASE_DIR = '<PROJECT_DIR>'


def panel_a_gene_length(ax):
    df = pd.read_csv(f'{BASE_DIR}/results/reviewer_response/analysis3_length/gene_length_analysis.csv')
    stats_df = pd.read_csv(f'{BASE_DIR}/results/reviewer_response/analysis3_length/summary_statistics.csv')

    reg_coef = stats_df['Regression_length_coef'].iloc[0]
    reg_p = stats_df['Regression_length_pvalue'].iloc[0]
    r2_change = stats_df['R2_change'].iloc[0] * 100
    anova_f_p = stats_df['ANOVA_quartile_p'].iloc[0]

    x = np.log10(df['length'])
    y = df['pct_case_enriched']

    for ct, color in CT_COLORS.items():
        mask = df['Cell_Type'] == ct
        if mask.sum() > 0:
            ax.scatter(x[mask], y[mask], c=color, s=20, alpha=0.7,
                       label=ct, edgecolor='white', linewidth=0.3, zorder=3)

    from numpy.polynomial.polynomial import polyfit
    b, m = polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, b + m * x_line, 'k--', linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_xlabel('Gene length (log$_{10}$ bp)')
    ax.set_ylabel('Case-enriched variants (%)')
    ax.set_title('(a)', fontweight='bold', loc='left', fontsize=10)
    ax.legend(loc='upper right', fontsize=5.5, markerscale=0.7, framealpha=0.8)

    stat_text = (f'Coef = {reg_coef:.2f}, P = {reg_p:.2f}\n'
                 f'R$^2$ change = {r2_change:.2f}%\n'
                 f'ANOVA P = {anova_f_p:.2f}')
    ax.text(0.03, 0.03, stat_text, transform=ax.transAxes,
            fontsize=5.5, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

    print(f"  Panel (a): N={len(df)} genes, Coef={reg_coef:.2f}, P={reg_p:.4f}, R2_change={r2_change:.2f}%")


def panel_b_cohens_d(ax):
    df = pd.read_csv(f'{BASE_DIR}/results/reviewer_response/analysis4_context/alphgenome_cohens_d.csv',
                     index_col=0)

    modalities = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']
    labels = ['RNA-seq', 'CAGE', 'DNase', 'ChIP-histone']
    colors = ['#E64B35', '#4DBBD5', '#F39B7F', '#00A087']

    y_pos = np.arange(len(modalities))
    d_values = [df.loc[m, 'cohen_d'] for m in modalities]

    for i, (d, color) in enumerate(zip(d_values, colors)):
        ax.plot(d, y_pos[i], 'o', color=color, markersize=8, zorder=3,
                markeredgecolor='white', markeredgewidth=0.5)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d")
    ax.set_title('(b)', fontweight='bold', loc='left', fontsize=10)

    max_abs = max(abs(min(d_values)), abs(max(d_values)))
    ax.set_xlim(-max_abs * 2.5, max_abs * 2.5)

    ax.text(0.97, 0.97, 'Individual: negligible',
            transform=ax.transAxes, fontsize=5.5, va='top', ha='right',
            fontstyle='italic', color='#555555')
    ax.text(0.97, 0.88, 'Aggregate: IR = 1.074\nOR = 1.30 [1.12\u20131.52]',
            transform=ax.transAxes, fontsize=5.5, va='top', ha='right',
            fontweight='bold', color='#333333')

    print(f"  Panel (b): Cohen's d values = {dict(zip(labels, [f'{d:.4f}' for d in d_values]))}")


def panel_c_eqtl(ax):
    df = pd.read_csv(f'{BASE_DIR}/results/reviewer_response/analysis1_eqtl/eqtl_enrichment_by_tissue.csv')

    blood = df[df['tissue'].str.contains('Blood')].iloc[0]
    brain = df[df['tissue'].str.contains('Brain')].iloc[0]

    tissues = ['Blood\n(eQTLGen)', 'Brain\n(GTEx V10)']
    case_rates = [blood['case_rate'] * 100, brain['case_rate'] * 100]
    ctrl_rates = [blood['ctrl_rate'] * 100, brain['ctrl_rate'] * 100]

    x = np.arange(len(tissues))
    width = 0.3

    bars_case = ax.bar(x - width/2, case_rates, width, label='Case-enriched',
                       color=CASE_COLOR, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_ctrl = ax.bar(x + width/2, ctrl_rates, width, label='Control-enriched',
                       color=CTRL_COLOR, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tissues)
    ax.set_ylabel('eQTL overlap (%)')
    ax.set_title('(c)', fontweight='bold', loc='left', fontsize=10)
    ax.legend(loc='upper right', fontsize=5.5, framealpha=0.8)

    blood_or = blood['odds_ratio']
    blood_p = blood['fisher_p']
    brain_or = brain['odds_ratio']
    brain_p = brain['fisher_p']

    max_blood = max(case_rates[0], ctrl_rates[0])
    ax.text(0, max_blood + 2.5, f'OR={blood_or:.2f}\nP={blood_p:.1e}',
            ha='center', fontsize=5, va='bottom')

    max_brain = max(case_rates[1], ctrl_rates[1])
    ax.text(1, max_brain + 2.5, f'OR={brain_or:.2f}\nP={brain_p:.2f}',
            ha='center', fontsize=5, va='bottom')

    ax.set_ylim(0, max(max_blood, max_brain) + 12)

    print(f"  Panel (c): Blood case={case_rates[0]:.1f}% ctrl={ctrl_rates[0]:.1f}%, "
          f"Brain case={case_rates[1]:.1f}% ctrl={ctrl_rates[1]:.1f}%")
    print(f"  Panel (c): Blood OR={blood_or:.2f} P={blood_p:.1e}, Brain OR={brain_or:.2f} P={brain_p:.2f}")


def main():
    print("=" * 60)
    print("Figure S: Sensitivity Analyses")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(7.09, 3.0))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.90, wspace=0.45)

    print("\nGenerating panels...")
    panel_a_gene_length(axes[0])
    panel_b_cohens_d(axes[1])
    panel_c_eqtl(axes[2])

    out_supp = f'{BASE_DIR}/manuscript/figures/supplementary'
    out_pdf = f'{BASE_DIR}/manuscript/figures/pdf'

    png_path = f'{out_supp}/FigureS_SensitivityAnalyses.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {png_path}")

    pdf_path = f'{out_pdf}/FigureS_SensitivityAnalyses.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")

    plt.close()
    print("\nFigure S (Sensitivity) complete!")


if __name__ == '__main__':
    main()
