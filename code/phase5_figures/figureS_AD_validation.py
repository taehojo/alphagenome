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

MOD_COLORS = {
    'RNA_SEQ':       '#E64B35',
    'CAGE':          '#4DBBD5',
    'CHIP_HISTONE':  '#00A087',
    'DNASE':         '#F39B7F',
}
MOD_COLORS_LIGHT = {
    'RNA_SEQ':       '#F2A09B',
    'CAGE':          '#A6DDE9',
    'CHIP_HISTONE':  '#80D0C3',
    'DNASE':         '#F9CDBF',
}

MODALITIES = ['RNA_SEQ', 'CAGE', 'CHIP_HISTONE', 'DNASE']
MOD_LABELS = ['RNA-seq', 'CAGE', 'ChIP-histone', 'DNase']

BASE_DIR = '<PROJECT_DIR>'


def panel_a_ir_comparison(ax):
    df = pd.read_csv(f'{BASE_DIR}/analysis/population_stratified_1.8M/IR_comparison_all_populations.csv')

    ad_ir = []
    nonad_ir = []
    ad_p = []
    for mod in MODALITIES:
        ad_row = df[(df['Group'] == 'AD') & (df['Modality'] == mod)].iloc[0]
        nonad_row = df[(df['Group'] == 'Non-AD') & (df['Modality'] == mod)].iloc[0]
        ad_ir.append(ad_row['IR'])
        nonad_ir.append(nonad_row['IR'])
        ad_p.append(ad_row['P_value'])

    x = np.arange(len(MODALITIES))
    width = 0.3

    for i, mod in enumerate(MODALITIES):
        ax.bar(x[i] - width/2, ad_ir[i], width,
               color=MOD_COLORS[mod], alpha=0.9, edgecolor='white', linewidth=0.5,
               label='AD genes' if i == 0 else '')
        ax.bar(x[i] + width/2, nonad_ir[i], width,
               color=MOD_COLORS_LIGHT[mod], alpha=0.9, edgecolor='white', linewidth=0.5,
               label='Non-AD genes' if i == 0 else '')

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(MOD_LABELS, rotation=15, ha='right')
    ax.set_ylabel('Interaction Ratio (IR)')
    ax.set_title('(a)', fontweight='bold', loc='left', fontsize=10)
    ax.legend(loc='upper left', fontsize=5.5, framealpha=0.8)
    ax.set_ylim(0.98, 1.09)

    for i in range(len(MODALITIES)):
        p = ad_p[i]
        p_str = f'P={p:.1e}' if p < 0.01 else f'P={p:.3f}'
        ax.text(x[i] - width/2, ad_ir[i] + 0.002, p_str,
                ha='center', fontsize=4.5, va='bottom')

    ad_n = int(df[df['Group'] == 'AD']['n'].iloc[0])
    nonad_n = int(df[df['Group'] == 'Non-AD']['n'].iloc[0])
    ax.text(0.03, 0.03, f'AD: n={ad_n:,}\nNon-AD: n={nonad_n:,}',
            transform=ax.transAxes, fontsize=5, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

    print(f"  Panel (a): AD IR = {[f'{v:.3f}' for v in ad_ir]}")
    print(f"  Panel (a): Non-AD IR = {[f'{v:.3f}' for v in nonad_ir]}")
    print(f"  Panel (a): AD n={ad_n:,}, Non-AD n={nonad_n:,}")


def panel_b_effect_ratio(ax):
    df = pd.read_csv(f'{BASE_DIR}/analysis/population_stratified_1.8M/IR_comparison_all_populations.csv')

    fold_diffs = []
    for mod in MODALITIES:
        ad_row = df[(df['Group'] == 'AD') & (df['Modality'] == mod)].iloc[0]
        nonad_row = df[(df['Group'] == 'Non-AD') & (df['Modality'] == mod)].iloc[0]
        ad_effect = abs(ad_row['IR'] - 1.0)
        nonad_effect = abs(nonad_row['IR'] - 1.0)
        if nonad_effect > 0:
            fold_diffs.append(ad_effect / nonad_effect)
        else:
            fold_diffs.append(np.nan)

    y_pos = np.arange(len(MODALITIES))

    for i, (fd, mod) in enumerate(zip(fold_diffs, MODALITIES)):
        ax.barh(y_pos[i], fd, height=0.5, color=MOD_COLORS[mod],
                alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.text(fd + 1, y_pos[i], f'{fd:.1f}x', va='center', fontsize=6, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(MOD_LABELS)
    ax.set_xlabel('Effect size ratio (AD / Non-AD)')
    ax.set_title('(b)', fontweight='bold', loc='left', fontsize=10)

    ax.text(0.97, 0.03, '10\u201370 fold larger\nin AD genes',
            transform=ax.transAxes, fontsize=6, va='bottom', ha='right',
            fontstyle='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray', linewidth=0.5))

    print(f"  Panel (b): Fold differences = {dict(zip(MOD_LABELS, [f'{f:.1f}x' for f in fold_diffs]))}")


def panel_c_forest_plot(ax):
    df = pd.read_csv(f'{BASE_DIR}/analysis/population_stratified_1.8M/IR_comparison_pop_stratified.csv')

    ad_data = df[df['Group'] == 'AD'].set_index('Modality')
    nonad_data = df[df['Group'] == 'NonAD'].set_index('Modality')

    y_positions_ad = []
    y_positions_na = []
    y_pos = 0
    y_ticks = []
    y_tick_labels = []

    for i, (mod, label) in enumerate(zip(MODALITIES, MOD_LABELS)):
        y_ad = y_pos
        y_na = y_pos + 0.4
        y_positions_ad.append(y_ad)
        y_positions_na.append(y_na)
        y_ticks.extend([y_ad, y_na])
        y_tick_labels.extend([f'AD', f'Non-AD'])
        y_pos += 1.2

    ad_strs = []
    for i, mod in enumerate(MODALITIES):
        if mod in ad_data.index:
            row = ad_data.loc[mod]
            ax.plot([row['CI_lower'], row['CI_upper']], [y_positions_ad[i]]*2,
                    color=MOD_COLORS[mod], linewidth=2, zorder=2)
            ax.plot(row['IR'], y_positions_ad[i], 'D', color=MOD_COLORS[mod],
                    markersize=6, zorder=3, markeredgecolor='white', markeredgewidth=0.5)
            ad_strs.append(f"{row['IR']:.3f} [{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]")

    nonad_strs = []
    for i, mod in enumerate(MODALITIES):
        if mod in nonad_data.index:
            row = nonad_data.loc[mod]
            ax.plot([row['CI_lower'], row['CI_upper']], [y_positions_na[i]]*2,
                    color=MOD_COLORS_LIGHT[mod], linewidth=2, zorder=2)
            ax.plot(row['IR'], y_positions_na[i], 'o', color=MOD_COLORS_LIGHT[mod],
                    markersize=5, zorder=3, markeredgecolor='white', markeredgewidth=0.5)
            nonad_strs.append(f"{row['IR']:.3f} [{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]")

    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    for i, label in enumerate(MOD_LABELS):
        y_mid = (y_positions_ad[i] + y_positions_na[i]) / 2
        ax.text(-0.02, y_mid, label, transform=ax.get_yaxis_transform(),
                fontsize=6, fontweight='bold', va='center', ha='right')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=5)
    ax.tick_params(axis='y', length=2, pad=1)
    ax.set_xlabel('Interaction Ratio (IR)')
    ax.set_title('(c)', fontweight='bold', loc='left', fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(y_positions_na[-1] + 0.5, y_positions_ad[0] - 0.5)

    ad_n = int(ad_data['n'].iloc[0]) if 'n' in ad_data.columns else 0
    ax.text(0.97, 0.03, f'NHW n={ad_n:,}',
            transform=ax.transAxes, fontsize=5.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

    print(f"  Panel (c): AD IR values = {ad_strs}")
    print(f"  Panel (c): Non-AD IR values = {nonad_strs}")


def main():
    print("=" * 60)
    print("Figure S: AD-Specific Validation")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(7.09, 3.0))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.90, wspace=0.50)

    print("\nGenerating panels...")
    panel_a_ir_comparison(axes[0])
    panel_b_effect_ratio(axes[1])
    panel_c_forest_plot(axes[2])

    out_supp = f'{BASE_DIR}/manuscript/figures/supplementary'
    out_pdf = f'{BASE_DIR}/manuscript/figures/pdf'

    png_path = f'{out_supp}/FigureS_AD_Validation.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {png_path}")

    pdf_path = f'{out_pdf}/FigureS_AD_Validation.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")

    plt.close()
    print("\nFigure S (AD Validation) complete!")


if __name__ == '__main__':
    main()
