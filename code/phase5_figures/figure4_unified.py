import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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

CELL_TYPES = {
    'Neuron': ['APP', 'PSEN1', 'PSEN2', 'MAPT', 'BIN1', 'CLU', 'SORL1', 'ANK3',
               'PTK2B', 'ADAM10', 'APH1B', 'FERMT2', 'SLC24A4', 'CASS4', 'PICALM',
               'CD2AP', 'EPHA1'],
    'Microglia': ['TREM2', 'PLCG2', 'ABI3', 'INPP5D', 'SPI1', 'CD33', 'MS4A4A',
                  'MS4A6A', 'CR1', 'PILRA', 'LILRB2', 'TREML2', 'SCIMP', 'CLNK',
                  'BLNK'],
    'Astrocyte': ['CLU', 'APOE', 'ABCA7', 'ABCA1', 'GRN', 'CTSH', 'CTSB'],
    'Ubiquitous': ['ADAM17', 'JAZF1', 'UMAD1', 'RHOH', 'RASGEF1C', 'HS3ST5', 'SNX1',
                   'PLEKHA1', 'WDR81', 'WDR12', 'MINDY2', 'TSPAN14', 'EPDR1', 'NCK2',
                   'TMEM106B', 'SPPL2A', 'EED', 'ACE', 'TPCN1', 'MME', 'ICA1', 'SORT1',
                   'ANKH', 'FOXF1', 'USP6NL', 'IDUA', 'KLF16', 'COX7C', 'SPDYE3',
                   'RBCK1', 'SHARPIN', 'TNIP1', 'TSPOAP1', 'CASP7', 'PRKD3', 'WNT3',
                   'HLA-DQA1', 'MYO15A', 'PRDM7', 'RIN3', 'ADAMTS1', 'IL34', 'DOC2A',
                   'APBB3', 'SIGLEC11', 'BCKDK', 'UNC5CL', 'SEC61G'],
}

CT_COLORS = {
    'Neuron':     '#00A087',
    'Microglia':  '#E64B35',
    'Astrocyte':  '#4DBBD5',
    'Ubiquitous': '#8C8C8C',
}

ANALYSIS_ORDER = ['Microglia', 'Neuron', 'Astrocyte', 'Ubiquitous']


def load_data():
    df = pd.read_csv('<DATA_DIR>/variant_cc_with_alphgenome.csv')

    df = df[df['ag_match_type'].notna() & (df['ag_match_type'] != '')]

    df = df[np.isfinite(df['cc_ratio'])]

    gene_to_ct = {}
    for ct in ['Neuron', 'Microglia', 'Astrocyte', 'Ubiquitous']:
        for gene in CELL_TYPES[ct]:
            if gene not in gene_to_ct:
                gene_to_ct[gene] = ct

    df['cell_type'] = df['gene_name'].map(gene_to_ct).fillna('Ubiquitous')

    return df


def panel_a_cc_ratio(ax, df):
    means, sems, ns = [], [], []

    for ct in ANALYSIS_ORDER:
        vals = df[df['cell_type'] == ct]['cc_ratio'].copy()
        means.append(vals.mean())
        sems.append(vals.std() / np.sqrt(len(vals)))
        ns.append(len(vals))

    x = np.arange(len(ANALYSIS_ORDER))
    bars = ax.bar(x, means, yerr=sems, capsize=3,
                  color=[CT_COLORS[ct] for ct in ANALYSIS_ORDER],
                  alpha=0.85, edgecolor='white', linewidth=0.5)

    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sems[i] + 0.15,
                f'n={n:,}', ha='center', fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ANALYSIS_ORDER, rotation=15, ha='right')
    ax.set_ylabel('Mean CC ratio')
    ax.set_title('(a)', fontweight='bold', loc='left', fontsize=10)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    groups = [df[df['cell_type'] == ct]['cc_ratio'].values for ct in ANALYSIS_ORDER]
    _, p_val = stats.f_oneway(*groups)
    ax.text(0.97, 0.97, f'ANOVA P = {p_val:.2e}',
            transform=ax.transAxes, fontsize=6, va='top', ha='right')

    print(f"  Panel (a) CC ratio means: {dict(zip(ANALYSIS_ORDER, [f'{m:.2f}' for m in means]))}")
    print(f"  Panel (a) ANOVA P = {p_val:.2e}")


def panel_b_rna_effect(ax, df):
    means, sems = [], []

    for ct in ANALYSIS_ORDER:
        vals = df[df['cell_type'] == ct]['rna_seq_effect'].dropna()
        means.append(vals.mean())
        sems.append(vals.std() / np.sqrt(len(vals)))

    x = np.arange(len(ANALYSIS_ORDER))
    bars = ax.bar(x, means, yerr=sems, capsize=3,
                  color=[CT_COLORS[ct] for ct in ANALYSIS_ORDER],
                  alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ANALYSIS_ORDER, rotation=15, ha='right')
    ax.set_ylabel('Mean RNA-seq effect')
    ax.set_title('(b)', fontweight='bold', loc='left', fontsize=10)

    groups = [df[df['cell_type'] == ct]['rna_seq_effect'].dropna().values
              for ct in ANALYSIS_ORDER]
    _, p_val = stats.f_oneway(*groups)
    ax.text(0.97, 0.97, f'ANOVA P = {p_val:.2e}',
            transform=ax.transAxes, fontsize=6, va='top', ha='right')

    print(f"  Panel (b) RNA-seq means: {dict(zip(ANALYSIS_ORDER, [f'{m:.4f}' for m in means]))}")
    print(f"  Panel (b) ANOVA P = {p_val:.2e}")


def panel_c_scatter(ax, df):
    gene_stats = df.groupby(['gene_name', 'cell_type']).agg(
        mean_cc_ratio=('cc_ratio', 'mean'),
        mean_rna_effect=('rna_seq_effect', 'mean'),
        n_variants=('variant_id', 'count')
    ).reset_index()

    gene_stats = gene_stats[
        gene_stats['cell_type'].isin(['Microglia', 'Neuron'])
    ]

    def modified_z_score(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            return np.zeros(len(x))
        return 0.6745 * (x - med) / mad

    mz_cc = modified_z_score(gene_stats['mean_cc_ratio'].values)
    mz_rna = modified_z_score(gene_stats['mean_rna_effect'].values)
    is_outlier = (np.abs(mz_cc) > 3.5) | (np.abs(mz_rna) > 3.5)
    gene_stats = gene_stats.copy()
    gene_stats['is_outlier'] = is_outlier

    outlier_genes = gene_stats[gene_stats['is_outlier']]['gene_name'].tolist()
    inliers = gene_stats[~gene_stats['is_outlier']]

    print(f"  Panel (c) Outliers (|modified Z| > 3.5): {outlier_genes}")

    mic = gene_stats[gene_stats['cell_type'] == 'Microglia']
    neu = gene_stats[gene_stats['cell_type'] == 'Neuron']
    _, p_cc = stats.mannwhitneyu(mic['mean_cc_ratio'], neu['mean_cc_ratio'], alternative='two-sided')
    _, p_rna = stats.mannwhitneyu(mic['mean_rna_effect'], neu['mean_rna_effect'], alternative='two-sided')

    for ct in ['Microglia', 'Neuron']:
        ct_all = gene_stats[gene_stats['cell_type'] == ct]
        x_vals = ct_all['mean_cc_ratio'].values
        y_vals = ct_all['mean_rna_effect'].values
        if len(ct_all) > 2:
            cov = np.cov(x_vals, y_vals)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * 1.5 * np.sqrt(vals)
            ell = Ellipse(xy=(x_vals.mean(), y_vals.mean()), width=w, height=h, angle=angle,
                          facecolor=CT_COLORS[ct], alpha=0.08,
                          edgecolor=CT_COLORS[ct], linewidth=0.8, zorder=1)
            ax.add_patch(ell)

    for ct in ['Microglia', 'Neuron']:
        ct_in = inliers[inliers['cell_type'] == ct]
        ax.scatter(ct_in['mean_cc_ratio'], ct_in['mean_rna_effect'],
                   c=CT_COLORS[ct], s=ct_in['n_variants'] * 0.6 + 12,
                   alpha=0.55, label=ct, edgecolor='none', zorder=3)

    label_offsets = {
        'LILRB2': (3, -10),
    }
    label_va = {
        'LILRB2': 'top',
    }
    for _, row in gene_stats[gene_stats['is_outlier']].iterrows():
        ax.scatter(row['mean_cc_ratio'], row['mean_rna_effect'],
                   c=CT_COLORS[row['cell_type']], s=row['n_variants'] * 0.6 + 12,
                   alpha=0.55, edgecolor='none', zorder=3)
        gene = row['gene_name']
        ofs = label_offsets.get(gene, (3, 3))
        va = label_va.get(gene, 'bottom')
        ax.annotate(gene, xy=(row['mean_cc_ratio'], row['mean_rna_effect']),
                    fontsize=6, color='#666666', ha='left', va=va,
                    xytext=ofs, textcoords='offset points')

    ax.set_xlabel('Mean CC ratio (Quantity)', fontsize=9)
    ax.set_ylabel('Mean RNA-seq effect (Quality)', fontsize=9)
    ax.set_title('(c)', fontweight='bold', loc='left', fontsize=11)
    ax.tick_params(labelsize=8)
    ax.axvline(x=1, color='#CCCCCC', linestyle='-', linewidth=0.4)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CT_COLORS['Microglia'],
               markersize=5, label='Microglia', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CT_COLORS['Neuron'],
               markersize=5, label='Neuron', linestyle='None'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
              frameon=False, handletextpad=0.3)

    print(f"  Panel (c) Mann-Whitney CC ratio P = {p_cc:.4f}")
    print(f"  Panel (c) Mann-Whitney RNA-seq P = {p_rna:.4f}")
    print(f"  Panel (c) N genes = {len(gene_stats)}")


def main():
    print("=" * 60)
    print("Figure 4: Cell Type-Specific Dual Pattern (Unified)")
    print("=" * 60)

    print("\nLoading data...")
    df = load_data()

    print(f"\nTotal variants with AlphaGenome + finite cc_ratio: {len(df)}")
    print("\nCell type distribution:")
    for ct in ANALYSIS_ORDER:
        ct_df = df[df['cell_type'] == ct]
        n_genes = ct_df['gene_name'].nunique()
        print(f"  {ct}: {len(ct_df)} variants, {n_genes} genes")

    fig = plt.figure(figsize=(7.09, 7.5))
    gs = fig.add_gridspec(2, 2,
                          height_ratios=[1, 1.2],
                          hspace=0.45, wspace=0.35,
                          left=0.10, right=0.97, top=0.96, bottom=0.07)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    print("\nGenerating panels...")
    panel_a_cc_ratio(ax_a, df)
    panel_b_rna_effect(ax_b, df)
    panel_c_scatter(ax_c, df)

    out_main = '<OUTPUT_DIR>/figures/main'
    out_pdf = '<OUTPUT_DIR>/figures/pdf'

    png_path = f'{out_main}/Figure4_CellTypePattern.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {png_path}")

    pdf_path = f'{out_pdf}/Figure4_CellTypePattern.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")

    plt.close()
    print("\nFigure 4 complete!")


if __name__ == '__main__':
    main()
