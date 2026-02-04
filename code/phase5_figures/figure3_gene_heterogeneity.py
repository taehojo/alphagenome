import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
    'reversed': '#7E6148',
    'neutral': '#B09C85',
    'enhanced': '#00A087',
    'highlight_rev': '#E64B35',
    'highlight_enh': '#3C5488',
    'case': '#E64B35',
    'ctrl': '#4DBBD5',
}

def load_data():
    ir_df = pd.read_csv('<RESULTS_DIR>/all_gene_ir.csv')
    ir_df = ir_df.sort_values('IR', ascending=True).reset_index(drop=True)

    ir_df['category'] = 'neutral'
    ir_df.loc[ir_df['IR'] < 0.7, 'category'] = 'reversed'
    ir_df.loc[ir_df['IR'] >= 1.5, 'category'] = 'enhanced'

    return ir_df

def panel_a_distribution(ax, ir_df):

    bins = np.arange(0.3, 3.5, 0.15)
    n, bins_out, patches = ax.hist(ir_df['IR'], bins=bins, color='#B09C85',
                                    edgecolor='white', linewidth=0.5, alpha=0.8)

    for patch, left_edge in zip(patches, bins_out[:-1]):
        if left_edge < 0.7:
            patch.set_facecolor(COLORS['reversed'])
        elif left_edge >= 1.5:
            patch.set_facecolor(COLORS['enhanced'])

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(x=0.7, color=COLORS['reversed'], linestyle=':', linewidth=0.8, alpha=0.7)
    ax.axvline(x=1.5, color=COLORS['enhanced'], linestyle=':', linewidth=0.8, alpha=0.7)

    ymax = ax.get_ylim()[1]
    ax.text(0.45, ymax * 0.92, 'Reversed\n(IR<0.7)', fontsize=6,
            color=COLORS['reversed'], ha='center', fontweight='bold')
    ax.text(2.8, ymax * 0.92, 'Enhanced\n(IR\u22651.5)', fontsize=6,
            color=COLORS['enhanced'], ha='center', fontweight='bold')

    ax.set_xlabel('Interaction Ratio (IR)')
    ax.set_ylabel('Number of genes')
    ax.set_title('a', fontweight='bold', loc='left', fontsize=10)
    ax.set_xlim(0.2, 3.5)

def panel_b_lollipop(ax, ir_df):

    for i, row in ir_df.iterrows():
        color = COLORS[row['category']]
        ax.hlines(y=i, xmin=1, xmax=row['IR'], color=color, linewidth=0.8, alpha=0.7)
        ax.scatter(row['IR'], i, color=color, s=15, zorder=3)

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    label_offsets = {
        'APH1B':    (-3, -8),
        'CASP7':    (-3, 0),
        'CD2AP':    (-3, 8),
        'LILRB2':   (3, -6),
        'TREML2':   (3, 0),
        'SIGLEC11': (3, 0),
    }
    highlight_genes = ['APH1B', 'CASP7', 'CD2AP', 'SIGLEC11', 'TREML2', 'LILRB2']
    for i, row in ir_df.iterrows():
        if row['gene'] in highlight_genes:
            ofs = label_offsets[row['gene']]
            ha = 'right' if row['category'] == 'reversed' else 'left'
            color = COLORS['reversed'] if row['category'] == 'reversed' else COLORS['enhanced']
            ax.annotate(row['gene'], xy=(row['IR'], i), xytext=ofs,
                       textcoords='offset points', fontsize=5.5, va='center', ha=ha,
                       fontweight='bold', color=color)

    ax.set_xlabel('Interaction Ratio (IR)')
    ax.set_ylabel('Genes (ranked by IR)')
    ax.set_title('b', fontweight='bold', loc='left', fontsize=10)
    ax.set_xlim(0.2, 3.5)
    ax.set_ylim(-1, len(ir_df))
    ax.set_yticks([0, 20, 40, 64])

    n_neutral = len(ir_df[ir_df['category'] == 'neutral'])
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['reversed'],
               markersize=5, label=f'Reversed (n=3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['neutral'],
               markersize=5, label=f'Neutral (n={n_neutral})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['enhanced'],
               markersize=5, label=f'Enhanced (n=3)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=6, frameon=True,
              framealpha=0.9, edgecolor='#CCCCCC')

def panel_c_extreme_genes(ax, ir_df):

    reversed_genes = ir_df[ir_df['category'] == 'reversed'].sort_values('IR')
    enhanced_genes = ir_df[ir_df['category'] == 'enhanced'].sort_values('IR', ascending=False)

    genes_to_plot = pd.concat([reversed_genes, enhanced_genes])

    x = np.arange(len(genes_to_plot))
    width = 0.35

    high_pct = genes_to_plot['high_pct'].values
    low_pct = genes_to_plot['low_pct'].values

    bars1 = ax.bar(x - width/2, high_pct, width, label='High-effect variants',
                   color=COLORS['case'], alpha=0.8)
    bars2 = ax.bar(x + width/2, low_pct, width, label='Low-effect variants',
                   color=COLORS['ctrl'], alpha=0.8)

    ax.axvline(x=2.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_ylabel('Case-enriched variants (%)')
    ax.set_xlabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(genes_to_plot['gene'], rotation=45, ha='right', fontsize=7)
    ax.set_title('c', fontweight='bold', loc='left', fontsize=10)
    ax.set_ylim(0, 115)

    ax.text(1, 108, 'Reversed', fontsize=6.5, ha='center',
            color=COLORS['reversed'], fontweight='bold')
    ax.text(4, 108, 'Enhanced', fontsize=6.5, ha='center',
            color=COLORS['enhanced'], fontweight='bold')

    ax.legend(loc='upper center', fontsize=6, frameon=True,
              framealpha=0.9, edgecolor='#CCCCCC', ncol=2)

def panel_d_mechanism(ax, ir_df):

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('d', fontweight='bold', loc='left', fontsize=10)

    ax.text(5, 9.5, 'Gene-Specific Regulatory Patterns', fontsize=8,
            ha='center', fontweight='bold')

    rect1 = mpatches.FancyBboxPatch((0.5, 5), 4, 3.5, boxstyle="round,pad=0.1",
                                     facecolor=COLORS['reversed'], alpha=0.2,
                                     edgecolor=COLORS['reversed'], linewidth=1)
    ax.add_patch(rect1)
    ax.text(2.5, 8, 'REVERSED (IR < 0.7)', fontsize=7, ha='center',
            fontweight='bold', color=COLORS['reversed'])
    ax.text(2.5, 7.2, 'APH1B, CASP7, CD2AP', fontsize=6, ha='center', style='italic')
    ax.text(2.5, 6.3, 'High-effect → Control-enriched\nLow-effect → Case-enriched',
            fontsize=6, ha='center', linespacing=1.5)

    rect2 = mpatches.FancyBboxPatch((5.5, 5), 4, 3.5, boxstyle="round,pad=0.1",
                                     facecolor=COLORS['enhanced'], alpha=0.2,
                                     edgecolor=COLORS['enhanced'], linewidth=1)
    ax.add_patch(rect2)
    ax.text(7.5, 8, 'ENHANCED (IR ≥ 1.5)', fontsize=7, ha='center',
            fontweight='bold', color=COLORS['enhanced'])
    ax.text(7.5, 7.2, 'SIGLEC11, TREML2, LILRB2', fontsize=6, ha='center', style='italic')
    ax.text(7.5, 6.3, 'High-effect → Case-enriched\nLow-effect → Control-enriched',
            fontsize=6, ha='center', linespacing=1.5)

    rect3 = mpatches.FancyBboxPatch((1, 0.5), 8, 3.5, boxstyle="round,pad=0.1",
                                     facecolor='#F0F0F0', alpha=0.8,
                                     edgecolor='gray', linewidth=1)
    ax.add_patch(rect3)
    ax.text(5, 3.5, 'Biological Implication', fontsize=7, ha='center', fontweight='bold')
    ax.text(5, 2.5, '7.5-fold range in IR demonstrates that regulatory\nimpact interpretation must be gene-specific.',
            fontsize=6, ha='center', linespacing=1.5)
    ax.text(5, 1.2, 'Universal threshold-based prioritization is not appropriate.',
            fontsize=6, ha='center', style='italic', color='#666666')

def main():

    print("Loading data...")
    ir_df = load_data()

    print(f"Total genes: {len(ir_df)}")
    print(f"IR range: {ir_df['IR'].min():.2f} - {ir_df['IR'].max():.2f}")
    print(f"Reversed (IR < 0.7): {(ir_df['category'] == 'reversed').sum()}")
    print(f"Enhanced (IR >= 1.5): {(ir_df['category'] == 'enhanced').sum()}")

    fig = plt.figure(figsize=(7.09, 5.5))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1],
                          hspace=0.40, wspace=0.30,
                          left=0.08, right=0.97, top=0.96, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    print("Generating panels...")
    panel_a_distribution(ax_a, ir_df)
    panel_b_lollipop(ax_b, ir_df)
    panel_c_extreme_genes(ax_c, ir_df)

    output_dir = '<OUTPUT_DIR>/figures/main'

    png_path = f'{output_dir}/Figure3_GeneHeterogeneity.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png_path}")

    pdf_path = '<OUTPUT_DIR>/figures/pdf/Figure3_GeneHeterogeneity.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")

    plt.close()
    print("\nFigure 3 complete!")

if __name__ == '__main__':
    main()
