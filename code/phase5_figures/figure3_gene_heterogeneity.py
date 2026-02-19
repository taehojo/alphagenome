import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, Rectangle

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
})

RED   = '#C62828'
BLUE  = '#1565C0'
GRAY_FILL = '#D0D0D0'
GRAY_EDGE = '#AAAAAA'
GRAY_TEXT = '#666666'
BG_POS = '#FFF8F8'
BG_NEG = '#F6F8FF'

DATA = '<WORK_DIR>/analysis/manuscript_revision/gene_continuous_metric.csv'
OUT  = '<WORK_DIR>/analysis/manuscript_revision/Figure3_v4'

df = pd.read_csv(DATA)

rna = df[['gene', 'spearman_r_rna_seq', 'spearman_p_rna_seq']].dropna().copy()
rna.columns = ['gene', 'r', 'p']
rna['sig'] = rna['p'] < 0.05
N = len(rna)
n_pos = (rna['r'] > 0).sum()
n_neg = (rna['r'] <= 0).sum()

rna_sorted = rna.sort_values('r', ascending=False).reset_index(drop=True)
top_set = set(rna_sorted.head(10)['gene'])
bot_set = set(rna_sorted.tail(10)['gene'])

for _, row in rna[rna['sig']].iterrows():
    if row['gene'] not in top_set and row['gene'] not in bot_set:
        if row['r'] >= 0:
            top_set.add(row['gene'])
        else:
            bot_set.add(row['gene'])

top_df = rna[rna['gene'].isin(top_set)].sort_values('r', ascending=False)
bot_df = rna[rna['gene'].isin(bot_set)].sort_values('r', ascending=False)
n_top = len(top_df)
n_bot = len(bot_df)
n_mid = N - n_top - n_bot

print(f"Top group: {n_top} genes, Bot group: {n_bot} genes, Middle omitted: {n_mid}")
print(f"Top genes: {list(top_df['gene'])}")
print(f"Bot genes: {list(bot_df['gene'])}")

modalities  = ['rna_seq', 'cage', 'chip_histone', 'dnase']
mod_labels  = ['RNA-seq', 'CAGE', 'ChIP-\nhistone', 'DNase']

sig_any = set()
for mod in modalities:
    p_col = f'spearman_p_{mod}'
    mask = df[p_col].notna() & (df[p_col] < 0.05)
    sig_any.update(df.loc[mask, 'gene'])

heat_genes = sorted(sig_any,
    key=lambda g: df.loc[df['gene']==g, 'spearman_r_rna_seq'].values[0]
                  if len(df.loc[df['gene']==g, 'spearman_r_rna_seq'].dropna()) else 0,
    reverse=True)

heat_r   = np.full((len(heat_genes), 4), np.nan)
heat_sig = np.full((len(heat_genes), 4), False)
for i, gene in enumerate(heat_genes):
    row = df[df['gene'] == gene].iloc[0]
    for j, mod in enumerate(modalities):
        rv = row.get(f'spearman_r_{mod}', np.nan)
        pv = row.get(f'spearman_p_{mod}', np.nan)
        if pd.notna(rv): heat_r[i, j] = rv
        if pd.notna(pv) and pv < 0.05: heat_sig[i, j] = True
n_sig_mods = heat_sig.sum(axis=1)

print(f"Heatmap genes: {len(heat_genes)}")

fig = plt.figure(figsize=(180/25.4, 140/25.4))

gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1.15, 1],
                       height_ratios=[1, 0.75],
                       hspace=0.42, wspace=0.38,
                       left=0.08, right=0.92, top=0.95, bottom=0.10)

ax_a = fig.add_subplot(gs[0, 0])
ax_c = fig.add_subplot(gs[0, 1])
ax_b = fig.add_subplot(gs[1, :])

ax_a.text(-0.14, 1.06, 'a', transform=ax_a.transAxes,
          fontsize=10, fontweight='bold', va='top')

bins = np.arange(-0.50, 0.60, 0.05)
ax_a.hist(rna['r'], bins=bins, color=GRAY_FILL, edgecolor='white',
          linewidth=0.5, zorder=2)

sig_pos_r = rna.loc[(rna['sig']) & (rna['r'] > 0), 'r']
sig_neg_r = rna.loc[(rna['sig']) & (rna['r'] < 0), 'r']
if len(sig_pos_r):
    ax_a.hist(sig_pos_r, bins=bins, color=RED, edgecolor='white',
              linewidth=0.5, alpha=0.9, zorder=3)
if len(sig_neg_r):
    ax_a.hist(sig_neg_r, bins=bins, color=BLUE, edgecolor='white',
              linewidth=0.5, alpha=0.9, zorder=3)

ax_a.axvline(0, color='#333', linewidth=0.6, linestyle='--', zorder=1)
ax_a.set_xlabel('Spearman r (RNA-seq)')
ax_a.set_ylabel('Number of genes')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

ax_a.text(0.96, 0.96,
          f'{n_pos} positive ({n_pos/N*100:.1f}%)\n{n_neg} negative ({n_neg/N*100:.1f}%)',
          transform=ax_a.transAxes, fontsize=5.5, va='top', ha='right',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCC', lw=0.4))

handles = [Patch(fc=RED,       label='Positive (P < 0.05)'),
           Patch(fc=BLUE,      label='Negative (P < 0.05)'),
           Patch(fc=GRAY_FILL, label='n.s.')]
ax_a.legend(handles=handles, fontsize=5.5, loc='upper left', frameon=True,
            edgecolor='#CCC', fancybox=False, handlelength=1, handletextpad=0.3)

ax_b.text(-0.035, 1.13, 'b', transform=ax_b.transAxes,
          fontsize=10, fontweight='bold', va='top')

GAP = 2.5

x_top = np.arange(n_top)
x_bot = np.arange(n_top + GAP, n_top + GAP + n_bot)
total_width = x_bot[-1] + 1

ax_b.axhspan(0, 0.6, color=BG_POS, zorder=0)
ax_b.axhspan(-0.55, 0, color=BG_NEG, zorder=0)

for yg in [-0.4, -0.2, 0.2, 0.4]:
    ax_b.axhline(yg, color='#E8E8E8', linewidth=0.4, zorder=1)

ax_b.axhline(0, color='#555', linewidth=0.6, zorder=2)

bar_w = 0.72
for i, (_, row) in enumerate(top_df.iterrows()):
    fc = RED if row['sig'] else GRAY_FILL
    ec = RED if row['sig'] else GRAY_EDGE
    ax_b.bar(x_top[i], row['r'], width=bar_w, color=fc, edgecolor=ec,
             linewidth=0.4, zorder=3)
    if row['sig']:
        ax_b.text(x_top[i], row['r'] + 0.025, f"{row['r']:.2f}",
                  ha='center', va='bottom', fontsize=4.5, color=RED, fontweight='bold')

for i, (_, row) in enumerate(bot_df.iterrows()):
    fc = BLUE if row['sig'] else GRAY_FILL
    ec = BLUE if row['sig'] else GRAY_EDGE
    ax_b.bar(x_bot[i], row['r'], width=bar_w, color=fc, edgecolor=ec,
             linewidth=0.4, zorder=3)
    if row['sig']:
        ax_b.text(x_bot[i], row['r'] - 0.025, f"{row['r']:.2f}",
                  ha='center', va='top', fontsize=4.5, color=BLUE, fontweight='bold')

gap_cx = n_top + GAP / 2 - 0.5
ax_b.text(gap_cx, 0, f'{n_mid} genes\n(n.s.)',
          ha='center', va='center', fontsize=5.5, style='italic', color=GRAY_TEXT,
          bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#DDD', lw=0.4))

ax_b.axvline(n_top - 0.5 + 0.3, color='#CCC', linewidth=0.5, linestyle=':', zorder=1)
ax_b.axvline(n_top + GAP - 0.5 - 0.3, color='#CCC', linewidth=0.5, linestyle=':', zorder=1)

all_x = np.concatenate([x_top, x_bot])
all_genes = list(top_df['gene']) + list(bot_df['gene'])
all_sig   = list(top_df['sig']) + list(bot_df['sig'])
all_r_val = list(top_df['r'])   + list(bot_df['r'])

ax_b.set_xticks(all_x)
ax_b.set_xticklabels(all_genes, rotation=45, ha='right', fontsize=5.5)

for i, lab in enumerate(ax_b.get_xticklabels()):
    if all_sig[i]:
        lab.set_color(RED if all_r_val[i] > 0 else BLUE)
        lab.set_fontweight('bold')
    else:
        lab.set_color(GRAY_TEXT)

ax_b.set_ylabel('Spearman r\n(RNA-seq)', fontsize=7)
ax_b.set_xlim(-0.8, total_width - 0.2)
ax_b.set_ylim(-0.55, 0.60)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

ax_c.text(-0.20, 1.06, 'c', transform=ax_c.transAxes,
          fontsize=10, fontweight='bold', va='top')

vmax = 0.55
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
heat_masked = np.ma.masked_invalid(heat_r)

im = ax_c.imshow(heat_masked, aspect='auto', cmap=cmap, norm=norm,
                 interpolation='nearest')

for i in range(len(heat_genes)):
    for j in range(4):
        if heat_sig[i, j]:
            ax_c.text(j, i + 0.07, '*', ha='center', va='center',
                      fontsize=9, fontweight='bold', color='white',
                      path_effects=[pe.withStroke(linewidth=0.7, foreground='black')])

for i in range(len(heat_genes)):
    for j in range(4):
        if np.isnan(heat_r[i, j]):
            ax_c.text(j, i, '–', ha='center', va='center',
                      fontsize=5, color='#CCC')

ax_c.set_xticks(range(4))
ax_c.set_xticklabels(mod_labels, fontsize=6, rotation=45, ha='right')
ax_c.set_yticks(range(len(heat_genes)))
ax_c.set_yticklabels(heat_genes, fontsize=5.5)

for i, lab in enumerate(ax_c.get_yticklabels()):
    gene = heat_genes[i]
    rna_row = rna[rna['gene'] == gene]
    if len(rna_row) and rna_row.iloc[0]['p'] < 0.05:
        lab.set_color(RED if rna_row.iloc[0]['r'] > 0 else BLUE)
        lab.set_fontweight('bold')

if 'SORL1' in heat_genes:
    idx = heat_genes.index('SORL1')
    ax_c.add_patch(Rectangle((-0.5, idx - 0.5), 4, 1,
                              fill=False, edgecolor='black', linewidth=1.5, zorder=5))

for i in range(len(heat_genes)):
    if n_sig_mods[i] >= 2:
        ax_c.text(3.65, i, f'({int(n_sig_mods[i])})',
                  fontsize=4.5, va='center', ha='left', color='#333')

cbar = plt.colorbar(im, ax=ax_c, shrink=0.65, pad=0.08, aspect=20)
cbar.set_label('Spearman r', fontsize=6)
cbar.ax.tick_params(labelsize=5, width=0.4, length=2)
cbar.outline.set_linewidth(0.4)

fig.text(0.08, 0.015,
         '* P < 0.05  |  Box = SORL1 (significant in 3 modalities)  |  '
         'Bold gene names = significant in RNA-seq  |  '
         'Parentheses = number of significant modalities',
         fontsize=5, color='#888')

plt.savefig(f'{OUT}.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {OUT}.png / .pdf")
print(f"Figure: 180mm x 140mm, 300 DPI")
