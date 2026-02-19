import os
import pandas as pd
import numpy as np
from scipy import stats

BASE = '<WORK_DIR>'
DATA = os.path.join(BASE, 'data', 'Table_S2_variant_data.csv')
OUT  = os.path.join(BASE, 'analysis', 'manuscript_revision', 'gene_continuous_metric.csv')

df = pd.read_csv(DATA)

modalities = [
    ('rna_seq',       'rna_seq_effect'),
    ('cage',          'cage_effect'),
    ('chip_histone',  'chip_histone_effect'),
    ('dnase',         'dnase_effect'),
]

effect_cols = [col for _, col in modalities]
df = df.dropna(subset=effect_cols)
df['is_CE'] = df['enrichment'].isin(['case_enriched', 'case_only']).astype(int)
print(f'Variants loaded: {len(df):,}')

MIN_VARIANTS = 10
results = []

for gene in sorted(df['gene_name'].unique()):
    gdf = df[df['gene_name'] == gene]
    n = len(gdf)
    if n < MIN_VARIANTS:
        continue

    row = {'gene': gene, 'N': n}
    y = gdf['is_CE'].values

    for mod_short, mod_col in modalities:
        x = gdf[mod_col].values

        if x.std() == 0 or y.std() == 0:
            row[f'spearman_r_{mod_short}'] = np.nan
            row[f'spearman_p_{mod_short}'] = np.nan
            row[f'pb_r_{mod_short}'] = np.nan
            row[f'pb_p_{mod_short}'] = np.nan
            continue

        rho, p = stats.spearmanr(x, y)
        row[f'spearman_r_{mod_short}'] = round(rho, 4)
        row[f'spearman_p_{mod_short}'] = p

        pb_r, pb_p = stats.pointbiserialr(y, x)
        row[f'pb_r_{mod_short}'] = round(pb_r, 4)
        row[f'pb_p_{mod_short}'] = pb_p

    results.append(row)

res = pd.DataFrame(results)

sig_cols = [f'spearman_p_{m}' for m, _ in modalities]
for m, _ in modalities:
    res[f'sig_{m}'] = res[f'spearman_p_{m}'] < 0.05
res['n_sig_modalities'] = res[[f'sig_{m}' for m, _ in modalities]].sum(axis=1).astype(int)
res = res.drop(columns=[f'sig_{m}' for m, _ in modalities])

save_cols = ['gene', 'N']
for m, _ in modalities:
    save_cols += [f'spearman_r_{m}', f'spearman_p_{m}', f'pb_r_{m}', f'pb_p_{m}']
save_cols.append('n_sig_modalities')

res_sorted = res[save_cols].sort_values('spearman_r_rna_seq', ascending=False, na_position='last')
res_sorted.to_csv(OUT, index=False)
print(f'Saved: {OUT}')
print(f'Total genes (N≥{MIN_VARIANTS}): {len(res)}')

print('\n' + '=' * 70)
for mod_short, _ in modalities:
    col = f'spearman_r_{mod_short}'
    valid = res[col].dropna()
    p_col = f'spearman_p_{mod_short}'
    n_sig = (res[p_col] < 0.05).sum()
    print(f'\n--- {mod_short} Spearman r (N={len(valid)} genes) ---')
    print(f'  Range:    [{valid.min():.4f}, {valid.max():.4f}]')
    print(f'  Mean:     {valid.mean():.4f}')
    print(f'  Positive: {(valid > 0).sum()}/{len(valid)} ({(valid > 0).mean()*100:.1f}%)')
    print(f'  Negative: {(valid < 0).sum()}/{len(valid)} ({(valid < 0).mean()*100:.1f}%)')
    print(f'  P < 0.05: {n_sig} genes')

print(f'\n--- Cross-modality consistency ---')
for k in range(5):
    n_k = (res['n_sig_modalities'] == k).sum()
    if n_k > 0:
        print(f'  {k} modalities significant: {n_k} genes')
        if k >= 2:
            genes = res.loc[res['n_sig_modalities'] == k, 'gene'].tolist()
            print(f'    → {genes}')
