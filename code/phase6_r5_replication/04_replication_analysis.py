import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os

OUT_DIR = '<WORK_DIR>/analysis/r5_replication'

AG_MODALITIES = [
    'rna_seq_effect', 'cage_effect', 'procap_effect',
    'splice_sites_effect', 'splice_site_usage_effect', 'splice_junctions_effect',
    'atac_effect', 'dnase_effect', 'chip_histone_effect',
    'chip_tf_effect', 'contact_maps_effect'
]

SIGNIFICANT_MODALITIES = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']

print(f"{'='*70}")
print(f"Prompt 4: R5-only Replication Analysis")
print(f"Started: {datetime.now()}")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("Step 1: Loading data")
print(f"{'='*70}")

r5_ag = pd.read_csv(f'{OUT_DIR}/R5_only_variants_AG_matched.tsv', sep='\t')
print(f"R5-only AG-matched variants: {len(r5_ag):,}")

r5_all = pd.read_csv(f'{OUT_DIR}/R5_only_variants_AC3.tsv', sep='\t')
print(f"R5-only all variants: {len(r5_all):,}")

r4 = pd.read_csv('<WORK_DIR>/data/variant_cc_with_alphgenome.csv')
r4['total_AC'] = r4['case_AC'] + r4['ctrl_AC']
r4_filt = r4[r4['total_AC'] >= 3].sort_values('total_AC', ascending=False).drop_duplicates('variant_id', keep='first')
print(f"R4 variants (AC>=3, unique): {len(r4_filt):,}")

r4_ir = pd.read_csv('<WORK_DIR>/results/final_valid_audit_20260202/all_gene_ir.csv')
if 'gene' in r4_ir.columns and 'gene_name' not in r4_ir.columns:
    r4_ir = r4_ir.rename(columns={'gene': 'gene_name'})
print(f"R4 gene IR values: {len(r4_ir):,}")
print(f"R4 IR columns: {list(r4_ir.columns)}")

print(f"\n{'='*70}")
print("Step 2: Calculating Interaction Ratio (IR) for R5-only")
print(f"{'='*70}")

def calculate_ir(df, modality, threshold_pct=80):
    mod_vals = df[modality].dropna()
    if len(mod_vals) < 20:
        return np.nan, np.nan, np.nan, 0, 0

    threshold = mod_vals.quantile(threshold_pct / 100)

    high = df[df[modality] >= threshold] if threshold > 0 else df[df[modality] > 0]
    low = df[df[modality] < threshold] if threshold > 0 else df[df[modality] == 0]

    if len(high) < 5 or len(low) < 5:
        return np.nan, np.nan, np.nan, len(high), len(low)

    high_ce_pct = (high['enrichment'] == 'case_enriched').sum() / len(high) * 100
    low_ce_pct = (low['enrichment'] == 'case_enriched').sum() / len(low) * 100

    ir = high_ce_pct / low_ce_pct if low_ce_pct > 0 else np.inf

    return ir, high_ce_pct, low_ce_pct, len(high), len(low)


print(f"\n--- Overall IR (all AG-matched variants) ---")
print(f"{'Modality':30s} {'IR':>7s} {'High%CE':>8s} {'Low%CE':>8s} {'N_high':>7s} {'N_low':>7s}")
print("-" * 70)

overall_ir = {}
for mod in AG_MODALITIES:
    ir, high_pct, low_pct, n_high, n_low = calculate_ir(r5_ag, mod)
    overall_ir[mod] = ir
    ir_str = f"{ir:.3f}" if not np.isnan(ir) and ir != np.inf else "N/A"
    high_str = f"{high_pct:.1f}%" if not np.isnan(high_pct) else "N/A"
    low_str = f"{low_pct:.1f}%" if not np.isnan(low_pct) else "N/A"
    print(f"{mod:30s} {ir_str:>7s} {high_str:>8s} {low_str:>8s} {n_high:>7,} {n_low:>7,}")

print(f"\n{'='*70}")
print("Step 3: Gene-level IR calculation")
print(f"{'='*70}")

gene_ir_results = []

for gene in r5_ag['gene_name'].unique():
    gene_data = r5_ag[r5_ag['gene_name'] == gene]
    n_variants = len(gene_data)
    n_ce = (gene_data['enrichment'] == 'case_enriched').sum()
    pct_ce = n_ce / n_variants * 100 if n_variants > 0 else 0

    row = {
        'gene_name': gene,
        'n_variants': n_variants,
        'n_case_enriched': n_ce,
        'pct_case_enriched': round(pct_ce, 1)
    }

    for mod in SIGNIFICANT_MODALITIES:
        ir, high_pct, low_pct, n_h, n_l = calculate_ir(gene_data, mod)
        row[f'{mod}_IR'] = ir
        row[f'{mod}_high_pct'] = high_pct
        row[f'{mod}_low_pct'] = low_pct

    gene_ir_results.append(row)

gene_ir_df = pd.DataFrame(gene_ir_results)
gene_ir_df = gene_ir_df.sort_values('n_variants', ascending=False)

print(f"\n{'Gene':15s} {'N':>5s} {'%CE':>6s} {'RNA_IR':>7s} {'CAGE_IR':>8s} {'DNASE_IR':>9s} {'CHIPH_IR':>9s}")
print("-" * 65)
for _, g in gene_ir_df.head(20).iterrows():
    rna_ir = f"{g['rna_seq_effect_IR']:.3f}" if pd.notna(g['rna_seq_effect_IR']) and g['rna_seq_effect_IR'] != np.inf else "N/A"
    cage_ir = f"{g['cage_effect_IR']:.3f}" if pd.notna(g['cage_effect_IR']) and g['cage_effect_IR'] != np.inf else "N/A"
    dnase_ir = f"{g['dnase_effect_IR']:.3f}" if pd.notna(g['dnase_effect_IR']) and g['dnase_effect_IR'] != np.inf else "N/A"
    chiph_ir = f"{g['chip_histone_effect_IR']:.3f}" if pd.notna(g['chip_histone_effect_IR']) and g['chip_histone_effect_IR'] != np.inf else "N/A"
    print(f"{g['gene_name']:15s} {g['n_variants']:>5,} {g['pct_case_enriched']:>5.1f}% {rna_ir:>7s} {cage_ir:>8s} {dnase_ir:>9s} {chiph_ir:>9s}")

gene_ir_df.to_csv(f'{OUT_DIR}/R5_only_gene_IR.tsv', sep='\t', index=False)

print(f"\n{'='*70}")
print("Step 4: R4 vs R5-only IR comparison")
print(f"{'='*70}")

r4_ir_cols = [c for c in r4_ir.columns if '_IR' in c or c == 'gene_name']
r5_ir_cols = ['gene_name'] + [c for c in gene_ir_df.columns if '_IR' in c]

ir_comparison = r4_ir[['gene_name']].merge(
    gene_ir_df[['gene_name', 'n_variants'] + [c for c in gene_ir_df.columns if '_IR' in c]],
    on='gene_name', how='inner', suffixes=('', '_r5')
)

r4_rna_col = [c for c in r4_ir.columns if 'rna' in c.lower() and 'ir' in c.lower()]
r4_cage_col = [c for c in r4_ir.columns if 'cage' in c.lower() and 'ir' in c.lower()]
r4_dnase_col = [c for c in r4_ir.columns if 'dnase' in c.lower() and 'ir' in c.lower()]
r4_chiph_col = [c for c in r4_ir.columns if 'chip' in c.lower() and 'histone' in c.lower() and 'ir' in c.lower()]

print(f"R4 IR columns: {list(r4_ir.columns)}")
print(f"  RNA columns: {r4_rna_col}")

print(f"\n--- Spearman Correlation: R4 IR vs R5 IR ---")

print(f"\nR4 IR first 3 rows:")
print(r4_ir.head(3).to_string())

print(f"\n--- Recalculating R4 gene-level IR for comparison ---")

r4_gene_ir = []
for gene in r4_filt['gene_name'].unique():
    gene_data = r4_filt[r4_filt['gene_name'] == gene]
    n_v = len(gene_data)
    n_ce = (gene_data['enrichment'] == 'case_enriched').sum()

    row = {'gene_name': gene, 'r4_n_variants': n_v, 'r4_pct_ce': round(n_ce/n_v*100, 1) if n_v > 0 else 0}

    for mod in SIGNIFICANT_MODALITIES:
        if mod in gene_data.columns:
            ir, _, _, _, _ = calculate_ir(gene_data, mod)
            row[f'r4_{mod}_IR'] = ir

    r4_gene_ir.append(row)

r4_gene_ir_df = pd.DataFrame(r4_gene_ir)

ir_merged = r4_gene_ir_df.merge(
    gene_ir_df[['gene_name', 'n_variants', 'pct_case_enriched'] +
               [c for c in gene_ir_df.columns if '_IR' in c]],
    on='gene_name', how='inner'
)
ir_merged = ir_merged.rename(columns={'n_variants': 'r5_n_variants', 'pct_case_enriched': 'r5_pct_ce'})

print(f"\nGenes in both R4 and R5: {len(ir_merged)}")

print(f"\n{'Modality':20s} {'Spearman r':>11s} {'P-value':>12s} {'N genes':>8s} {'Direction':>10s}")
print("-" * 65)

corr_results = []
for mod in SIGNIFICANT_MODALITIES:
    r4_col = f'r4_{mod}_IR'
    r5_col = f'{mod}_IR'

    if r4_col in ir_merged.columns and r5_col in ir_merged.columns:
        valid = ir_merged[[r4_col, r5_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) >= 5:
            r, p = stats.spearmanr(valid[r4_col], valid[r5_col])
            direction = "concordant" if r > 0 else "discordant"
            print(f"{mod:20s} {r:>11.3f} {p:>12.2e} {len(valid):>8d} {direction:>10s}")
            corr_results.append({'modality': mod, 'spearman_r': r, 'p_value': p, 'n_genes': len(valid), 'direction': direction})
        else:
            print(f"{mod:20s} {'N/A':>11s} {'N/A':>12s} {len(valid):>8d}")

valid_ce = ir_merged[['r4_pct_ce', 'r5_pct_ce']].dropna()
if len(valid_ce) >= 5:
    r_ce, p_ce = stats.spearmanr(valid_ce['r4_pct_ce'], valid_ce['r5_pct_ce'])
    print(f"\n{'CE%':20s} {r_ce:>11.3f} {p_ce:>12.2e} {len(valid_ce):>8d} {'concordant' if r_ce > 0 else 'discordant':>10s}")

print(f"\n{'='*70}")
print("Step 5: CC_ratio-based replication (all variants)")
print(f"{'='*70}")

r5_gene_cc = r5_all.groupby('gene_name').agg(
    r5_n_variants=('variant_id', 'count'),
    r5_n_ce=('enrichment', lambda x: (x == 'case_enriched').sum()),
    r5_mean_cc=('cc_ratio', lambda x: x[(x != np.inf) & (x.notna())].mean()),
    r5_median_cc=('cc_ratio', lambda x: x[(x != np.inf) & (x.notna())].median())
).reset_index()
r5_gene_cc['r5_pct_ce'] = (r5_gene_cc['r5_n_ce'] / r5_gene_cc['r5_n_variants'] * 100).round(1)

r4_gene_cc = r4_filt.groupby('gene_name').agg(
    r4_n_variants=('variant_id', 'count'),
    r4_n_ce=('enrichment', lambda x: (x == 'case_enriched').sum()),
    r4_mean_cc=('cc_ratio', lambda x: x[(x != np.inf) & (x.notna())].mean()),
    r4_median_cc=('cc_ratio', lambda x: x[(x != np.inf) & (x.notna())].median())
).reset_index()
r4_gene_cc['r4_pct_ce'] = (r4_gene_cc['r4_n_ce'] / r4_gene_cc['r4_n_variants'] * 100).round(1)

cc_comparison = r4_gene_cc.merge(r5_gene_cc, on='gene_name', how='inner')

print(f"\n--- Gene-level Correlations (R4 vs R5-only, all variants) ---")
print(f"Genes in both: {len(cc_comparison)}")

for metric, r4_col, r5_col in [
    ('CE%', 'r4_pct_ce', 'r5_pct_ce'),
    ('Mean CC_ratio', 'r4_mean_cc', 'r5_mean_cc'),
    ('Median CC_ratio', 'r4_median_cc', 'r5_median_cc'),
]:
    valid = cc_comparison[[r4_col, r5_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) >= 5:
        r, p = stats.spearmanr(valid[r4_col], valid[r5_col])
        print(f"  {metric:20s}: r={r:.3f}, p={p:.2e} (n={len(valid)})")

print(f"\n--- Top R4 Genes and R5 Replication ---")
print(f"{'Gene':15s} {'R4 %CE':>7s} {'R5 %CE':>7s} {'R4 MedCC':>9s} {'R5 MedCC':>9s} {'Replicated':>11s}")
print("-" * 65)

cc_comparison = cc_comparison.sort_values('r4_pct_ce', ascending=False)
replicated_count = 0
for _, g in cc_comparison.head(30).iterrows():
    replicated = "Yes" if g['r5_pct_ce'] > 50 else "Partial" if g['r5_pct_ce'] > 40 else "No"
    if g['r5_pct_ce'] > 50:
        replicated_count += 1
    r4_med = f"{g['r4_median_cc']:.3f}" if pd.notna(g['r4_median_cc']) else "N/A"
    r5_med = f"{g['r5_median_cc']:.3f}" if pd.notna(g['r5_median_cc']) else "N/A"
    print(f"{g['gene_name']:15s} {g['r4_pct_ce']:>6.1f}% {g['r5_pct_ce']:>6.1f}% {r4_med:>9s} {r5_med:>9s} {replicated:>11s}")

r5_above_50 = (cc_comparison['r5_pct_ce'] > 50).sum()
r5_above_45 = (cc_comparison['r5_pct_ce'] > 45).sum()
print(f"\nReplication summary:")
print(f"  Genes with R5 CE% > 50%: {r5_above_50} / {len(cc_comparison)} ({r5_above_50/len(cc_comparison)*100:.1f}%)")
print(f"  Genes with R5 CE% > 45%: {r5_above_45} / {len(cc_comparison)} ({r5_above_45/len(cc_comparison)*100:.1f}%)")

cc_comparison.to_csv(f'{OUT_DIR}/R4_vs_R5_CC_ratio_comparison.tsv', sep='\t', index=False)

print(f"\n{'='*70}")
print("Step 6: Population-stratified replication")
print(f"{'='*70}")

pop_replication = {}
for pop in ['NHW', 'AA', 'Hispanic', 'Asian', 'Other']:
    pop_file = f'{OUT_DIR}/R5_only_{pop}_variants_AC3.tsv'
    if not os.path.exists(pop_file):
        continue

    pop_df = pd.read_csv(pop_file, sep='\t')
    n_v = len(pop_df)
    ce = (pop_df['enrichment'] == 'case_enriched').sum()
    pct_ce = ce / n_v * 100 if n_v > 0 else 0

    pop_gene = pop_df.groupby('gene_name').agg(
        n_var=('variant_id', 'count'),
        n_ce=('enrichment', lambda x: (x == 'case_enriched').sum())
    ).reset_index()
    pop_gene['pct_ce'] = (pop_gene['n_ce'] / pop_gene['n_var'] * 100).round(1)
    genes_above_50 = (pop_gene['pct_ce'] > 50).sum()

    pop_replication[pop] = {
        'n_variants': n_v,
        'pct_ce': round(pct_ce, 1),
        'n_genes': pop_gene['gene_name'].nunique(),
        'genes_ce_above_50': genes_above_50
    }

    print(f"\n  {pop}: {n_v:,} variants, CE={pct_ce:.1f}%, Genes CE>50%: {genes_above_50}")

print(f"\n{'='*70}")
print("Step 7: Saving output files")
print(f"{'='*70}")

if corr_results:
    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(f'{OUT_DIR}/R4_vs_R5_IR_correlation.tsv', sep='\t', index=False)
    print(f"Saved: R4_vs_R5_IR_correlation.tsv")

if len(ir_merged) > 0:
    ir_merged.to_csv(f'{OUT_DIR}/R4_vs_R5_gene_IR_merged.tsv', sep='\t', index=False)
    print(f"Saved: R4_vs_R5_gene_IR_merged.tsv ({len(ir_merged)} genes)")

print(f"Saved: R5_only_gene_IR.tsv ({len(gene_ir_df)} genes)")
print(f"Saved: R4_vs_R5_CC_ratio_comparison.tsv ({len(cc_comparison)} genes)")

import os

print(f"\n{'='*70}")
print("PROMPT 4 COMPLETE")
print(f"{'='*70}")
print(f"R5-only AG-matched variants: {len(r5_ag):,}")
print(f"R5-only all variants: {len(r5_all):,}")
print(f"Gene-level IR correlation (RNA-seq): {'r={:.3f}'.format(corr_results[0]['spearman_r']) if corr_results else 'N/A'}")
print(f"Gene-level CE% correlation: r={r_ce:.3f}, p={p_ce:.2e}")
print(f"\nFinished: {datetime.now()}")
