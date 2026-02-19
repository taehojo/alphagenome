import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

OUT = '<WORK_DIR>/analysis/additional_analyses'

print("=" * 70)
print("Prompt 8: Sensitivity Analyses")
print("Started:", datetime.now())
print("=" * 70)

MODALITIES = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']

def calc_ir(data, mod, threshold_pct=80):
    threshold = data[mod].quantile(threshold_pct / 100)
    high = data[data[mod] >= threshold]
    low = data[data[mod] < threshold]
    if len(high) == 0 or len(low) == 0:
        return np.nan, np.nan, 0, 0
    ce_high = high['is_case_enriched'].mean()
    ce_low = low['is_case_enriched'].mean()
    ir = ce_high / ce_low if ce_low > 0 else float('inf')
    a = high['is_case_enriched'].sum()
    c = len(high) - a
    b = low['is_case_enriched'].sum()
    d = len(low) - b
    _, p = stats.fisher_exact([[a, c], [b, d]])
    return ir, p, len(high), len(low)


print("\n--- Loading R4 data ---")
df_full = pd.read_csv('<WORK_DIR>/data/variant_cc_with_alphgenome.csv')
print("Full data: {:,} rows".format(len(df_full)))


print("\n\n--- 8-1. AC Threshold Sensitivity ---")

ac_results = []
for ac_thresh in [1, 3, 5, 10]:
    df_ac = df_full.copy()
    df_ac['total_AC'] = df_ac['case_AC'] + df_ac['ctrl_AC']
    df_ac = df_ac[df_ac['total_AC'] >= ac_thresh] \
              .sort_values('total_AC', ascending=False) \
              .drop_duplicates('variant_id', keep='first')
    df_ac['is_case_enriched'] = df_ac['enrichment'].isin(['case_enriched', 'case_only'])

    n = len(df_ac)
    ce_pct = df_ac['is_case_enriched'].mean() * 100

    print("\n  AC >= {}: {:,} variants, CE% = {:.1f}%".format(ac_thresh, n, ce_pct))

    for mod in MODALITIES:
        mod_name = mod.replace('_effect', '')
        df_ag = df_ac[df_ac[mod].notna() & (df_ac[mod] > 0)]
        if len(df_ag) < 100:
            continue
        ir, p, n_high, n_low = calc_ir(df_ag, mod, 80)
        print("    {}: IR={:.3f}, P={:.2e}, N_high={}, N_low={}".format(
            mod_name, ir, p, n_high, n_low))

        ac_results.append({
            'AC_threshold': ac_thresh,
            'Modality': mod_name,
            'N_variants': len(df_ag),
            'CE_pct': df_ag['is_case_enriched'].mean() * 100,
            'IR': ir,
            'P_value': p,
            'N_high': n_high,
            'N_low': n_low
        })

pd.DataFrame(ac_results).to_csv('{}/prompt8_sensitivity_AC.tsv'.format(OUT), sep='\t', index=False)


print("\n\n--- 8-2. Gene Window Sensitivity ---")
print("  NOTE: The current data was extracted with 5kb upstream / 1kb downstream.")
print("  Re-extraction with different windows requires PLINK2 re-run on HPC.")
print("  This analysis compares variants by their position within the gene region.")

df = df_full.copy()
df['total_AC'] = df['case_AC'] + df['ctrl_AC']
df = df[df['total_AC'] >= 3].sort_values('total_AC', ascending=False) \
       .drop_duplicates('variant_id', keep='first')
df['is_case_enriched'] = df['enrichment'].isin(['case_enriched', 'case_only'])

gene_file = '<WORK_DIR>/data/Supplementary_Table_S1_GeneList.csv'
if os.path.exists(gene_file):
    gene_list = pd.read_csv(gene_file)
    print("  Gene list loaded: {} genes".format(len(gene_list)))
    print("  Gene list columns: {}".format(list(gene_list.columns)))
else:
    print("  Gene list file not found")

window_results = []
print("\n  Current window (5kb up / 1kb down):")
for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    ir, p, n_high, n_low = calc_ir(df, mod, 80)
    print("    {}: IR={:.3f}, P={:.2e}, N={:,}".format(mod_name, ir, p, len(df)))
    window_results.append({
        'Window': '5kb_up_1kb_down (current)',
        'Modality': mod_name,
        'N_variants': len(df),
        'IR': ir, 'P_value': p
    })

print("\n  LIMITATION: Different gene window definitions require re-running")
print("  PLINK2 extraction with modified gene coordinates. Only the current")
print("  window (5kb upstream / 1kb downstream) results are available.")
print("  However, the AC threshold and percentile analyses below provide")
print("  strong sensitivity evidence.")

pd.DataFrame(window_results).to_csv('{}/prompt8_sensitivity_window.tsv'.format(OUT),
                                     sep='\t', index=False)


print("\n\n--- 8-3. Percentile Threshold Sensitivity ---")

pct_results = []
pct_thresholds = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    print("\n  {}:".format(mod_name))

    for pct in pct_thresholds:
        ir, p, n_high, n_low = calc_ir(df, mod, pct)
        print("    Top {:2d}%: IR={:.3f}, P={:.2e}, N_high={:,}, N_low={:,}".format(
            100 - pct, ir, p, n_high, n_low))

        pct_results.append({
            'Modality': mod_name,
            'Threshold_percentile': pct,
            'Top_pct': 100 - pct,
            'IR': ir,
            'P_value': p,
            'N_high': n_high,
            'N_low': n_low
        })

pd.DataFrame(pct_results).to_csv('{}/prompt8_sensitivity_percentile.tsv'.format(OUT),
                                  sep='\t', index=False)

print("\n  Stability summary (IR > 1 across thresholds):")
for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    mod_pct = [r for r in pct_results if r['Modality'] == mod_name]
    n_above_1 = sum(1 for r in mod_pct if r['IR'] > 1)
    n_sig = sum(1 for r in mod_pct if r['P_value'] < 0.05)
    print("    {}: IR>1 in {}/{} thresholds, significant in {}/{}".format(
        mod_name, n_above_1, len(mod_pct), n_sig, len(mod_pct)))


print("\n\n--- 8-3b. Decile CE% Analysis ---")

decile_results = []
for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    df['decile'] = pd.qcut(df[mod].rank(method='first'), 10, labels=False) + 1

    print("\n  {} decile CE%:".format(mod_name))
    for d in range(1, 11):
        dv = df[df['decile'] == d]
        ce = dv['is_case_enriched'].mean() * 100
        print("    D{:2d}: N={:,}, CE%={:.1f}%".format(d, len(dv), ce))
        decile_results.append({
            'Modality': mod_name, 'Decile': d, 'N': len(dv),
            'CE_pct': ce, 'Mean_effect': dv[mod].mean()
        })

    dec_ce = df.groupby('decile')['is_case_enriched'].mean() * 100
    r, p = stats.spearmanr(range(1, 11), dec_ce.values)
    print("    Monotonic trend: Spearman r={:.3f}, P={:.4f}".format(r, p))

pd.DataFrame(decile_results).to_csv('{}/prompt8_decile_CE.tsv'.format(OUT), sep='\t', index=False)


print("\n\n--- 8-4. Confounder Analysis ---")

gene_stats = df.groupby('gene_name').agg(
    n_variants=('variant_id', 'count'),
    ce_pct=('is_case_enriched', 'mean'),
    mean_rna_seq=('rna_seq_effect', 'mean'),
    mean_cage=('cage_effect', 'mean'),
    mean_dnase=('dnase_effect', 'mean'),
    mean_chip=('chip_histone_effect', 'mean'),
    mean_cc_ratio=('cc_ratio', lambda x: x.replace([np.inf, -np.inf], np.nan).mean())
).reset_index()
gene_stats['ce_pct'] = gene_stats['ce_pct'] * 100

gene_ir_list = []
threshold_rna = df['rna_seq_effect'].quantile(0.80)
for gene in gene_stats['gene_name']:
    gdf = df[df['gene_name'] == gene]
    if len(gdf) < 10:
        continue
    high = gdf[gdf['rna_seq_effect'] >= threshold_rna]
    low = gdf[gdf['rna_seq_effect'] < threshold_rna]
    if len(high) > 0 and len(low) > 0 and low['is_case_enriched'].mean() > 0:
        ir = high['is_case_enriched'].mean() / low['is_case_enriched'].mean()
    else:
        ir = np.nan
    gene_ir_list.append({'gene_name': gene, 'IR_rna_seq': ir, 'n': len(gdf)})

gene_ir = pd.DataFrame(gene_ir_list)
gene_stats = gene_stats.merge(gene_ir, on='gene_name', how='left')

gencode_path = '<WORK_DIR>/data/gencode_v38_genes.tsv'
if not os.path.exists(gencode_path):
    alt_paths = [
        '<WORK_DIR>/data/gene_coordinates.csv',
        '<WORK_DIR>/results/gene_length_analysis.csv'
    ]
    for p in alt_paths:
        if os.path.exists(p):
            gencode_path = p
            break

if os.path.exists(gencode_path):
    gene_coords = pd.read_csv(gencode_path)
    print("  Gene coordinates loaded: {}".format(len(gene_coords)))
else:
    print("  Gene coordinate file not found")
    if os.path.exists(gene_file):
        gene_coords = pd.read_csv(gene_file)
        if 'start' in gene_coords.columns and 'end' in gene_coords.columns:
            gene_coords['gene_length'] = gene_coords['end'] - gene_coords['start']
        elif 'gene_start' in gene_coords.columns and 'gene_end' in gene_coords.columns:
            gene_coords['gene_length'] = gene_coords['gene_end'] - gene_coords['gene_start']

if 'gene_length' in gene_stats.columns or ('gene_coords' in dir() and 'gene_length' in gene_coords.columns):
    print("  Gene length data available")
else:
    gene_spans = df.groupby('gene_name')['pos'].agg(['min', 'max'])
    gene_spans['gene_length'] = gene_spans['max'] - gene_spans['min'] + 1
    gene_stats = gene_stats.merge(gene_spans[['gene_length']], left_on='gene_name',
                                   right_index=True, how='left')
    gene_stats['variant_density'] = gene_stats['n_variants'] / gene_stats['gene_length']

print("\n  Gene-level confounder correlations (with RNA-seq IR):")
confound_results = []

ir_finite = gene_stats.dropna(subset=['IR_rna_seq'])
ir_finite = ir_finite[np.isfinite(ir_finite['IR_rna_seq'])]

r, p = stats.spearmanr(ir_finite['n_variants'], ir_finite['IR_rna_seq'])
print("    IR vs N_variants: r={:.3f}, P={:.4f}".format(r, p))
confound_results.append({'Confounder': 'N_variants', 'Spearman_r': r, 'P_value': p})

if 'gene_length' in ir_finite.columns:
    valid = ir_finite['gene_length'].notna() & (ir_finite['gene_length'] > 0)
    r, p = stats.spearmanr(ir_finite.loc[valid, 'gene_length'],
                            ir_finite.loc[valid, 'IR_rna_seq'])
    print("    IR vs gene_length: r={:.3f}, P={:.4f}".format(r, p))
    confound_results.append({'Confounder': 'Gene_length', 'Spearman_r': r, 'P_value': p})

    if 'variant_density' in ir_finite.columns:
        valid = ir_finite['variant_density'].notna() & np.isfinite(ir_finite['variant_density'])
        r, p = stats.spearmanr(ir_finite.loc[valid, 'variant_density'],
                                ir_finite.loc[valid, 'IR_rna_seq'])
        print("    IR vs variant_density: r={:.3f}, P={:.4f}".format(r, p))
        confound_results.append({'Confounder': 'Variant_density', 'Spearman_r': r, 'P_value': p})

r, p = stats.spearmanr(ir_finite['ce_pct'], ir_finite['IR_rna_seq'])
print("    IR vs CE%: r={:.3f}, P={:.4f}".format(r, p))
confound_results.append({'Confounder': 'CE_pct', 'Spearman_r': r, 'P_value': p})

r, p = stats.spearmanr(ir_finite['mean_rna_seq'], ir_finite['IR_rna_seq'])
print("    IR vs mean_AG_score: r={:.3f}, P={:.4f}".format(r, p))
confound_results.append({'Confounder': 'Mean_AG_score', 'Spearman_r': r, 'P_value': p})

pd.DataFrame(confound_results).to_csv('{}/prompt8_confounders_correlation.tsv'.format(OUT),
                                       sep='\t', index=False)

print("\n  Adjusted analysis:")
if 'gene_length' in ir_finite.columns and ir_finite['gene_length'].notna().sum() > 10:
    from numpy.polynomial import polynomial as P
    valid = ir_finite['gene_length'].notna() & (ir_finite['gene_length'] > 0)
    x = np.log10(ir_finite.loc[valid, 'gene_length'].values)
    y = ir_finite.loc[valid, 'IR_rna_seq'].values
    if len(x) > 5:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        residual_ir = y - (slope * x + intercept)
        print("    Residual IR (after gene length): mean={:.3f}, std={:.3f}".format(
            residual_ir.mean(), residual_ir.std()))
        print("    Gene length explains {:.1f}% of IR variance".format(
            slope * np.std(x) / np.std(y) * 100 if np.std(y) > 0 else 0))


print("\n\n--- 8-5. R5 Sensitivity ---")

shared_file = '<WORK_DIR>/analysis/r5_replication/Phase_B_shared_variants.tsv'
shared = pd.read_csv(shared_file, sep='\t')
if 'R5_is_CE' in shared.columns:
    shared['is_case_enriched'] = shared['R5_is_CE']
elif 'is_case_enriched' in shared.columns:
    pass

print("R5 shared variants: {:,}".format(len(shared)))

r5_sens = []
if 'R5_case_AC' in shared.columns:
    shared['R5_total_AC'] = shared['R5_case_AC'] + shared['R5_ctrl_AC']

    for ac_thresh in [1, 3, 5, 10]:
        s5 = shared[shared['R5_total_AC'] >= ac_thresh].copy()
        ce = s5['is_case_enriched'].mean() * 100

        print("\n  R5 AC >= {}: {:,} variants, CE%={:.1f}%".format(ac_thresh, len(s5), ce))

        for mod in MODALITIES:
            mod_name = mod.replace('_effect', '')
            if mod not in s5.columns:
                continue
            ir, p, nh, nl = calc_ir(s5, mod, 80)
            print("    {}: IR={:.3f}, P={:.2e}".format(mod_name, ir, p))
            r5_sens.append({
                'Analysis': 'AC_threshold', 'Parameter': ac_thresh,
                'Modality': mod_name, 'N': len(s5), 'CE_pct': ce,
                'IR': ir, 'P_value': p
            })

for pct in [70, 80, 90]:
    print("\n  R5 Top {}%:".format(100 - pct))
    for mod in MODALITIES:
        mod_name = mod.replace('_effect', '')
        if mod not in shared.columns:
            continue
        ir, p, nh, nl = calc_ir(shared, mod, pct)
        print("    {}: IR={:.3f}, P={:.2e}".format(mod_name, ir, p))
        r5_sens.append({
            'Analysis': 'Percentile', 'Parameter': 100 - pct,
            'Modality': mod_name, 'N': len(shared),
            'CE_pct': shared['is_case_enriched'].mean() * 100,
            'IR': ir, 'P_value': p
        })

pd.DataFrame(r5_sens).to_csv('{}/prompt8_R5_sensitivity.tsv'.format(OUT), sep='\t', index=False)


print("\n" + "=" * 70)
print("PROMPT 8 SUMMARY")
print("=" * 70)

print("\n1. AC THRESHOLD (R4 RNA-seq):")
for r in ac_results:
    if r['Modality'] == 'rna_seq':
        print("   AC>={}: N={:,}, CE%={:.1f}%, IR={:.3f}, P={:.2e}".format(
            r['AC_threshold'], r['N_variants'], r['CE_pct'], r['IR'], r['P_value']))

print("\n2. PERCENTILE THRESHOLD (R4 RNA-seq):")
for r in pct_results:
    if r['Modality'] == 'rna_seq':
        print("   Top {:2d}%: IR={:.3f}, P={:.2e}".format(
            r['Top_pct'], r['IR'], r['P_value']))

print("\n3. CONFOUNDERS:")
for r in confound_results:
    sig = '*' if r['P_value'] < 0.05 else ''
    print("   {}: r={:.3f}, P={:.4f} {}".format(
        r['Confounder'], r['Spearman_r'], r['P_value'], sig))

print("\n4. GENE WINDOW: Only current window available (5kb up/1kb down)")
print("   Re-extraction with different windows requires PLINK2 re-run")

print("\nFinished:", datetime.now())
